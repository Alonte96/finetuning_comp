"""The eval gate must be *disableable*, because a memory run must not evaluate.

bench_memory.py passes ``--eval-every 1000000000`` with the comment "no eval,
this is a memory run". That sentinel only means anything if the training loop
honours it. The old gate was::

    if step % cfg.eval_every == 0 or step == cfg.max_steps:

whose second clause fires *unconditionally* on the final step, so all six memory
configs still ran a full eval pass -- burning paid GPU time and folding eval
activations into a peak-MEMORY measurement. The gate now reads::

    eval_at_end = cfg.eval_every <= cfg.max_steps        # computed before the loop
    ...
    if step % cfg.eval_every == 0 or (step == cfg.max_steps and eval_at_end):

The claims, each tested:
1. eval_every > max_steps (the sentinel) => ZERO evals, and evaluate() is never
   even called -- the saving is skipped compute, not just a missing record;
2. the ``<=`` boundary is exactly right: eval_every == max_steps still evaluates
   (once, at the final step), eval_every == max_steps + 1 does not;
3. a normal periodic setting still evaluates on schedule AND on the final step,
   without double-counting when the two clauses coincide.

Everything runs on the offline smoke path (tiny synthetic LlamaForCausalLM +
synthetic tokens, no network) and writes to a pytest tmp_path, never results/.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import train  # noqa: E402

SENTINEL = 10**9  # what bench_memory.py actually passes for --eval-every


def _argv(out_dir, *, max_steps, eval_every, run_name):
    """The smallest honest smoke run: everything fairness-critical pinned small.

    seq_len/micro_batch/grad_accum/grad_checkpointing are passed explicitly so a
    --preset can never clobber them (presets only fill in flags the caller
    omitted), keeping this test identical on a CUDA box and on a laptop.
    """
    return [
        "--smoke", "true",
        "--method", "full",
        "--dtype", "fp32",
        "--seq-len", "32",
        "--micro-batch", "2",
        "--grad-accum", "1",
        "--eval-examples", "4",
        "--grad-checkpointing", "false",
        "--warmup-steps", "1",
        "--log-every", "100",
        "--max-steps", str(max_steps),
        "--eval-every", str(eval_every),
        "--run-name", run_name,
        "--phase", "mem",
        "--out-dir", str(out_dir),
    ]


def _run(out_dir, monkeypatch, *, max_steps, eval_every, run_name="t"):
    """Run the loop, returning (record, list-of-steps-evaluate()-was-called-at).

    evaluate() is spied on rather than inferred from the record, so the test
    asserts the gate itself -- whether the eval forward passes happened at all.
    """
    calls = []
    real_evaluate = train.evaluate

    def spy(model, eval_loader, device, autocast_ctx):
        calls.append(True)
        return real_evaluate(model, eval_loader, device, autocast_ctx)

    monkeypatch.setattr(train, "evaluate", spy)
    record = train.main(_argv(out_dir, max_steps=max_steps,
                              eval_every=eval_every, run_name=run_name))

    assert record["status"] == "completed", record.get("error")
    # The spy and the record must agree: one appended entry per actual call.
    assert len(calls) == len(record["evals"])
    return record, [ev["step"] for ev in record["evals"]]


def test_sentinel_eval_every_disables_eval_entirely(tmp_path, monkeypatch):
    """The bug, stated directly: --eval-every 1e9 on a 2-step memory run.

    Old gate: step 2 == max_steps => a full eval pass anyway. New gate: none.
    """
    record, steps = _run(tmp_path, monkeypatch, max_steps=2, eval_every=SENTINEL,
                         run_name="mem_sentinel")

    assert steps == [], f"sentinel --eval-every did not disable eval: {steps}"
    assert record["evals"] == []


def test_sentinel_run_records_no_eval_metrics_downstream(tmp_path, monkeypatch):
    """report.py reads the JSONL, not the return value; both must show no eval."""
    from config import read_jsonl

    _run(tmp_path, monkeypatch, max_steps=2, eval_every=SENTINEL, run_name="mem_full")

    out_path = tmp_path / "smoke_runs.jsonl"
    assert out_path.exists(), "smoke run must write to --out-dir, not results/"
    rows = read_jsonl(out_path)
    assert len(rows) == 1
    assert rows[0]["evals"] == []
# improvement
    assert "eval_loss" not in str(rows[0]["evals"])


def test_eval_every_just_above_max_steps_is_the_off_by_one_boundary(tmp_path, monkeypatch):
    """max_steps=2, eval_every=3: nothing is due, and the end is not special.

    This pins ``eval_every <= max_steps`` against the neighbouring ``<``/``>=``
    mistakes -- it is the tightest case, one step away from evaluating.
    """
    _, steps = _run(tmp_path, monkeypatch, max_steps=2, eval_every=3, run_name="boundary_above")
    assert steps == []


def test_eval_every_equal_to_max_steps_evaluates_once_at_the_end(tmp_path, monkeypatch):
    """The other side of the boundary: eval_every == max_steps is NOT the
    sentinel, so the final-step eval must survive the fix."""
    _, steps = _run(tmp_path, monkeypatch, max_steps=2, eval_every=2, run_name="boundary_equal")
    assert steps == [2]


def test_periodic_eval_still_fires_on_schedule_and_at_the_final_step(tmp_path, monkeypatch):
    """max_steps=4, eval_every=2: evals at 2 and 4.

    Step 4 satisfies BOTH clauses of the `or`; exactly two entries proves the
    final-step eval is neither lost nor double-counted.
    """
    _, steps = _run(tmp_path, monkeypatch, max_steps=4, eval_every=2, run_name="periodic")
    assert steps == [2, 4]


def test_final_step_eval_survives_a_non_divisible_eval_every(tmp_path, monkeypatch):
    """max_steps=4, eval_every=3: the schedule fires at 3, and the end-of-run
    eval at 4 is the whole point of the second clause -- the fix must keep it."""
    _, steps = _run(tmp_path, monkeypatch, max_steps=4, eval_every=3, run_name="nondivisible")
    assert steps == [3, 4]
