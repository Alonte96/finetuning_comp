"""The LR sweep must never crown a diverged run, and must accept the run knobs.

Two bugs, both of which would have spent paid GPU time on garbage:

1. winners were picked with ``min(results, key=lambda t: t[1])`` over raw eval
   losses, with no divergence filter. A diverged candidate reports NaN, and NaN
   is unordered: ``min`` keeps its running best unless a later element compares
   strictly smaller, and *every* comparison against NaN is False -- so a NaN
   candidate that comes FIRST in the grid survives as the winner and its LR then
   feeds the 4-6h full runs. The fix drops non-finite losses via math.isfinite
   (and warns when the surviving winner sits on a grid edge).

2. the sweep's argparse knew only --methods/--steps/--smoke, so every knob the
   RUNBOOK documents (--preset, --dtype, --galore-rank, ...) died with
   "unrecognized arguments" -- the sweep could not be run under the same
   settings as the full runs, and a winner found under other settings is not a
   winner. The fix collects leftovers with parse_known_args, validates them
   against RunConfig.build_parser() so a typo fails instantly, rejects the flags
   the sweep sets per candidate, and forwards the rest verbatim to train.main.

Nothing here trains: train.main is replaced by a fake returning synthetic
records, so these are unit tests of the selection and forwarding logic only.
"""

import math
import runpy
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import config  # noqa: E402
import sweep_lr  # noqa: E402

GALORE_GRID = sweep_lr.DEFAULT_LR_GRIDS["galore"]  # [1e-5, 3e-5, 1e-4]
NAN = float("nan")


def _fake_train(outcomes):
    """A stand-in for train.main.

    ``outcomes`` maps an LR to either an eval loss (float, possibly NaN) or a
    non-"completed" status string. Every argv it is handed is recorded, so the
    forwarding tests can inspect exactly what train.py would have seen.
    """

    def train_main(argv):
        train_main.calls.append(list(argv))
        lr = float(argv[argv.index("--lr") + 1])
        outcome = outcomes[lr]
        if isinstance(outcome, str):
            return {"status": outcome, "evals": []}
        return {"status": "completed",
                "evals": [{"eval_loss": outcome, "eval_ppl": math.exp(1)}]}

    train_main.calls = []
    return train_main


def _install(monkeypatch, tmp_path, outcomes):
    """Point the sweep at a fake trainer and a throwaway results dir."""
    fake = _fake_train(outcomes)
    monkeypatch.setattr(sweep_lr, "train_main", fake)
    monkeypatch.setattr(sweep_lr, "RESULTS_DIR", tmp_path)
    return fake


def _contains(seq, sub):
    """True if ``sub`` appears as a contiguous run inside ``seq``."""
    return any(seq[i:i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))


# --------------------------------------------------------------------------
# 1. divergence must not win
# --------------------------------------------------------------------------

def test_leading_nan_would_win_naive_min_but_is_excluded(monkeypatch, tmp_path):
    """The exact shape of the old bug: NaN first in the grid.

    ``min`` returns its first element unless a later one is strictly smaller,
    and ``1.2 < nan`` is False -- so the old selection handed the full runs the
    LR that had just blown up.
    """
    losses = [(GALORE_GRID[0], NAN), (GALORE_GRID[1], 1.5), (GALORE_GRID[2], 1.2)]
    # Documents the old behaviour on this very data, no monkeypatching needed.
    assert min(losses, key=lambda t: t[1])[0] == GALORE_GRID[0]
    assert math.isnan(min(losses, key=lambda t: t[1])[1])

    fake = _install(monkeypatch, tmp_path, dict(losses))
    winners = sweep_lr.run_sweep(["galore"], steps=3, smoke=True)

    assert winners == {"galore": GALORE_GRID[2]}, "diverged LR crowned as winner"
    assert len(fake.calls) == 3, "every candidate should still be trained"


@pytest.mark.parametrize("nan_index", [0, 1, 2])
def test_nan_candidate_is_never_selected(monkeypatch, tmp_path, nan_index):
    """Wherever divergence lands in the grid, the best *finite* LR wins."""
    base = {GALORE_GRID[0]: 2.0, GALORE_GRID[1]: 1.0, GALORE_GRID[2]: 3.0}
    outcomes = dict(base)
    outcomes[GALORE_GRID[nan_index]] = NAN
    expected = min((lr for lr in GALORE_GRID if lr != GALORE_GRID[nan_index]),
                   key=lambda lr: base[lr])

    _install(monkeypatch, tmp_path, outcomes)
    winners = sweep_lr.run_sweep(["galore"], steps=3, smoke=True)

    assert winners["galore"] == expected
    assert math.isfinite(base[winners["galore"]])


def test_infinite_loss_is_also_excluded(monkeypatch, tmp_path):
    """+inf orders below nothing, so naive min would never pick it -- but -inf
    from an overflowed logit would win every comparison. isfinite covers both."""
    outcomes = {GALORE_GRID[0]: float("-inf"), GALORE_GRID[1]: 1.0,
                GALORE_GRID[2]: 3.0}
    _install(monkeypatch, tmp_path, outcomes)
    winners = sweep_lr.run_sweep(["galore"], steps=3, smoke=True)
    assert winners == {"galore": GALORE_GRID[1]}


def test_all_nan_method_is_absent_from_winners(monkeypatch, tmp_path):
    """No usable candidate means no winner -- not a NaN winner, and not a
    silently-defaulted LR that the full runs would then trust."""
    outcomes = {lr: NAN for lr in GALORE_GRID}
    _install(monkeypatch, tmp_path, outcomes)
    winners = sweep_lr.run_sweep(["galore"], steps=3, smoke=True)

    assert "galore" not in winners
    assert winners == {}


def test_failed_and_empty_records_are_excluded(monkeypatch, tmp_path):
    """A run that OOM'd or died mid-flight has no score to compare."""
    outcomes = {GALORE_GRID[0]: "oom", GALORE_GRID[1]: "started", GALORE_GRID[2]: 4.0}
    _install(monkeypatch, tmp_path, outcomes)
    winners = sweep_lr.run_sweep(["galore"], steps=3, smoke=True)
    assert winners == {"galore": GALORE_GRID[2]}


def test_edge_winner_is_flagged(monkeypatch, tmp_path, capsys):
    """A winner at either end of the grid means the true optimum is probably
    outside it; the sweep says so instead of quietly reporting a boundary."""
    _install(monkeypatch, tmp_path,
             {GALORE_GRID[0]: 3.0, GALORE_GRID[1]: 2.0, GALORE_GRID[2]: 1.0})
    sweep_lr.run_sweep(["galore"], steps=3, smoke=True)
    assert "EDGE" in capsys.readouterr().out

    _install(monkeypatch, tmp_path,
             {GALORE_GRID[0]: 3.0, GALORE_GRID[1]: 1.0, GALORE_GRID[2]: 2.0})
    sweep_lr.run_sweep(["galore"], steps=3, smoke=True)
    assert "EDGE" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# 2. the sweep must be runnable under the full runs' settings
# --------------------------------------------------------------------------

def test_passthrough_is_forwarded_verbatim_to_train(monkeypatch, tmp_path):
    """Every candidate must be trained under the caller's knobs, unmodified."""
    passthrough = ["--preset", "24gb", "--dtype", "fp32", "--galore-rank", "64"]
    fake = _install(monkeypatch, tmp_path, {lr: 1.0 + i for i, lr in enumerate(GALORE_GRID)})

    sweep_lr.run_sweep(["galore"], steps=3, smoke=True, passthrough=passthrough)

    assert len(fake.calls) == len(GALORE_GRID)
    for argv in fake.calls:
        assert _contains(argv, passthrough), argv
        # the sweep still owns the per-candidate flags, exactly once each
        for flag in ("--method", "--lr", "--max-steps", "--run-name", "--phase"):
            assert argv.count(flag) == 1, argv
        assert argv[argv.index("--phase") + 1] == "sweep"

    # provenance: the winners file records what the sweep was run under
    written = config.read_jsonl(tmp_path / "sweep_winners.jsonl")
    assert written[-1]["passthrough"] == passthrough


def test_passthrough_does_not_leak_between_methods(monkeypatch, tmp_path):
    """The list is copied per call, not accumulated across the method loop."""
    passthrough = ["--dtype", "fp32"]
    outcomes = {lr: 1.0 for lr in GALORE_GRID}
    outcomes.update({lr: 1.0 for lr in sweep_lr.DEFAULT_LR_GRIDS["lora"]})
    fake = _install(monkeypatch, tmp_path, outcomes)

    sweep_lr.run_sweep(["galore", "lora"], steps=3, smoke=True, passthrough=passthrough)

    for argv in fake.calls:
        assert argv.count("--dtype") == 1, argv
    assert passthrough == ["--dtype", "fp32"], "caller's list was mutated"


def _run_cli(monkeypatch, tmp_path, argv, train_main):
    """Execute sweep_lr.py's __main__ block in-process, with train.main faked.

    The argparse wiring under test lives in ``if __name__ == '__main__'``, so it
    is reachable only by running the file; injecting a fake ``train`` module
    keeps that run from touching a GPU or the network.
    """
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    fake_module = types.ModuleType("train")
# improvement
    fake_module.main = train_main
    monkeypatch.setitem(sys.modules, "train", fake_module)
    monkeypatch.setattr(sys, "argv", ["sweep_lr.py", *argv])
    runpy.run_path(sweep_lr.__file__, run_name="__main__")


def test_cli_accepts_and_forwards_documented_knobs(monkeypatch, tmp_path):
    """The regression for bug 2: these flags used to exit 2 with
    "unrecognized arguments", so the sweep could not mirror the full runs."""
    fake = _fake_train({lr: 1.0 + i for i, lr in enumerate(GALORE_GRID)})
    knobs = ["--preset", "24gb", "--dtype", "fp32", "--galore-rank", "64"]

    _run_cli(monkeypatch, tmp_path, ["--methods", "galore", "--steps", "3",
                                     "--smoke", "true", *knobs], fake)

    assert len(fake.calls) == len(GALORE_GRID)
    for argv in fake.calls:
        assert _contains(argv, knobs), argv


def test_cli_rejects_a_typoed_knob_before_training_anything(monkeypatch, tmp_path):
    """Forwarding blind would surface the typo 40 minutes into rented GPU time;
    validating against RunConfig's parser surfaces it immediately."""
    fake = _fake_train({lr: 1.0 for lr in GALORE_GRID})

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, tmp_path,
                 ["--methods", "galore", "--galore-rnak", "64"], fake)

    assert exc.value.code == 2
    assert fake.calls == [], "a typo must not start any run"


def test_cli_rejects_flags_the_sweep_sets_itself(monkeypatch, tmp_path):
    """--lr is chosen per candidate; accepting one from the caller would make
    the sweep silently sweep nothing."""
    fake = _fake_train({lr: 1.0 for lr in GALORE_GRID})

    with pytest.raises(SystemExit) as exc:
        _run_cli(monkeypatch, tmp_path,
                 ["--methods", "galore", "--lr", "1e-4"], fake)

    assert isinstance(exc.value.code, str) and "--lr" in exc.value.code
    assert fake.calls == []

# Optimized

# Optimized

# Enhanced

# Refined

# Refined
