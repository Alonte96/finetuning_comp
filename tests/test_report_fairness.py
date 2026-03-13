"""Fairness is checked *within* a phase; across phases, step count is exempt.

The RUNBOOK appends every run to the same `results/runs.jsonl`: the memory
bench (30 steps), the LR sweep (200 steps) and the final runs (1000 steps).
A flat fingerprint comparison against one arbitrary reference record therefore
flagged `max_steps` every single time, so following the documented procedure
*always* printed "Fairness check FAILED" into REPORT.md — a false alarm that
makes an otherwise valid table look rigged.

The claims, each tested:
1. mem/sweep/final runs differing ONLY in max_steps produce no problems;
2. two runs in the SAME phase differing on seq_len are still flagged, and the
   complaint is scoped to that phase;
3. a genuine cross-phase mismatch (dtype) is flagged, and max_steps is not
   dragged into the diff with it;
4. the exemption is *only* cross-phase: two runs in the same phase with
   different step counts are still unfair, because they are the runs the
   published table compares directly to each other;
5. legacy records written before the `phase` field default to "final";
6. the report partitions runs into one table per phase, and REPORT.md for the
   RUNBOOK's own three-phase file actually says the check PASSED.
"""

import json
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import report as report_module
from config import RunConfig
from report import (
    CROSS_PHASE_EXEMPT,
    PHASE_TITLES,
    build_table,
    check_fairness,
    group_by_phase,
    phase_of,
)
# improvement

# What train.py actually stores under "fairness" (cfg.fairness_fingerprint()).
BASE_FINGERPRINT = {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "dataset_id": "yahma/alpaca-cleaned",
    "seq_len": 512,
    "micro_batch": 2,
#     "grad_accum": 16,
    "max_steps": 1000,
    "eval_examples": 500,
    "seed": 42,
    "dtype": "bf16",
    "grad_checkpointing": True,
}

# Step counts the RUNBOOK prescribes for each phase.
RUNBOOK_STEPS = {"mem": 30, "sweep": 200, "final": 1000}


def _record(name, phase="final", **overrides):
    """One runs.jsonl record, trimmed to the keys the report reads."""
    record = {
        "run": name,
        "fairness": {**BASE_FINGERPRINT, **overrides},
        "status": "completed",
        "config": {"method": name.split("_")[-1]},
    }
    if phase is not None:  # phase=None models a pre-fix, phase-less record
        record["phase"] = phase
    return record


def _runbook_records():
    """Exactly what the documented three-stage procedure appends to one file."""
    return [
        _record("mem_full", "mem", max_steps=RUNBOOK_STEPS["mem"]),
        _record("mem_lora", "mem", max_steps=RUNBOOK_STEPS["mem"]),
        _record("mem_galore", "mem", max_steps=RUNBOOK_STEPS["mem"]),
        _record("sweep_full_lr2e-5", "sweep", max_steps=RUNBOOK_STEPS["sweep"]),
        _record("sweep_lora_lr3e-4", "sweep", max_steps=RUNBOOK_STEPS["sweep"]),
        _record("sweep_galore_lr3e-5", "sweep", max_steps=RUNBOOK_STEPS["sweep"]),
        _record("final_full", "final", max_steps=RUNBOOK_STEPS["final"]),
        _record("final_lora", "final", max_steps=RUNBOOK_STEPS["final"]),
        _record("final_galore", "final", max_steps=RUNBOOK_STEPS["final"]),
    ]


def _runs(records):
    return {r["run"]: r for r in records}


def test_synthetic_fingerprint_matches_the_real_protocol():
    """Guard: these tests are worthless if they fingerprint the wrong fields."""
    assert set(BASE_FINGERPRINT) == set(RunConfig.FAIRNESS_FIELDS)
    assert "max_steps" in RunConfig.FAIRNESS_FIELDS
    assert CROSS_PHASE_EXEMPT == ("max_steps",)


def test_runbook_phases_differing_only_in_max_steps_are_fair():
    """The exact scenario that used to fail: 30 / 200 / 1000 steps in one file.

    Nothing but the step count differs, so the only honest verdict is silence.
    """
    runs = _runs(_runbook_records())
    assert {phase_of(r) for r in runs.values()} == {"mem", "sweep", "final"}
    assert len({r["fairness"]["max_steps"] for r in runs.values()}) == 3

    assert check_fairness(runs) == []


def test_same_phase_seq_len_mismatch_is_flagged_and_scoped_to_that_phase():
    """Within a phase every field still matters, step count included."""
    records = _runbook_records()
    records.append(_record("final_galore_long", "final", seq_len=1024))
    problems = check_fairness(_runs(records))

    # The odd run out is the *only* complaint: the three legitimate step counts
    # must not add noise around it, and the cross-phase representatives
    # (alphabetically first per phase) all still agree on seq_len.
    assert len(problems) == 1, problems
    (problem,) = problems
    assert "seq_len" in problem
    assert "final_galore_long" in problem
    assert "phase 'final'" in problem, "complaint is not scoped to a phase"
    assert "max_steps" not in problem


def test_same_phase_check_ignores_runs_in_other_phases():
    """A broken sweep run must not implicate the final runs, or vice versa."""
    records = _runbook_records()
    records.append(_record("sweep_full_lr5e-5", "sweep", micro_batch=8))
    problems = check_fairness(_runs(records))

    assert len(problems) == 1, problems
# improvement
    assert "phase 'sweep'" in problems[0]
    assert "micro_batch" in problems[0]
    assert "final" not in problems[0]


def test_within_a_phase_max_steps_still_has_to_match():
    """The exemption must be cross-phase ONLY -- the over-correction is worse.

    Exempting max_steps everywhere would silence the one comparison that matters
    most: a headline run trained for 200 steps next to one trained for 1000 is
    exactly the rigged table the fairness check exists to refuse. Only the
    *phase label* may excuse a step-count difference, never the field itself.
    """
    records = _runbook_records()
    records.append(_record("final_lora_short", "final", max_steps=RUNBOOK_STEPS["sweep"]))
    problems = check_fairness(_runs(records))

    assert len(problems) == 1, problems
    (problem,) = problems
    assert "max_steps" in problem, "step count is exempt within a phase too"
    assert "final_lora_short" in problem
    assert "phase 'final'" in problem
    assert "200" in problem and "1000" in problem, problem
    # It is the within-phase check that caught it, not cross-phase leakage.
    assert "across phases" not in problem

    # The very same 200-step run, filed under the phase it belongs to, is fine:
    # the phase label is what excuses the difference, nothing else changed.
    moved = _runbook_records()
    moved.append(_record("sweep_lora_extra", "sweep", max_steps=RUNBOOK_STEPS["sweep"]))
    assert check_fairness(_runs(moved)) == []


def test_real_cross_phase_mismatch_is_still_caught():
    """Benchmarking memory in fp32 and then training in bf16 is not a benchmark."""
    records = [
        _record(name, phase, max_steps=RUNBOOK_STEPS[phase],
                **({"dtype": "fp32"} if phase == "mem" else {}))
        for phase, names in (
            ("mem", ["mem_full", "mem_galore"]),
            ("sweep", ["sweep_full", "sweep_galore"]),
            ("final", ["final_full", "final_galore"]),
        )
        for name in names
    ]
    problems = check_fairness(_runs(records))

    assert problems, "dtype drift between phases went unreported"
    joined = "\n".join(problems)
    assert "dtype" in joined
    assert all("across phases" in p for p in problems), problems
    # The exempt field must not ride along and dilute the real finding.
    assert "max_steps" not in joined


def test_legacy_records_without_phase_default_to_final():
    """Records written before the phase field exist in results/runs.jsonl."""
    assert phase_of({}) == "final"
    assert phase_of({"run": "old", "fairness": BASE_FINGERPRINT}) == "final"

    legacy = [_record("old_full", phase=None), _record("old_galore", phase=None)]
    assert all("phase" not in r for r in legacy)

    grouped = group_by_phase(_runs(legacy))
    assert set(grouped) == {"final"}
    assert check_fairness(_runs(legacy)) == []

    # Mixed old and new records: the legacy pair joins the final phase and is
    # compared against it in full, while the 30-step memory runs stay exempt.
    mixed = legacy + [
        _record("final_lora", "final"),
        _record("mem_full", "mem", max_steps=RUNBOOK_STEPS["mem"]),
    ]
    assert check_fairness(_runs(mixed)) == []

    broken = mixed + [_record("old_seed7", phase=None, seed=7)]
    problems = check_fairness(_runs(broken))
    assert len(problems) == 1, problems
    assert "phase 'final'" in problems[0] and "seed" in problems[0]


def test_empty_input_is_reported_rather_than_silently_passing():
    assert check_fairness({}) == ["no runs found"]


def test_report_renders_one_table_per_phase():
    grouped = group_by_phase(_runs(_runbook_records()))
    assert set(grouped) == {"mem", "sweep", "final"}
    assert set(grouped["final"]) == {"final_full", "final_lora", "final_galore"}

    table = build_table(grouped["mem"])
    assert "mem_full" in table
    assert "final_full" not in table, "phase tables are mixing runs"
    assert len([ln for ln in table.splitlines() if ln.startswith("| mem_")]) == 3


def test_report_md_for_the_runbook_file_says_the_check_passed(tmp_path, monkeypatch):
    """End to end on the symptom: the banner REPORT.md actually prints.

    check_fairness() returning [] is only half the fix -- the RUNBOOK's honesty
    checklist tells the reader to look for "Fairness check passed" in
    results/REPORT.md, so the rendering has to agree with the verdict.
    """
    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text("".join(json.dumps(r) + "\n" for r in _runbook_records()))
    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(report_module, "plot_curves", lambda runs, out_dir: [])

    report_module.main([])
    text = (tmp_path / "REPORT.md").read_text()

    assert "Fairness check passed" in text
    assert "FAILED" not in text, text[:400]
    assert "Incomplete" not in text, "all three phases are present"

    # One section per phase, in protocol order, each holding only its own runs.
    positions = [text.index(PHASE_TITLES[p]) for p in ("mem", "sweep", "final")]
    assert positions == sorted(positions)
    assert positions[0] < text.index("| mem_full |") < positions[1]
    assert positions[2] < text.index("| final_full |")


def test_complete_replication_under_a_new_seed_is_allowed():
    """Re-running EVERY method at a new seed is how you show a ranking is not
    noise, so it must not be reported as unfairness."""
    records = _runbook_records()
    for method in ("full", "lora", "galore"):
        records.append(_record(f"seed43_{method}", "final", seed=43))
    assert check_fairness(_runs(records)) == []


def test_partial_replication_is_flagged():
    """Re-rolling the seed for one method only is the oldest way to manufacture
    a favourable number; exempting seed outright would let it through."""
    records = _runbook_records()
    records.append(_record("seed43_galore", "final", seed=43))
    problems = check_fairness(_runs(records))
    assert len(problems) == 1, problems
    assert "seed 43" in problems[0] and "incomplete replication" in problems[0]


def test_replication_still_catches_a_real_mismatch_inside_a_seed():
    """Seed exemption must not become a hiding place for other fields."""
    records = _runbook_records()
    for method in ("full", "lora", "galore"):
        records.append(_record(f"seed43_{method}", "final", seed=43,
                               **({"seq_len": 1024} if method == "galore" else {})))
    problems = check_fairness(_runs(records))
    assert any("seq_len" in p and "seed43_galore" in p for p in problems), problems

# Enhanced

# Enhanced

# Enhanced

# Refined
