"""--preset must be a *default provider*, not an override, and "auto" must run.

Two separate claims about ``experiments/config.py``, both previously false:

1. **Auto-detection is live.** ``--preset`` documents a default of ``auto``
   (RUNBOOK.md: bench_memory "auto-detects GPU tier"). The dataclass default
   used to be ``None``, and the guard was ``if cfg.preset:`` -- so
   ``select_preset(None)`` / ``detect_gpu_gib()`` in ``gradproj.presets`` were
   never reached from the CLI at all. The tier chosen for a card was dead code.
2. **Explicit flags win.** A preset used to unconditionally assign seq_len,
   micro_batch, grad_accum and grad_checkpointing, so
   ``--preset 16gb --seq-len 999`` silently ran at 256 tokens. The fix records
   which flags were actually typed (``build_parser(suppress_defaults=True)``)
   and ``_apply_preset`` skips those fields while still filling the rest.

Pure argv parsing -- no model, no training, no network, no CUDA.
``_apply_preset`` imports ``gradproj.presets`` lazily *inside the function*, so
simulating a card means patching the attribute on that module (which
``select_preset`` also resolves through), not a copy bound into config.py.
"""

import pytest
import torch

import gradproj.presets as presets
from experiments.config import RunConfig

PRESET_FIELDS = ("seq_len", "micro_batch", "grad_accum", "grad_checkpointing")

# The dataclass defaults the smoke/laptop path must keep.
BUILTIN_DEFAULTS = {"seq_len": 512, "micro_batch": 2, "grad_accum": 16}


def _knobs(cfg) -> dict:
    return {f: getattr(cfg, f) for f in PRESET_FIELDS}


def _expected(name: str) -> dict:
    p = presets.PRESETS[name]
    return {f: getattr(p, f) for f in PRESET_FIELDS}


def _fake_card(monkeypatch, gib):
    """Simulate a CUDA card of `gib` GiB (or None for CPU/MPS)."""
    monkeypatch.setattr(presets, "detect_gpu_gib", lambda device=0: gib)


# --------------------------------------------------------------------------
# 1. an explicit preset supplies its whole tier
# --------------------------------------------------------------------------

def test_explicit_preset_applies_its_tier():
    cfg = RunConfig.from_args(["--preset", "16gb"])

    assert _knobs(cfg) == _expected("16gb")
    assert _knobs(cfg) == {
        "seq_len": 256, "micro_batch": 1, "grad_accum": 32, "grad_checkpointing": True,
    }
    assert cfg.preset == "16gb"


def test_explicit_preset_is_the_only_thing_it_touches():
    """A preset must not reach outside its four fairness knobs."""
    baseline = RunConfig.from_args(["--preset", "24gb", "--method", "lora", "--lr", "3e-4"])

    assert baseline.method == "lora"
    assert baseline.lr == 3e-4
    assert baseline.max_steps == 1000
    assert baseline.seed == 42


# --------------------------------------------------------------------------
# 2. the precedence bug: an explicit flag beats the preset, others still apply
# --------------------------------------------------------------------------

def test_explicit_seq_len_survives_a_preset():
    """`--preset 16gb --seq-len 999` used to silently run at 256 tokens."""
    cfg = RunConfig.from_args(["--preset", "16gb", "--seq-len", "999"])

    assert cfg.seq_len == 999, "preset clobbered an explicitly-passed --seq-len"
    # ...and the preset still fills in everything the caller did NOT type.
    assert cfg.micro_batch == 1
    assert cfg.grad_accum == 32
    assert cfg.grad_checkpointing is True


def test_explicit_bool_survives_a_preset_that_disagrees():
    """40gb turns checkpointing OFF; typing it ON by hand must stick.

    Chosen because the dataclass default (True) and the preset value (False)
    differ, so a surviving True can only have come from the explicit flag.
    """
    cfg = RunConfig.from_args(["--preset", "40gb", "--grad-checkpointing", "true"])

    assert cfg.grad_checkpointing is True
    assert cfg.seq_len == 1024      # preset still applied elsewhere
    assert cfg.micro_batch == 4
    assert cfg.grad_accum == 8


def test_all_four_knobs_can_be_pinned_against_a_preset():
    cfg = RunConfig.from_args([
        "--preset", "80gb",
        "--seq-len", "128", "--micro-batch", "3",
        "--grad-accum", "5", "--grad-checkpointing", "true",
    ])

    assert _knobs(cfg) == {
        "seq_len": 128, "micro_batch": 3, "grad_accum": 5, "grad_checkpointing": True,
    }
    assert _knobs(cfg) != _expected("80gb")


def test_a_flag_typed_at_its_default_value_still_counts_as_explicit():
    """Precedence is by *presence on the command line*, not by value.

    `--seq-len 512` equals the dataclass default, so a value-comparison
    implementation would let 16gb overwrite it with 256.
    """
    cfg = RunConfig.from_args(["--preset", "16gb", "--seq-len", "512"])

    assert cfg.seq_len == 512
    assert cfg.micro_batch == 1     # untyped -> preset wins


# --------------------------------------------------------------------------
# 3. "auto" is the default and it actually resolves
# --------------------------------------------------------------------------

def test_preset_field_defaults_to_auto():
    """The documented default. It used to be None, which made the auto branch
    unreachable from the CLI (`if cfg.preset:` was never true)."""
    assert RunConfig.preset == "auto"
    assert RunConfig().preset == "auto"


@pytest.mark.skipif(torch.cuda.is_available(), reason="this assertion is the no-CUDA path")
def test_auto_on_this_machine_leaves_builtin_defaults_alone():
    """Real hardware, no patching: MPS/CPU must not inherit a VRAM tier."""
    assert presets.detect_gpu_gib() is None

    cfg = RunConfig.from_args([])

    for name, value in BUILTIN_DEFAULTS.items():
        assert getattr(cfg, name) == value, f"{name} was rewritten on a card-less machine"
    assert cfg.preset is None, "auto should resolve to 'no preset' without CUDA"


def test_auto_without_cuda_is_a_no_op(monkeypatch):
    """No-op by *deciding* to be one, not by never running.

    The spy is the whole point: with the old ``preset = None`` default and the
    ``if cfg.preset:`` guard, detection was never consulted at all, and the
    identical end state (defaults intact, preset None) hid that.
    """
    calls = []

    def _no_card(device=0):
        calls.append(device)
        return None

    monkeypatch.setattr(presets, "detect_gpu_gib", _no_card)

    cfg = RunConfig.from_args([])

    assert calls, "the auto branch never ran -- detect_gpu_gib() was not consulted"
    assert cfg.preset is None
    for name, value in BUILTIN_DEFAULTS.items():
        assert getattr(cfg, name) == value


def test_auto_picks_the_24gb_tier_on_a_24gib_card(monkeypatch):
    """The dead code, now reached: no --preset flag at all, yet a tier lands."""
    _fake_card(monkeypatch, 24.0)

    cfg = RunConfig.from_args([])

    assert cfg.preset == "24gb"
    assert _knobs(cfg) == _expected("24gb")


def test_auto_picks_the_80gb_tier_on_an_80gib_card(monkeypatch):
    _fake_card(monkeypatch, 80.0)

    cfg = RunConfig.from_args([])

    assert cfg.preset == "80gb"
    assert _knobs(cfg) == _expected("80gb")
    # 80gb is the tier that visibly differs from the dataclass defaults.
    assert cfg.seq_len == 1024
    assert cfg.micro_batch == 8
    assert cfg.grad_accum == 4
    assert cfg.grad_checkpointing is False


def test_auto_can_be_written_out_explicitly(monkeypatch):
    _fake_card(monkeypatch, 80.0)

    assert RunConfig.from_args(["--preset", "auto"]).preset == "80gb"


def test_auto_detected_tier_still_yields_to_explicit_flags(monkeypatch):
    """Precedence and auto-detection compose."""
    _fake_card(monkeypatch, 80.0)

    cfg = RunConfig.from_args(["--micro-batch", "1"])

    assert cfg.preset == "80gb"
    assert cfg.micro_batch == 1     # typed by hand
    assert cfg.seq_len == 1024      # from the detected tier


# --------------------------------------------------------------------------
# 4. bad input dies with SystemExit, not a stray ValueError/KeyError
# --------------------------------------------------------------------------

def test_unknown_preset_name_exits():
    with pytest.raises(SystemExit) as exc:
        RunConfig.from_args(["--preset", "12gb"])
    assert "preset" in str(exc.value)
    assert "12gb" in str(exc.value)


def test_unknown_preset_is_not_a_bare_valueerror():
    """select_preset() raises ValueError; the CLI must translate it."""
    with pytest.raises(ValueError):
        presets.select_preset("12gb")

    with pytest.raises(SystemExit):
        RunConfig.from_args(["--preset", "12gb"])


def test_invalid_phase_exits():
    with pytest.raises(SystemExit) as exc:
        RunConfig.from_args(["--phase", "warmup"])
    assert "phase" in str(exc.value)

    for phase in ("mem", "sweep", "final"):
        assert RunConfig.from_args(["--phase", phase]).phase == phase


def test_invalid_method_exits():
    with pytest.raises(SystemExit):
        RunConfig.from_args(["--method", "adapters"])


# --------------------------------------------------------------------------
# 5. the mechanism itself: suppress_defaults records exactly what was typed
# --------------------------------------------------------------------------

def test_suppressed_parser_records_only_typed_flags():
    parser = RunConfig.build_parser(suppress_defaults=True)

    seen = vars(parser.parse_args(["--preset", "16gb", "--seq-len", "999"]))

    assert seen == {"preset": "16gb", "seq_len": 999}


def test_suppressed_parser_is_empty_for_an_empty_command_line():
    parser = RunConfig.build_parser(suppress_defaults=True)
    assert vars(parser.parse_args([])) == {}


def test_normal_parser_still_fills_every_field():
    """Only the shadow parser suppresses; the real one must stay complete."""
    seen = vars(RunConfig.build_parser().parse_args([]))

    assert seen["seq_len"] == 512
    assert seen["preset"] == "auto"
    assert RunConfig(**seen).fairness_fingerprint() == RunConfig().fairness_fingerprint()


def test_suppressed_parser_accepts_the_same_flags_as_the_real_one():
    typed = ["--method", "galore", "--grad-checkpointing", "false", "--lr", "0.0001"]

    full = vars(RunConfig.build_parser().parse_args(typed))
    seen = vars(RunConfig.build_parser(suppress_defaults=True).parse_args(typed))

    assert set(seen) == {"method", "grad_checkpointing", "lr"}
    for key, value in seen.items():
        assert full[key] == value, f"{key} parsed differently by the two parsers"

# Enhanced

# Optimized

# Enhanced
