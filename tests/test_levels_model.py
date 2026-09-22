"""Slow tests against the real Laya checkpoints.

Run with:  SWAY_MODEL_TESTS=1 pytest -q tests/test_levels_model.py
On small machines load one checkpoint at a time, e.g. LAYA_CHECKPOINTS=multilingual;
levels routed to a checkpoint that is not loaded are skipped.
"""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SWAY_MODEL_TESTS") != "1", reason="set SWAY_MODEL_TESTS=1")


@pytest.fixture(scope="module")
def engine():
    from sway.engine import Engine

    return Engine().start()


def _levels():
    from sway.levels import LEVELS

    return LEVELS


@pytest.mark.parametrize("level", _levels(), ids=lambda lvl: lvl.id)
def test_example_passes(engine, level):
    from sway.rules import judge, pre_check, sanitize

    text = sanitize(level.example, level.max_chars)
    decision = engine.route(level.build_state(text))
    if decision["model"] not in engine.checkpoints:
        pytest.skip(f"routes to {decision['model']}, not loaded")
    gate = pre_check(level, text, decision["reason"], decision["model"])
    assert gate.ok, gate.reason
    verdict = engine.predict(level.build_state(text), level.questions)
    outcome = judge(level, verdict.answers, 0)
    assert outcome.passed, [(c.label, c.detail) for c in outcome.checks]


def test_lab_presets_run(engine):
    from sway.lab import PRESET_NAMES, parse_state, preset, validate_questions

    for name in PRESET_NAMES:
        key, q, example = preset(name)
        v = engine.predict(parse_state(example, key), validate_questions(q))
        assert v.answers and v.latency_ms > 0


def test_quick_autoplay_solves_most_cases(engine):
    """Quick autoplay, seed 42, against the real checkpoints."""
    from sway.autoplay import RunLimits, play_run
    from sway.levels import LEVELS

    ids = [lvl.id for lvl in LEVELS]
    summary = None
    for event in play_run(ids, engine, "Quick", 42, RunLimits()):
        if event.kind == "summary":
            summary = event.summary
    assert summary is not None
    report = ", ".join(f"{k} {v}" for k, v in summary.best.items())
    assert summary.solved >= 6, f"solved {summary.solved}/{summary.attempted}: {report}"
