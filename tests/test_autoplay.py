"""Autoplay tests. No model: a fake engine answers every question."""
import random

import pytest

from sway.autoplay import (ALL_CASES, BUDGETS, MAX_RUN_CALLS, PASS_SCORE, RunLimits, case_choices,
                           compose, mutate, play_level, play_run, resolve_cases)
from sway.autoplay_data import LIBRARY, OPTIONAL, SLOTS
from sway.engine import EngineNotReady, Verdict
from sway.levels import BY_ID, LEVELS
from sway.rules import find_banned, letter_count


# ---------------------------------------------------------------- fake engine

ENGLISH_WORDS = {
    w.strip(".,:;?!").lower()
    for books in LIBRARY.values() for book in books if book.lang == "en"
    for frags in book.slots.values() for frag in frags for w in frag.split()
}


def _looks_english(text: str) -> bool:
    """Stand-in for the router's language detection: mostly known English words."""
    words = [w.strip(".,:;?!¿").lower() for w in text.split() if w]
    if not words:
        return True
    known = sum(w in ENGLISH_WORDS for w in words)
    return known / len(words) >= 0.8


class FakeEngine:
    """Answers every question shape. `bias` shifts how well candidates score."""

    def __init__(self, bias=0.5, raises=None):
        self.bias = bias
        self.raises = raises
        self.seen = []          # every text that reached predict()
        self.routed = []        # (text, model) for every route() call

    def route(self, state):
        text = " ".join(str(v) for v in state.values())
        model = "english" if _looks_english(text) else "multilingual"
        self.routed.append((text, model))
        return {"model": model, "reason": f"fake detection: {model}"}

    def predict(self, state, questions):
        if self.raises is not None:
            raise self.raises
        self.seen.append(max(state.values(), key=len))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": self.bias, "confidence": 0.9}
            elif q["type"] == "choice":
                keys = list(q["criteria"])
                rest = (1 - self.bias) / max(1, len(keys) - 1)
                probs = {k: (self.bias if i == 0 else rest) for i, k in enumerate(keys)}
                top = max(probs, key=probs.get)
                answers[qid] = {"type": "choice", "choice": top, "probabilities": probs,
                                "confidence": probs[top]}
            else:
                n = len(q["criteria"])
                answers[qid] = {"type": "score", "score": self.bias * (n - 1),
                                "probabilities": {str(i): 1 / n for i in range(n)}, "confidence": 0.5}
        return Verdict(answers, {"model": "multilingual", "reason": "fake"}, 12.5, 30)


def stepping_clock(step=0.0):
    t = [0.0]

    def clock():
        t[0] += step
        return t[0]

    return clock


# ---------------------------------------------------------------- library integrity

def test_every_level_has_a_library_a_use_case_and_a_hint():
    assert set(LIBRARY) == {lvl.id for lvl in LEVELS}
    for lvl in LEVELS:
        assert lvl.use_case, lvl.id
        assert lvl.hint, lvl.id
        assert lvl.example not in [f for bk in LIBRARY[lvl.id] for fr in bk.slots.values() for f in fr]
        for book in LIBRARY[lvl.id]:
            assert book.order(), (lvl.id, book.lang)
            assert set(book.slots) <= set(SLOTS), (lvl.id, book.lang)
            assert [s for s in book.order() if s not in OPTIONAL], (lvl.id, book.lang)
            for slot, frags in book.slots.items():
                assert frags, (lvl.id, book.lang, slot)
                for frag in frags:
                    assert not find_banned(frag, lvl.banned), (lvl.id, slot, frag)


def test_non_english_levels_have_no_english_phrasebook():
    for lvl in LEVELS:
        if lvl.require_non_english:
            assert {bk.lang for bk in LIBRARY[lvl.id]} and "en" not in {bk.lang for bk in LIBRARY[lvl.id]}, lvl.id


def test_composed_candidates_fit_the_rules():
    rng = random.Random(7)
    for lvl in LEVELS:
        for _ in range(200):
            cand = compose(lvl, rng)
            assert len(cand.text) <= lvl.max_chars, (lvl.id, cand.text)
            assert letter_count(cand.text) >= lvl.min_letters, (lvl.id, cand.text)
            assert not find_banned(cand.text, lvl.banned), (lvl.id, cand.text)


def test_mutations_stay_inside_the_rules_and_change_something():
    rng = random.Random(11)
    for lvl in LEVELS:
        parent = compose(lvl, rng)
        for _ in range(50):
            child = mutate(lvl, parent, rng)
            if child is None:
                continue
            assert child != parent
            assert child.book == parent.book          # never mixes two languages
            assert len(child.text) <= lvl.max_chars
            assert not find_banned(child.text, lvl.banned)
            parent = child


# ---------------------------------------------------------------- reproducibility

def _trace(seed, budget="Quick", bias=0.5):
    engine = FakeEngine(bias)
    events = list(play_run([lvl.id for lvl in LEVELS], engine, budget, seed,
                           RunLimits(clock=stepping_clock(0.0))))
    return [(e.kind, e.level_id, e.attempt, e.text, e.blocked) for e in events], engine


def test_same_seed_gives_the_same_candidate_sequence():
    first, engine_a = _trace(42)
    second, engine_b = _trace(42)
    assert first == second
    assert engine_a.seen == engine_b.seen


def test_a_different_seed_searches_differently():
    first, _ = _trace(42)
    other, _ = _trace(7)
    assert first != other


# ---------------------------------------------------------------- budgets and caps

@pytest.mark.parametrize("budget", sorted(BUDGETS))
def test_per_level_budget_is_never_exceeded(budget):
    for lvl in LEVELS:
        engine = FakeEngine(0.5)
        list(play_level(lvl, engine, budget, random.Random(3)))
        assert len(engine.seen) <= BUDGETS[budget], (lvl.id, len(engine.seen))


def test_run_call_cap_holds_even_across_every_case():
    engine = FakeEngine(0.5)
    limits = RunLimits(max_calls=17, clock=stepping_clock(0.0))
    events = list(play_run([lvl.id for lvl in LEVELS], engine, "Thorough", 42, limits))
    assert len(engine.seen) <= 17
    assert events[-1].summary.calls <= 17
    assert "model calls" in events[-1].summary.stopped


def test_run_time_cap_holds():
    engine = FakeEngine(0.5)
    limits = RunLimits(max_seconds=10.0, clock=stepping_clock(4.0))
    events = list(play_run([lvl.id for lvl in LEVELS], engine, "Thorough", 42, limits))
    assert events[-1].summary.attempted < len(LEVELS)
    assert "minute" in events[-1].summary.stopped


def test_default_caps_bound_a_full_thorough_run():
    engine = FakeEngine(0.5)
    events = list(play_run([lvl.id for lvl in LEVELS], engine, "Thorough", 42,
                           RunLimits(clock=stepping_clock(0.0))))
    assert len(engine.seen) <= MAX_RUN_CALLS
    assert events[-1].summary.calls == len(engine.seen)


# ---------------------------------------------------------------- gating

def test_no_banned_or_oversized_candidate_ever_reaches_predict():
    for lvl in LEVELS:
        engine = FakeEngine(0.5)
        list(play_level(lvl, engine, "Thorough", random.Random(5)))
        assert engine.seen, lvl.id
        for text in engine.seen:
            assert not find_banned(text, lvl.banned), (lvl.id, text)
            assert len(text) <= lvl.max_chars, (lvl.id, text)


def test_non_english_levels_never_send_english_candidates():
    for lvl in LEVELS:
        if not lvl.require_non_english:
            continue
        engine = FakeEngine(0.5)
        list(play_level(lvl, engine, "Thorough", random.Random(5)))
        assert engine.seen, lvl.id
        for text in engine.seen:
            assert not _looks_english(text), (lvl.id, text)


def test_blocked_candidates_cost_no_model_call():
    lvl = BY_ID["babel"]
    engine = FakeEngine(0.5)
    # An engine that hears everything as English blocks every candidate on a babel case.
    engine.route = lambda state: {"model": "english", "reason": "forced"}
    events = list(play_level(lvl, engine, "Quick", random.Random(1)))
    attempts = [e for e in events if e.kind == "attempt"]
    assert attempts and all(e.blocked for e in attempts)
    assert engine.seen == []
    assert all(e.calls == 0 for e in events)


# ---------------------------------------------------------------- stopping

def test_early_stop_once_a_candidate_scores_95_or_more():
    lvl = BY_ID["tick-tock"]
    engine = FakeEngine(1.0)  # every noul comes back at 1.0, so the first survivor wins
    events = list(play_level(lvl, engine, "Thorough", random.Random(2)))
    scored = [e for e in events if e.kind == "attempt" and not e.blocked]
    assert len(scored) == 1
    assert scored[0].score >= PASS_SCORE
    assert events[-1].kind == "level-end" and events[-1].passed
    assert len(engine.seen) == 1


def test_engine_not_ready_is_reported_not_raised():
    lvl = BY_ID["fog"]
    engine = FakeEngine(0.5, raises=EngineNotReady("The engine is still loading."))
    events = list(play_level(lvl, engine, "Quick", random.Random(4)))
    notices = [e for e in events if e.kind == "notice"]
    assert notices and "still loading" in notices[0].message
    assert events[-1].kind == "level-end" and events[-1].best.outcome is None


def test_engine_failure_ends_the_run_cleanly():
    engine = FakeEngine(0.5, raises=RuntimeError("boom"))
    events = list(play_run([lvl.id for lvl in LEVELS], engine, "Quick", 42,
                           RunLimits(clock=stepping_clock(0.0))))
    assert events[-1].kind == "summary"
    assert "boom" in events[-1].summary.stopped


# ---------------------------------------------------------------- summary and choices

def test_summary_counts_solved_cases_and_decision_time():
    engine = FakeEngine(1.0)
    events = list(play_run(["tick-tock", "fog"], engine, "Quick", 42,
                           RunLimits(clock=stepping_clock(0.0))))
    summary = events[-1].summary
    assert summary.attempted == 2
    assert summary.solved >= 1
    assert summary.average_best > 0
    assert summary.average_decision_ms == pytest.approx(12.5)


def test_case_choices_and_resolution():
    choices = case_choices()
    assert choices[0] == ("All cases", ALL_CASES)
    assert len(choices) == len(LEVELS) + 1
    assert resolve_cases(ALL_CASES) == [lvl.id for lvl in LEVELS]
    assert resolve_cases(None) == [lvl.id for lvl in LEVELS]
    assert resolve_cases("fog") == ["fog"]
    assert resolve_cases("nope") == []
