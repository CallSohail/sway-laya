"""Fast tests: no model download. Run with `pytest -q`."""
import json

import pytest

from sway import render
from sway.engine import Verdict, parse_checkpoints
from sway.game import (dismiss_welcome, new_session, play, repair, reset_level, share_text, shots_left,
                       use_hint, used_hint)
from sway.lab import LabError, parse_state, validate_questions
from sway.levels import BY_ID, LEVELS, SHOTS_PER_LEVEL
from sway.rules import Objective, find_banned, judge, pre_check, sanitize


# ---------------------------------------------------------------- sanitising and banned words

def test_sanitize_strips_invisible_and_clips():
    assert sanitize("re\u200bfund\x07", 100) == "refund"
    assert sanitize("  a   b \n\n\n\n c ", 100) == "a b \n\n c"
    assert sanitize(None, 10) == ""
    assert len(sanitize("x" * 50, 10)) == 10


@pytest.mark.parametrize("text", ["REFUND please", "r e f u n d", "r.e.f.u.n.d", "r3fund", "refunds", "Refünd"])
def test_banned_catches_evasions(text):
    assert find_banned(text, ["refund"]) == ["refund"]


@pytest.mark.parametrize("text", ["I know it", "snow day", "nowhere to go"])
def test_short_banned_words_match_whole_words_only(text):
    assert find_banned(text, ["now"]) == []


def test_short_banned_word_hits():
    assert find_banned("do it NOW", ["now"]) == ["now"]
    assert find_banned("n o w", ["now"]) == ["now"]


def test_multiword_banned_phrase():
    assert find_banned("please close my account", ["close my account"]) == ["close my account"]


# ---------------------------------------------------------------- objectives

def ans_noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def ans_choice(probs):
    top = max(probs, key=probs.get)
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": probs[top]}


def ans_score(s, k=4):
    return {"type": "score", "score": s, "probabilities": {str(i): 1 / k for i in range(k)}, "confidence": 0.5}


def test_above_below():
    assert Objective("above", "q", "x", t=0.7).evaluate({"q": ans_noul(0.8)}).passed
    assert not Objective("above", "q", "x", t=0.7).evaluate({"q": ans_noul(0.6)}).passed
    assert Objective("below", "q", "x", t=0.3).evaluate({"q": ans_noul(0.1)}).passed
    assert not Objective("below", "q", "x", t=0.3).evaluate({"q": ans_noul(0.5)}).passed


def test_choice_requires_label_and_probability():
    o = Objective("choice", "q", "x", arg="a", t=0.6)
    assert o.evaluate({"q": ans_choice({"a": 0.7, "b": 0.3})}).passed
    assert not o.evaluate({"q": ans_choice({"a": 0.55, "b": 0.45})}).passed
    assert not o.evaluate({"q": ans_choice({"a": 0.2, "b": 0.8})}).passed


def test_split_needs_top_two_and_small_gap():
    o = Objective("split", "q", "x", arg=("a", "b"), max_gap=0.25)
    assert o.evaluate({"q": ans_choice({"a": 0.45, "b": 0.4, "c": 0.15})}).passed
    assert not o.evaluate({"q": ans_choice({"a": 0.7, "b": 0.2, "c": 0.1})}).passed
    assert not o.evaluate({"q": ans_choice({"a": 0.4, "c": 0.35, "b": 0.25})}).passed


def test_band_directions():
    at_least = Objective("band", "q", "x", lo=2.0, hi=3.0)
    assert at_least.evaluate({"q": ans_score(2.9)}, 4).quality > at_least.evaluate({"q": ans_score(2.05)}, 4).quality
    at_most = Objective("band", "q", "x", lo=0.0, hi=1.2)
    assert at_most.evaluate({"q": ans_score(0.1)}, 4).quality > at_most.evaluate({"q": ans_score(1.1)}, 4).quality
    assert not at_most.evaluate({"q": ans_score(1.5)}, 4).passed


def test_missing_answer_fails_cleanly():
    c = Objective("above", "missing", "x").evaluate({})
    assert not c.passed and c.quality == 0


def test_judge_score_ranges_and_stars():
    lvl = BY_ID["tick-tock"]
    win = judge(lvl, {"urgent": ans_noul(0.99)}, 0)
    assert win.passed and 70 <= win.score <= 100 and win.stars == 3
    later = judge(lvl, {"urgent": ans_noul(0.99)}, 2)
    assert later.stars == 2
    lose = judge(lvl, {"urgent": ans_noul(0.2)}, 0)
    assert not lose.passed and lose.score <= 69 and lose.stars == 0


# ---------------------------------------------------------------- level data integrity

def test_levels_are_well_formed():
    ids = [lvl.id for lvl in LEVELS]
    assert len(ids) == len(set(ids)) == 10
    for lvl in LEVELS:
        assert lvl.objectives, lvl.id
        for o in lvl.objectives:
            assert o.q in lvl.questions, (lvl.id, o.q)
        assert lvl.example, lvl.id
        assert lvl.use_case and lvl.hint, lvl.id
        assert len(lvl.example) <= lvl.max_chars, lvl.id
        assert not find_banned(lvl.example, lvl.banned), (lvl.id, find_banned(lvl.example, lvl.banned))
        validate_questions(json.dumps(lvl.questions))


# ---------------------------------------------------------------- game flow with a fake engine

class FakeEngine:
    def __init__(self, answers, model="english"):
        self.answers = answers
        self.model = model
        self.calls = 0

    def route(self, state):
        return {"model": self.model, "reason": "test"}

    def predict(self, state, questions):
        self.calls += 1
        return Verdict(answers=self.answers, routing={"model": self.model, "reason": "test"},
                       latency_ms=12.0, input_tokens=20)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 5
        return self.t


def test_win_unlocks_next_and_records_best():
    s = new_session()
    eng = FakeEngine({"department": ans_choice({"billing": 0.9, "technical": 0.05, "sales": 0.03, "other": 0.02})})
    turn = play(s, "first-contact", "Someone took 40 euros from my card for nothing", eng, now=Clock())
    assert turn.kind == "judged" and turn.outcome.passed
    assert s["unlocked"] == 2 and s["best"]["first-contact"] >= 70
    assert shots_left(s, "first-contact") == SHOTS_PER_LEVEL - 1


def test_blocked_shots_are_free_and_do_not_call_model():
    s = new_session()
    eng = FakeEngine({})
    clock = Clock()
    for text in ["", "   ", "hi", "please refund my invoice now"]:
        turn = play(s, "first-contact", text, eng, now=clock)
        assert turn.kind == "blocked"
    assert eng.calls == 0 and shots_left(s, "first-contact") == SHOTS_PER_LEVEL


def test_locked_level_and_unknown_level():
    s = new_session()
    assert play(s, "grand-finale", "text text text", FakeEngine({}), now=Clock()).kind == "blocked"
    assert play(s, "nope", "text text text", FakeEngine({}), now=Clock()).kind == "blocked"


def test_rate_limit():
    s = new_session()
    eng = FakeEngine({"department": ans_choice({"billing": 0.1, "technical": 0.8, "sales": 0.05, "other": 0.05})})
    t = [100.0]
    fixed = lambda: t[0]
    assert play(s, "first-contact", "The screen shows nothing at all", eng, now=fixed).kind == "judged"
    assert play(s, "first-contact", "The screen shows nothing at all", eng, now=fixed).kind == "blocked"


def test_shots_run_out_and_reset():
    s = new_session()
    eng = FakeEngine({"department": ans_choice({"billing": 0.1, "technical": 0.8, "sales": 0.05, "other": 0.05})})
    clock = Clock()
    for _ in range(SHOTS_PER_LEVEL):
        assert play(s, "first-contact", "The screen shows nothing at all", eng, now=clock).kind == "judged"
    assert play(s, "first-contact", "The screen shows nothing at all", eng, now=clock).kind == "blocked"
    reset_level(s, "first-contact")
    assert shots_left(s, "first-contact") == SHOTS_PER_LEVEL


def test_non_english_requirement_blocks_english():
    s = new_session()
    s["unlocked"] = 10
    turn = play(s, "babel", "The website is down since this morning", FakeEngine({}, model="english"), now=Clock())
    assert turn.kind == "blocked" and "English" in turn.message


def test_over_limit_is_blocked_not_silently_clipped():
    s = new_session()
    turn = play(s, "first-contact", "a" * 500, FakeEngine({}), now=Clock())
    assert turn.kind == "blocked"


# ---------------------------------------------------------------- hints and onboarding

def test_hint_caps_stars_at_two():
    lvl = BY_ID["tick-tock"]
    answers = {"urgent": ans_noul(0.99)}
    assert judge(lvl, answers, 0).stars == 3
    assert judge(lvl, answers, 0, hint_used=True).stars == 2
    assert judge(lvl, {"urgent": ans_noul(0.2)}, 0, hint_used=True).stars == 0


def test_using_a_hint_records_it_and_caps_the_case():
    s = new_session()
    assert not used_hint(s, "fog")
    assert use_hint(s, "fog") == BY_ID["fog"].hint
    assert used_hint(s, "fog")
    eng = FakeEngine({"department": ans_choice({"billing": 0.4, "technical": 0.38, "sales": 0.12, "other": 0.1})})
    turn = play(s, "fog", "x", eng, now=Clock())  # locked, so nothing is judged
    assert turn.kind == "blocked"
    assert use_hint(s, "nope") == ""


def test_welcome_and_hint_flags_survive_repair():
    s = new_session()
    assert s["welcome_seen"] is False
    dismiss_welcome(s)
    use_hint(s, "fog")
    fixed = repair(s)
    assert fixed["welcome_seen"] is True
    assert fixed["hints"] == {"fog": 1}
    tampered = repair({"welcome_seen": "yes", "hints": {"not-a-level": 1, "fog": 1}})
    assert tampered["welcome_seen"] is True and tampered["hints"] == {"fog": 1}
    assert repair(None)["hints"] == {} and repair(None)["welcome_seen"] is False


def test_hint_is_a_nudge_not_the_answer():
    for lvl in LEVELS:
        assert lvl.hint and lvl.hint != lvl.example
        assert lvl.example.lower() not in lvl.hint.lower()
        assert lvl.use_case.endswith("."), lvl.id


def test_repair_rejects_tampered_state():
    s = repair({"best": {"first-contact": 90, "fake": 9999}, "unlocked": 999, "stars": "x", "last_ts": "y"})
    assert s["best"] == {"first-contact": 90} and s["unlocked"] == 10 and s["stars"] == {} and s["last_ts"] == 0.0
    assert s["hints"] == {} and s["welcome_seen"] is False
    assert repair(None)["unlocked"] == 1


def test_share_text():
    s = new_session()
    s["stars"]["first-contact"] = 2
    s["best"]["first-contact"] = 88
    out = share_text(s)
    assert "1/10" in out and "88 points" in out


# ---------------------------------------------------------------- lab validation

def test_lab_validation_errors():
    with pytest.raises(LabError):
        validate_questions("{")
    with pytest.raises(LabError):
        validate_questions("[]")
    with pytest.raises(LabError):
        validate_questions(json.dumps({"bad id!": {"type": "noul", "instructions": "x"}}))
    with pytest.raises(LabError):
        validate_questions(json.dumps({"q": {"type": "choice", "instructions": "x", "criteria": ["only"]}}))
    with pytest.raises(LabError):
        validate_questions(json.dumps({"q": {"type": "score", "instructions": "x", "criteria": "no"}}))
    with pytest.raises(LabError):
        validate_questions(json.dumps({f"q{i}": {"type": "noul", "instructions": "x"} for i in range(13)}))


def test_lab_validation_normalises_list_criteria():
    q = validate_questions(json.dumps({"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}))
    assert q["q"]["criteria"] == {"a": None, "b": None}


def test_parse_state():
    assert parse_state("hello", "message") == {"message": "hello"}
    assert parse_state('{"subject": "Hi", "body": "x"}', "message") == {"subject": "Hi", "body": "x"}
    assert parse_state("{not json", "message") == {"message": "{not json"}
    with pytest.raises(LabError):
        parse_state("   ", "message")


# ---------------------------------------------------------------- rendering safety

def test_render_escapes_model_and_user_text():
    lvl = BY_ID["first-contact"]
    outcome = judge(lvl, {"department": ans_choice({"billing": 0.9, "other": 0.1})}, 0)
    html = render.verdict(lvl, outcome, 10.0, {"model": "<b>x</b>", "reason": "<script>alert(1)</script>"}, 5, 4, True)
    assert "<script>" not in html and "&lt;script&gt;" in html
    lab = render.lab({"<img src=x>": ans_noul(0.5)}, 1.0, {"model": "m", "reason": "r"}, 1)
    assert "<img" not in lab


def test_parse_checkpoints():
    assert parse_checkpoints("english, Multilingual ,english") == ["english", "multilingual"]
    with pytest.raises(ValueError):
        parse_checkpoints("gpt")
    with pytest.raises(ValueError):
        parse_checkpoints(" , ")
