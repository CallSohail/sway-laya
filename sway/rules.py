"""Pure game logic: no model, no UI. Everything here is unit-tested."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- input hygiene

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e"), None)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})


def sanitize(text: Optional[str], max_chars: int) -> str:
    """Normalise what the player typed before anything else sees it.

    NFKC folds look-alike characters, zero-width and control characters are dropped,
    whitespace runs collapse, and the result is clipped to max_chars.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).translate(_ZERO_WIDTH)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def _skeleton(text: str) -> str:
    """Lowercase, accent-free, leet-decoded letters only. 'R.3-f u_n D' -> 'refund'."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_LEET)
    return "".join(ch for ch in text if ch.isalnum())


def _tokens(text: str) -> List[str]:
    """Word skeletons, with spaced-out letters ('r e f u n d') glued back together."""
    raw = [_skeleton(t) for t in re.split(r"\s+", text)]
    out: List[str] = []
    run = ""
    for tok in raw:
        if len(tok) == 1:
            run += tok
            continue
        if run:
            out.append(run)
            run = ""
        if tok:
            out.append(tok)
    if run:
        out.append(run)
    return out


def find_banned(text: str, banned: Sequence[str]) -> List[str]:
    """Return banned words found in text, spelled out, spaced out, punctuated or leet-spelled.

    Stems of 5+ letters match anywhere (so 'cancel' also catches 'cancellation' and
    'c.a.n.c.e.l'). Short stems only match whole words or a plural, so 'now' does not
    fire on 'know' or 'snow'.
    """
    if not banned or not text:
        return []
    tokens = _tokens(text)
    joined = "".join(tokens)
    hits: List[str] = []
    for word in banned:
        stem = _skeleton(word)
        if not stem or word in hits:
            continue
        if len(stem) >= 5:
            found = stem in joined
        else:
            found = any(t == stem or t == stem + "s" for t in tokens)
        if found:
            hits.append(word)
    return hits


def letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


# ---------------------------------------------------------------- objectives

@dataclass
class Check:
    label: str          # what the player sees
    passed: bool
    quality: float      # 0..1, partial credit
    value: float        # the measured number to draw on a bar (0..1 scale)
    target: Tuple[float, float]  # accepted band on the same 0..1 scale
    detail: str = ""


@dataclass
class Objective:
    """One condition on one answer.

    kind:
      choice   answer[q].choice == label, optional min probability
      above    noul probability >= t
      below    noul probability <= t
      band     score expected value within [lo, hi]
      split    choice top two are {a, b} and their gap <= max_gap (a calibrated tie)
    """
    kind: str
    q: str
    label: str
    arg: Any = None
    t: float = 0.5
    lo: float = 0.0
    hi: float = 0.0
    max_gap: float = 0.2

    def evaluate(self, answers: Dict[str, Any], n_levels: Optional[int] = None) -> Check:
        a = answers.get(self.q)
        if a is None:
            return Check(self.label, False, 0.0, 0.0, (0, 1), "no answer")
        if self.kind == "choice":
            p = float(a["probabilities"].get(self.arg, 0.0))
            ok = a["choice"] == self.arg and p >= self.t
            quality = _ramp(p, self.t) if not ok else 0.7 + 0.3 * _unit(p, self.t, 1.0)
            return Check(self.label, ok, quality, p, (self.t, 1.0),
                         f"picked {a['choice']} · p({self.arg}) = {p:.2f}")
        if self.kind == "above":
            p = float(a["noul"])
            ok = p >= self.t
            quality = 0.7 + 0.3 * _unit(p, self.t, 1.0) if ok else _ramp(p, self.t)
            return Check(self.label, ok, quality, p, (self.t, 1.0), f"p = {p:.2f}")
        if self.kind == "below":
            p = float(a["noul"])
            ok = p <= self.t
            quality = 0.7 + 0.3 * _unit(self.t - p, 0.0, self.t) if ok else _ramp(1 - p, 1 - self.t)
            return Check(self.label, ok, quality, p, (0.0, self.t), f"p = {p:.2f}")
        if self.kind == "band":
            top = max(1, (n_levels or len(a.get("probabilities", {})) or 2) - 1)
            s = float(a["score"])
            ok = self.lo <= s <= self.hi
            if ok:
                if self.hi >= top:        # "at least lo": deeper is better
                    quality = 0.7 + 0.3 * _unit(s, self.lo, self.lo + 0.5)
                elif self.lo <= 0:        # "at most hi": lower is better
                    quality = 0.7 + 0.3 * _unit(self.hi - s, 0.0, 0.5)
                else:                     # true band: centre is best
                    mid, half = (self.lo + self.hi) / 2, max(1e-6, (self.hi - self.lo) / 2)
                    quality = 0.7 + 0.3 * (1 - abs(s - mid) / half)
            else:
                dist = self.lo - s if s < self.lo else s - self.hi
                quality = max(0.0, 0.6 * (1 - dist / top))
            return Check(self.label, ok, quality, s / top, (self.lo / top, self.hi / top),
                         f"score {s:.2f} of {top}")
        if self.kind == "split":
            first, second = self.arg
            probs = a["probabilities"]
            ranked = sorted(probs, key=probs.get, reverse=True)
            pa, pb = float(probs.get(first, 0)), float(probs.get(second, 0))
            gap = abs(pa - pb)
            top_two = set(ranked[:2]) == {first, second}
            ok = top_two and gap <= self.max_gap
            closeness = max(0.0, 1 - gap)
            quality = (0.7 + 0.3 * (1 - gap / self.max_gap)) if ok else 0.6 * closeness * (1 if top_two else 0.5)
            return Check(self.label, ok, quality, 1 - gap, (1 - self.max_gap, 1.0),
                         f"{first} {pa:.2f} vs {second} {pb:.2f}")
        raise ValueError(f"unknown objective kind {self.kind!r}")


def _unit(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _ramp(x: float, target: float) -> float:
    """Partial credit for a miss: 0.6 at the threshold, 0 at the far end."""
    if target <= 0:
        return 0.6
    return max(0.0, min(0.6, 0.6 * x / target))


# ---------------------------------------------------------------- levels

@dataclass
class Level:
    id: str
    title: str
    brief: str                 # the scenario, one or two sentences
    lesson: str                # what this teaches about the engine, shown after a pass
    state_key: str
    questions: Dict[str, Any]
    objectives: List[Objective]
    max_chars: int = 240
    min_letters: int = 8
    banned: List[str] = field(default_factory=list)
    require_non_english: bool = False
    extra_state: Dict[str, str] = field(default_factory=dict)
    example: str = ""          # a known passing answer, used by tests, never shown

    def build_state(self, text: str) -> Dict[str, str]:
        state = dict(self.extra_state)
        state[self.state_key] = text
        return state

    def score_levels(self, qid: str) -> Optional[int]:
        q = self.questions.get(qid, {})
        return len(q["criteria"]) if q.get("type") == "score" else None


@dataclass
class Gate:
    ok: bool
    reason: str = ""


def pre_check(level: Level, text: str, route_reason: str = "", routed_model: str = "") -> Gate:
    """Checks that do not need the model. Failing here costs no shot."""
    if not text.strip():
        return Gate(False, "Write a message first.")
    if letter_count(text) < level.min_letters:
        return Gate(False, f"Too short. Use at least {level.min_letters} letters so the engine has something to read.")
    hits = find_banned(text, level.banned)
    if hits:
        return Gate(False, "Banned word spotted: " + ", ".join(hits) + ". Find another way to say it.")
    if level.require_non_english and routed_model == "english":
        return Gate(False, "The router hears this as English (" + route_reason +
                    "). Write in another language, and give it a full sentence so detection is sure.")
    return Gate(True)


@dataclass
class Outcome:
    checks: List[Check]
    passed: bool
    score: int
    stars: int


def judge(level: Level, answers: Dict[str, Any], shot_index: int) -> Outcome:
    """Score one attempt. shot_index is 0 for the first shot."""
    checks = [o.evaluate(answers, level.score_levels(o.q)) for o in level.objectives]
    passed = all(c.passed for c in checks)
    base = sum(c.quality for c in checks) / max(1, len(checks))
    score = round(100 * base)
    if passed:
        score = max(score, 70)
    else:
        score = min(score, 69)
    stars = 0
    if passed:
        stars = 1 + (score >= 85) + (score >= 95 and shot_index == 0)
    return Outcome(checks, passed, score, stars)
