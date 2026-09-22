"""Per-player session state and turn handling. UI-agnostic and unit-tested."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .levels import BY_ID, LEVELS, SHOTS_PER_LEVEL
from .rules import Outcome, judge, pre_check, sanitize

MIN_SECONDS_BETWEEN_SHOTS = 1.0


def new_session() -> Dict[str, Any]:
    return {"best": {}, "stars": {}, "shots": {}, "unlocked": 1, "last_ts": 0.0, "history": []}


def repair(session: Any) -> Dict[str, Any]:
    """Accept whatever came back from the browser and return a valid session."""
    base = new_session()
    if not isinstance(session, dict):
        return base
    for key in ("best", "stars", "shots"):
        value = session.get(key)
        if isinstance(value, dict):
            base[key] = {k: int(v) for k, v in value.items() if k in BY_ID and isinstance(v, (int, float))}
    unlocked = session.get("unlocked")
    if isinstance(unlocked, int):
        base["unlocked"] = max(1, min(len(LEVELS), unlocked))
    last = session.get("last_ts")
    base["last_ts"] = float(last) if isinstance(last, (int, float)) else 0.0
    return base


def level_number(level_id: str) -> int:
    return next(i for i, lvl in enumerate(LEVELS) if lvl.id == level_id) + 1


def is_unlocked(session: Dict[str, Any], level_id: str) -> bool:
    return level_id in BY_ID and level_number(level_id) <= session["unlocked"]


def shots_left(session: Dict[str, Any], level_id: str) -> int:
    return max(0, SHOTS_PER_LEVEL - session["shots"].get(level_id, 0))


def total_score(session: Dict[str, Any]) -> int:
    return sum(session["best"].values())


@dataclass
class Turn:
    kind: str                       # "blocked" (no shot used) | "judged" | "error"
    message: str = ""
    outcome: Optional[Outcome] = None
    verdict: Any = None
    new_best: bool = False
    text: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def play(session: Dict[str, Any], level_id: str, raw_text: str, engine,
         now: Callable[[], float] = time.monotonic) -> Turn:
    """Run one shot. Mutates session. Never raises for player mistakes."""
    level = BY_ID.get(level_id)
    if level is None:
        return Turn("blocked", "Pick a case first.")
    if not is_unlocked(session, level_id):
        return Turn("blocked", "Solve the earlier cases to open this one.")
    if shots_left(session, level_id) == 0:
        return Turn("blocked", "No shots left on this case. Reset it to try again.")
    t = now()
    if t - session["last_ts"] < MIN_SECONDS_BETWEEN_SHOTS:
        return Turn("blocked", "One shot per second. Take a breath and send again.")

    text = sanitize(raw_text, level.max_chars)
    clipped = raw_text is not None and len(sanitize(raw_text, 10_000)) > level.max_chars
    decision = engine.route(level.build_state(text)) if text else {"model": "", "reason": ""}
    gate = pre_check(level, text, decision["reason"], decision["model"])
    if not gate.ok:
        return Turn("blocked", gate.reason, text=text)
    if clipped:
        return Turn("blocked", f"That is over {level.max_chars} characters. Trim it down.", text=text)

    session["last_ts"] = t
    verdict = engine.predict(level.build_state(text), level.questions)
    shot_index = session["shots"].get(level_id, 0)
    session["shots"][level_id] = shot_index + 1
    outcome = judge(level, verdict.answers, shot_index)

    new_best = False
    if outcome.passed:
        if outcome.score > session["best"].get(level_id, -1):
            session["best"][level_id] = outcome.score
            new_best = True
        session["stars"][level_id] = max(session["stars"].get(level_id, 0), outcome.stars)
        session["unlocked"] = max(session["unlocked"], min(len(LEVELS), level_number(level_id) + 1))
    return Turn("judged", outcome=outcome, verdict=verdict, new_best=new_best, text=text)


def reset_level(session: Dict[str, Any], level_id: str) -> None:
    session["shots"].pop(level_id, None)


def share_text(session: Dict[str, Any]) -> str:
    solved = sum(1 for lvl in LEVELS if session["stars"].get(lvl.id))
    star_total = sum(session["stars"].values())
    line = "".join("★" * session["stars"].get(l.id, 0) + "☆" * (3 - session["stars"].get(l.id, 0)) + " "
                   for l in LEVELS).strip()
    return (f"SWAY: {solved}/{len(LEVELS)} cases, {star_total} stars, {total_score(session)} points\n{line}\n"
            "Played against the Laya decision engine.")
