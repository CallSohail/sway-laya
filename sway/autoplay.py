"""The automatic player: a beam search over phrase fragments, judged by the engine.

Pure Python, no Gradio. The search composes candidate messages from
`autoplay_data.LIBRARY`, runs each through the same `sanitize` and `pre_check` a
human shot goes through, scores the survivors with the engine, keeps the best few
and mutates them. Blocked candidates cost no model call.

`play_level` and `play_run` are generators that yield one `Event` per step, so a UI
can stream the search as it happens.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .autoplay_data import LIBRARY, OPTIONAL, SLOTS, Phrasebook
from .engine import EngineNotReady
from .levels import BY_ID, LEVELS
from .rules import Level, Outcome, judge, pre_check, sanitize

POOL_SIZE = 8            # candidates composed before the first round of scoring
BEAM = 3                 # survivors carried into the next round
MUTATIONS_PER_PARENT = 2
RETRIES = 8              # attempts to find text this level has not tried yet
PASS_SCORE = 95          # stop the level as soon as a candidate reaches this
OPTIONAL_CHANCE = 0.7    # how often an optional slot is used when composing

BUDGETS: Dict[str, int] = {"Quick": 12, "Thorough": 30}
BUDGET_NAMES: List[str] = list(BUDGETS)
MAX_RUN_CALLS = 300
MAX_RUN_SECONDS = 360.0

ALL_CASES = "__all__"


def budget_calls(budget: Any) -> int:
    """Accept a budget name ('Quick') or a plain number of model calls."""
    if isinstance(budget, str):
        if budget not in BUDGETS:
            raise ValueError(f"unknown budget {budget!r}. Valid: {', '.join(BUDGET_NAMES)}")
        return BUDGETS[budget]
    n = int(budget)
    if n < 1:
        raise ValueError("a budget needs at least one model call")
    return n


# ---------------------------------------------------------------- candidates

@dataclass(frozen=True)
class Candidate:
    book: int                            # index into LIBRARY[level_id]
    picks: Tuple[Tuple[str, int], ...]   # (slot, fragment index), in SLOTS order
    text: str

    def slots(self) -> Dict[str, int]:
        return dict(self.picks)


def _render(book: Phrasebook, picks: Dict[str, int]) -> str:
    return " ".join(book.slots[s][picks[s]] for s in SLOTS if s in picks).strip()


def _fit(level: Level, book: Phrasebook, picks: Dict[str, int]) -> Dict[str, int]:
    """Drop optional slots, least important first, until the text fits the limit."""
    picks = dict(picks)
    for slot in OPTIONAL:
        if len(_render(book, picks)) <= level.max_chars:
            break
        picks.pop(slot, None)
    return picks


def _make(level: Level, book_index: int, book: Phrasebook, picks: Dict[str, int]) -> Candidate:
    picks = _fit(level, book, picks)
    ordered = tuple((s, picks[s]) for s in SLOTS if s in picks)
    return Candidate(book_index, ordered, _render(book, picks))


def books(level_id: str) -> List[Phrasebook]:
    library = LIBRARY.get(level_id)
    if not library:
        raise KeyError(f"no fragment library for level {level_id!r}")
    return library


def compose(level: Level, rng: random.Random, book_index: Optional[int] = None) -> Candidate:
    """Build one candidate from random fragment choices."""
    library = books(level.id)
    if book_index is None:
        book_index = rng.randrange(len(library))
    book = library[book_index]
    picks: Dict[str, int] = {}
    for slot in book.order():
        if slot in OPTIONAL and rng.random() > OPTIONAL_CHANCE:
            continue
        picks[slot] = rng.randrange(len(book.slots[slot]))
    if not picks:  # every slot was optional and every coin came up tails
        slot = book.order()[0]
        picks[slot] = rng.randrange(len(book.slots[slot]))
    return _make(level, book_index, book, picks)


def mutate(level: Level, parent: Candidate, rng: random.Random) -> Optional[Candidate]:
    """Swap one fragment, drop one optional slot, or add one back. None if nothing changed."""
    book = books(level.id)[parent.book]
    picks = parent.slots()
    present = list(picks)
    droppable = [s for s in present if s in OPTIONAL]
    absent = [s for s in book.order() if s not in picks and s in OPTIONAL]

    moves = ["swap"]
    if droppable and len(present) > 1:
        moves.append("drop")
    if absent:
        moves.append("add")
    move = rng.choice(moves)

    if move == "drop":
        picks.pop(rng.choice(droppable))
    elif move == "add":
        slot = rng.choice(absent)
        picks[slot] = rng.randrange(len(book.slots[slot]))
    else:
        slot = rng.choice(present)
        options = [i for i in range(len(book.slots[slot])) if i != picks[slot]]
        if not options:
            return None
        picks[slot] = rng.choice(options)

    child = _make(level, parent.book, book, picks)
    return None if child == parent else child


# ---------------------------------------------------------------- run accounting

class RunLimits:
    """Hard caps shared by every level in one run."""

    def __init__(self, max_calls: int = MAX_RUN_CALLS, max_seconds: float = MAX_RUN_SECONDS,
                 clock: Callable[[], float] = time.monotonic):
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.clock = clock
        self.calls = 0
        self.started = clock()

    def reached(self) -> str:
        """Empty while the run may continue, otherwise the reason it must stop."""
        if self.calls >= self.max_calls:
            return f"Run cap reached: {self.max_calls} model calls."
        if self.elapsed() >= self.max_seconds:
            return f"Run cap reached: {self.max_seconds / 60:.0f} minutes."
        return ""

    def spend(self) -> None:
        self.calls += 1

    def elapsed(self) -> float:
        return self.clock() - self.started


@dataclass
class Best:
    text: str = ""
    score: int = 0
    passed: bool = False
    outcome: Optional[Outcome] = None
    verdict: Any = None


@dataclass
class Summary:
    attempted: int = 0
    solved: int = 0
    best: Dict[str, int] = field(default_factory=dict)
    calls: int = 0
    decision_ms: float = 0.0
    seconds: float = 0.0
    stopped: str = ""

    @property
    def average_best(self) -> float:
        return sum(self.best.values()) / len(self.best) if self.best else 0.0

    @property
    def average_decision_ms(self) -> float:
        return self.decision_ms / self.calls if self.calls else 0.0


@dataclass
class Event:
    """One step of the search. `kind` says which fields matter."""

    kind: str                       # level-start | attempt | level-end | summary | notice
    level_id: str = ""
    level_title: str = ""
    attempt: int = 0                # 1-based, counts blocked candidates too
    text: str = ""
    blocked: str = ""               # why pre_check refused it, empty when it was scored
    score: int = 0
    passed: bool = False
    latency_ms: float = 0.0
    calls: int = 0                  # model calls used by the run so far
    budget_left: int = 0            # model calls left for this level
    best: Optional[Best] = None
    summary: Optional[Summary] = None
    message: str = ""


# ---------------------------------------------------------------- the search

def play_level(level: Level, engine, budget: Any = "Quick", rng: Optional[random.Random] = None,
               limits: Optional[RunLimits] = None) -> Iterator[Event]:
    """Search one level. Yields an Event per candidate, then a level-end event."""
    rng = rng or random.Random(42)
    limits = limits or RunLimits()
    allowance = budget_calls(budget)
    used = 0
    attempt = 0
    scored = 0
    best = Best()
    seen: set = set()
    stopped = ""

    yield Event("level-start", level.id, level.title, calls=limits.calls, budget_left=allowance)

    library = books(level.id)
    pool: List[Candidate] = []
    while len(pool) < POOL_SIZE:
        cand = compose(level, rng, rng.randrange(len(library)))
        if cand.text not in seen:
            seen.add(cand.text)
            pool.append(cand)
        elif len(seen) >= POOL_SIZE * 4:  # the library is small, stop hunting for new text
            break

    while pool and not stopped:
        survivors: List[Tuple[int, Candidate]] = []
        for cand in pool:
            text = sanitize(cand.text, level.max_chars)
            decision = engine.route(level.build_state(text)) if text else {"model": "", "reason": ""}
            gate = pre_check(level, text, decision.get("reason", ""), decision.get("model", ""))
            attempt += 1
            if not gate.ok:
                yield Event("attempt", level.id, level.title, attempt=attempt, text=text,
                            blocked=gate.reason, calls=limits.calls,
                            budget_left=allowance - used, best=best)
                continue

            stopped = limits.reached()
            if stopped:
                break
            if used >= allowance:
                stopped = f"Budget used: {allowance} model calls for this case."
                break

            try:
                verdict = engine.predict(level.build_state(text), level.questions)
            except EngineNotReady as exc:
                stopped = str(exc) or "The engine is still loading."
                yield Event("notice", level.id, level.title, message=stopped, calls=limits.calls)
                break
            except Exception as exc:  # a bad call must not take the whole run down
                stopped = f"The engine failed on this case: {type(exc).__name__}: {exc}"
                yield Event("notice", level.id, level.title, message=stopped, calls=limits.calls)
                break

            used += 1
            limits.spend()
            outcome = judge(level, verdict.answers, scored)
            scored += 1
            if outcome.score > best.score or (outcome.passed and not best.passed):
                best = Best(text, outcome.score, outcome.passed, outcome, verdict)
            yield Event("attempt", level.id, level.title, attempt=attempt, text=text,
                        score=outcome.score, passed=outcome.passed, latency_ms=verdict.latency_ms,
                        calls=limits.calls, budget_left=allowance - used, best=best)

            if outcome.score >= PASS_SCORE:
                stopped = f"Solved with {outcome.score} of 100."
                break
            survivors.append((outcome.score, cand))

        if stopped:
            break
        survivors.sort(key=lambda pair: -pair[0])
        pool = []
        for _, parent in survivors[:BEAM]:
            for _ in range(MUTATIONS_PER_PARENT):
                for _ in range(RETRIES):  # a mutation can land on text already tried
                    child = mutate(level, parent, rng)
                    if child is not None and child.text not in seen:
                        seen.add(child.text)
                        pool.append(child)
                        break
        if not pool:  # the beam is exhausted, compose fresh candidates instead
            for _ in range(RETRIES * BEAM):
                cand = compose(level, rng, rng.randrange(len(library)))
                if cand.text not in seen:
                    seen.add(cand.text)
                    pool.append(cand)
                if len(pool) >= BEAM:
                    break

    yield Event("level-end", level.id, level.title, attempt=attempt, calls=limits.calls,
                budget_left=allowance - used, best=best, message=stopped,
                score=best.score, passed=best.passed)


def play_run(level_ids: Sequence[str], engine, budget: Any = "Quick", seed: int = 42,
             limits: Optional[RunLimits] = None) -> Iterator[Event]:
    """Search several levels in order, then yield a summary event."""
    limits = limits or RunLimits()
    summary = Summary()
    for level_id in level_ids:
        level = BY_ID.get(level_id)
        if level is None:
            continue
        reached = limits.reached()
        if reached:
            summary.stopped = reached
            break
        # One stream per level so the same seed always replays the same search.
        rng = random.Random(f"{seed}:{level_id}")
        for event in play_level(level, engine, budget, rng, limits):
            yield event
            if event.kind == "level-end":
                summary.attempted += 1
                summary.best[level_id] = event.best.score if event.best else 0
                summary.solved += int(bool(event.best and event.best.passed))
            elif event.kind == "attempt" and not event.blocked:
                summary.decision_ms += event.latency_ms
            elif event.kind == "notice":
                summary.stopped = event.message
        if summary.stopped:
            break
    summary.calls = limits.calls
    summary.seconds = limits.elapsed()
    yield Event("summary", summary=summary, calls=limits.calls)


def case_choices() -> List[Tuple[str, str]]:
    """Dropdown choices for the Autoplay tab, 'All cases' first."""
    return [("All cases", ALL_CASES)] + [(f"{i + 1}. {lvl.title}", lvl.id) for i, lvl in enumerate(LEVELS)]


def resolve_cases(choice: Optional[str]) -> List[str]:
    if not choice or choice == ALL_CASES:
        return [lvl.id for lvl in LEVELS]
    return [choice] if choice in BY_ID else []
