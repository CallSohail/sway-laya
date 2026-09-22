"""HTML fragments. Every piece of user or model text goes through esc()."""
from __future__ import annotations

import html
from typing import Any, Dict, Iterable, Optional

from .rules import Check, Level, Outcome


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _md_code(text: str) -> str:
    """Tiny formatter: `x` -> <code>x</code>, everything else escaped."""
    parts = esc(text).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


TOOLTIPS = {
    "Decision time": "Time the engine spent answering every question in one forward pass.",
    "Checkpoint": "Which Laya checkpoint answered: english or multilingual.",
    "Tokens read": "How many input tokens the engine read. Nothing is generated, so this is the whole cost.",
    "Why this checkpoint": "The router detects script and language before the model runs. This is what it saw.",
    "Questions": "How many typed questions were answered in this single pass.",
}


def _telemetry(rows: Iterable[tuple]) -> str:
    """rows: (label, value, wide). Labels carry a title= tooltip when we have one."""
    out = []
    for label, value, wide in rows:
        tip = TOOLTIPS.get(label, "")
        attr = f' title="{esc(tip)}"' if tip else ""
        out.append(f'<div class="{"wide" if wide else ""}"><dt{attr}>{esc(label)}</dt>'
                   f"<dd>{esc(value)}</dd></div>")
    return f'<dl class="telemetry">{"".join(out)}</dl>'


def stars(n: int, total: int = 3) -> str:
    return (f'<span class="stars" aria-label="{n} of {total} stars">'
            + "".join('<i class="on">★</i>' if i < n else '<i>★</i>' for i in range(total)) + "</span>")


def notice(text: str, kind: str = "info") -> str:
    return f'<div class="notice {esc(kind)}" role="status">{esc(text)}</div>'


def level_card(level: Level, number: int, shots_left: int, best: Optional[int]) -> str:
    rules = [f"Up to {level.max_chars} characters."]
    if level.require_non_english:
        rules.append("Must be written in a language other than English.")
    banned = "".join(f'<span class="chip">{esc(w)}</span>' for w in level.banned)
    goals = "".join(f"<li>{esc(o.label)}</li>" for o in level.objectives)
    best_txt = f"Best {best}" if best is not None else "Not solved yet"
    return f"""
<article class="case">
  <header class="case-head">
    <span class="case-no">Case {number}</span>
    <h2>{esc(level.title)}</h2>
    <span class="case-meta">{esc(best_txt)} · {shots_left} shot{'s' if shots_left != 1 else ''} left</span>
  </header>
  <p class="brief">{esc(level.brief)}</p>
  {f'<p class="usecase"><span>Where this is used</span> {esc(level.use_case)}</p>' if level.use_case else ''}
  <div class="cols">
    <div><h3>Win when</h3><ul class="goals">{goals}</ul></div>
    <div><h3>Rules</h3><ul class="rules">{''.join(f'<li>{esc(r)}</li>' for r in rules)}</ul>
      {f'<h3>Off limits</h3><div class="chips">{banned}</div>' if banned else ''}
    </div>
  </div>
</article>"""


def _bar(check: Check) -> str:
    lo, hi = check.target
    value = max(0.0, min(1.0, check.value))
    return f"""
<div class="bar" aria-hidden="true">
  <span class="band" style="left:{lo*100:.1f}%;width:{max(0.5,(hi-lo)*100):.1f}%"></span>
  <span class="fill {'ok' if check.passed else 'no'}" style="width:{value*100:.1f}%"></span>
</div>"""


def verdict(level: Level, outcome: Outcome, latency_ms: float, routing: Dict[str, Any],
            tokens: int, shots_left: int, is_new_best: bool, hint_used: bool = False) -> str:
    rows = "".join(f"""
<li class="check {'ok' if c.passed else 'no'}">
  <div class="check-top"><span class="mark">{'✓' if c.passed else '✗'}</span>
    <span class="label">{esc(c.label)}</span><span class="detail">{esc(c.detail)}</span></div>
  {_bar(c)}
</li>""" for c in outcome.checks)
    stamp = "Solved" if outcome.passed else "Not yet"
    telemetry = _telemetry([("Decision time", f"{latency_ms:.0f} ms", False),
                            ("Checkpoint", routing.get("model", "?"), False),
                            ("Tokens read", tokens, False),
                            ("Why this checkpoint", routing.get("reason", ""), True)])
    footer = []
    if outcome.passed:
        footer.append(f'<p class="lesson"><strong>What this shows.</strong> {_md_code(level.lesson)}</p>')
    if hint_used:
        footer.append('<p class="hint">You used the hint on this case, so it is capped at two stars.</p>')
    elif shots_left == 0:
        footer.append('<p class="hint">Out of shots. Reset the case to try again; your best score is kept.</p>')
    else:
        footer.append('<p class="hint">Bars show where the engine landed; the shaded zone is the target. '
                      'Rephrase and try again.</p>')
    return f"""
<section class="verdict {'ok' if outcome.passed else 'no'}" aria-live="polite">
  <div class="verdict-head">
    <span class="stamp">{stamp}</span>
    <span class="score">{outcome.score}<small>/100</small></span>
    {stars(outcome.stars)}
    {'<span class="newbest">New best</span>' if is_new_best else ''}
  </div>
  <ul class="checks">{rows}</ul>
  {telemetry}
  {''.join(footer)}
</section>"""


def progress(levels: Iterable[Level], best: Dict[str, int], star_map: Dict[str, int], unlocked: int) -> str:
    items = []
    total = 0
    for i, lvl in enumerate(levels):
        b = best.get(lvl.id)
        total += b or 0
        state = "locked" if i >= unlocked else ("done" if star_map.get(lvl.id, 0) else "open")
        right = stars(star_map.get(lvl.id, 0)) if state != "locked" else '<span class="lock">locked</span>'
        items.append(f'<li class="{state}"><span class="n">{i+1}</span>'
                     f'<span class="t">{esc(lvl.title)}</span>{right}</li>')
    return (f'<section class="progress"><div class="total"><span>Total</span><b>{total}</b></div>'
            f'<ol>{"".join(items)}</ol></section>')


def lab(answers: Dict[str, Any], latency_ms: float, routing: Dict[str, Any], tokens: int) -> str:
    blocks = []
    for qid, a in answers.items():
        kind = a.get("type")
        conf = float(a.get("confidence", 0.0))
        if kind == "noul":
            p = float(a["noul"])
            head = f"{p:.2f} probability of yes"
            bars = _prob_rows({"yes": p, "no": 1 - p}, "yes" if p >= 0.5 else "no")
        elif kind == "choice":
            head = f"{a['choice']}"
            bars = _prob_rows(a["probabilities"], a["choice"])
        else:
            legend = a.get("legend", {})
            head = f"expected level {float(a['score']):.2f}"
            probs = {f"{k} · {legend.get(k, '')}": v for k, v in a["probabilities"].items()}
            top = max(probs, key=probs.get) if probs else ""
            bars = _prob_rows(probs, top)
        blocks.append(f"""
<div class="qa"><div class="qa-head"><code>{esc(qid)}</code><span class="type">{esc(kind)}</span>
<b>{esc(head)}</b><span class="conf">confidence {conf:.2f}</span></div>{bars}</div>""")
    return f"""
<section class="labout">
  {_telemetry([("Questions", len(answers), False),
               ("Decision time", f"{latency_ms:.0f} ms", False),
               ("Checkpoint", routing.get("model", "?"), False),
               ("Tokens read", tokens, False),
               ("Why this checkpoint", routing.get("reason", ""), True)])}
  {''.join(blocks)}
</section>"""


def _prob_rows(probs: Dict[str, float], top: str) -> str:
    rows = []
    for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        rows.append(f'<div class="pr {"top" if label == top else ""}"><span class="pl">{esc(label)}</span>'
                    f'<span class="pb"><span style="width:{max(0.0, min(1.0, float(p)))*100:.1f}%"></span></span>'
                    f'<span class="pv">{float(p):.2f}</span></div>')
    return f'<div class="probs">{"".join(rows)}</div>'


# ---------------------------------------------------------------- onboarding

def welcome() -> str:
    return """
<section class="welcome">
  <h2>New here?</h2>
  <p>SWAY is a word game against a decision engine. The engine never writes text: it reads your
  message and answers a fixed set of typed questions about it in one pass.</p>
  <ol>
    <li>Read the case. It tells you what the engine's answers have to say, and which words are off limits.</li>
    <li>Write a message that makes the engine answer that way. You get five shots; blocked shots are free.</li>
    <li>Read the bars in the verdict. They show where the engine landed against the target, so you know what to change.</li>
  </ol>
</section>"""


def hint(level: Level) -> str:
    if not level.hint:
        return ""
    return (f'<div class="hintbox"><span>Hint</span><p>{esc(level.hint)}</p>'
            "<p class=\"cap\">This case is now capped at two stars.</p></div>")


# ---------------------------------------------------------------- autoplay

def autoplay_now(level_title: str, attempt: int, text: str, note: str = "") -> str:
    """The candidate currently being judged."""
    if not level_title:
        return ""
    body = f'<p class="cand">{esc(text)}</p>' if text else '<p class="cand quiet">Composing candidates…</p>'
    return f"""
<section class="nowplaying">
  <div class="np-head"><span class="np-case">{esc(level_title)}</span>
    <span class="np-att">attempt {attempt}</span></div>
  {body}
  {f'<p class="np-note">{esc(note)}</p>' if note else ''}
</section>"""


def autoplay_table(rows: Iterable[Any]) -> str:
    """rows are autoplay Event objects of kind 'attempt', newest last."""
    rows = list(rows)
    if not rows:
        return ""
    out = []
    for ev in rows:
        if ev.blocked:
            result = f'<span class="blocked">blocked · {esc(ev.blocked)}</span>'
            mark = "—"
            cls = "row blocked"
        else:
            result = f'<span class="num">{ev.score}</span> <span class="ms">{ev.latency_ms:.0f} ms</span>'
            mark = "✓" if ev.passed else "✗"
            cls = "row " + ("ok" if ev.passed else "no")
        out.append(f'<tr class="{cls}"><td class="n">{ev.attempt}</td>'
                   f'<td class="txt">{esc(ev.text)}</td><td class="res">{result}</td>'
                   f'<td class="mk">{mark}</td></tr>')
    return f"""
<div class="attempts">
  <table>
    <thead><tr><th>#</th><th>Candidate</th><th>Result</th><th>Pass</th></tr></thead>
    <tbody>{''.join(out)}</tbody>
  </table>
</div>"""


def autoplay_summary(summary: Any, human_best: Dict[str, int], levels: Iterable[Level]) -> str:
    """Final card. human_best is read from the player's session and never changed."""
    by_id = {lvl.id: lvl for lvl in levels}
    machine_total = human_total = 0
    rows = []
    for level_id, score in summary.best.items():
        lvl = by_id.get(level_id)
        mine = int(human_best.get(level_id, 0))
        machine_total += score
        human_total += mine
        rows.append(f'<tr><td>{esc(lvl.title if lvl else level_id)}</td>'
                    f'<td class="num">{mine or "—"}</td><td class="num">{score}</td></tr>')
    stats = [("Cases solved", f"{summary.solved} of {summary.attempted}"),
             ("Average best score", f"{summary.average_best:.0f}"),
             ("Model calls", str(summary.calls)),
             ("Average decision time", f"{summary.average_decision_ms:.0f} ms"),
             ("Total time", f"{summary.seconds:.0f} s")]
    cells = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in stats)
    stopped = f'<p class="np-note">{esc(summary.stopped)}</p>' if summary.stopped else ""
    return f"""
<section class="runsummary">
  <h3>Run summary</h3>
  <dl class="telemetry">{cells}</dl>
  {stopped}
  <h3>Human vs machine</h3>
  <table class="versus">
    <thead><tr><th>Case</th><th>Your best</th><th>Autoplay best</th></tr></thead>
    <tbody>{''.join(rows)}
      <tr class="tot"><td>Total</td><td class="num">{human_total}</td><td class="num">{machine_total}</td></tr>
    </tbody>
  </table>
</section>"""


def autoplay_best(level: Level, best: Any) -> str:
    """The best candidate found so far, with the engine's own verdict under it."""
    if best is None or best.outcome is None or best.verdict is None:
        return ""
    v = best.verdict
    head = (f'<section class="nowplaying"><div class="np-head">'
            f'<span class="np-case">Best message</span>'
            f'<span class="np-att">{best.score} / 100</span></div>'
            f'<p class="cand">{esc(best.text)}</p></section>')
    return head + verdict(level, best.outcome, v.latency_ms, v.routing, v.input_tokens, 1, False)
