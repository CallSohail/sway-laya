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
            tokens: int, shots_left: int, is_new_best: bool) -> str:
    rows = "".join(f"""
<li class="check {'ok' if c.passed else 'no'}">
  <div class="check-top"><span class="mark">{'✓' if c.passed else '✗'}</span>
    <span class="label">{esc(c.label)}</span><span class="detail">{esc(c.detail)}</span></div>
  {_bar(c)}
</li>""" for c in outcome.checks)
    stamp = "Solved" if outcome.passed else "Not yet"
    footer = []
    if outcome.passed:
        footer.append(f'<p class="lesson"><strong>What this shows.</strong> {_md_code(level.lesson)}</p>')
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
  <dl class="telemetry">
    <div><dt>Decision time</dt><dd>{latency_ms:.0f} ms</dd></div>
    <div><dt>Checkpoint</dt><dd>{esc(routing.get('model', '?'))}</dd></div>
    <div><dt>Tokens read</dt><dd>{tokens}</dd></div>
    <div class="wide"><dt>Why this checkpoint</dt><dd>{esc(routing.get('reason', ''))}</dd></div>
  </dl>
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
  <dl class="telemetry">
    <div><dt>Questions</dt><dd>{len(answers)}</dd></div>
    <div><dt>Decision time</dt><dd>{latency_ms:.0f} ms</dd></div>
    <div><dt>Checkpoint</dt><dd>{esc(routing.get('model', '?'))}</dd></div>
    <div><dt>Tokens read</dt><dd>{tokens}</dd></div>
    <div class="wide"><dt>Why this checkpoint</dt><dd>{esc(routing.get('reason', ''))}</dd></div>
  </dl>
  {''.join(blocks)}
</section>"""


def _prob_rows(probs: Dict[str, float], top: str) -> str:
    rows = []
    for label, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        rows.append(f'<div class="pr {"top" if label == top else ""}"><span class="pl">{esc(label)}</span>'
                    f'<span class="pb"><span style="width:{max(0.0, min(1.0, float(p)))*100:.1f}%"></span></span>'
                    f'<span class="pv">{float(p):.2f}</span></div>')
    return f'<div class="probs">{"".join(rows)}</div>'
