"""SWAY: a word game against the Laya decision engine. Hugging Face Spaces entry point."""
from __future__ import annotations

# ZeroGPU Spaces refuse to start without a @spaces.GPU function. SWAY runs on CPU,
# so this stub is registered but never called and uses no GPU quota.
try:
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_stub():
        return None
except ImportError:  # not installed locally or in CI
    pass

import json
import logging
import os
import threading
from pathlib import Path

import gradio as gr

from sway import __version__, render
from sway.engine import EngineNotReady, get_engine
from sway.game import (level_number, new_session, play, repair, reset_level, share_text, shots_left)
from sway.lab import PRESET_NAMES, LabError, parse_state, preset, validate_questions
from sway.levels import BY_ID, LEVELS

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sway.app")

ENGINE = get_engine()
HIDDEN = "undocumented"  # the UI can call these; they stay out of the public API page
CSS = (Path(__file__).parent / "sway" / "style.css").read_text(encoding="utf-8")


def _start_engine():
    try:
        ENGINE.start()
        log.info("engine ready in %.1fs with %s", ENGINE.load_seconds or 0, ENGINE.checkpoints)
    except Exception:
        pass  # ENGINE.error is shown in the UI


if os.environ.get("SWAY_SKIP_ENGINE") != "1":
    threading.Thread(target=_start_engine, name="laya-loader", daemon=True).start()


# ---------------------------------------------------------------- helpers

def status_html() -> str:
    if ENGINE.ready:
        return render.notice(f"Engine ready · checkpoints: {', '.join(ENGINE.checkpoints)}", "ok")
    if ENGINE.error:
        return render.notice("The engine failed to load: " + ENGINE.error, "error")
    return render.notice("Loading the Laya checkpoints. The first start takes about a minute.", "info")


def level_choices(session):
    out = []
    for i, lvl in enumerate(LEVELS):
        if i >= session["unlocked"]:
            break
        s = session["stars"].get(lvl.id, 0)
        out.append((f"{i + 1}. {lvl.title}  {'★' * s}{'☆' * (3 - s)}", lvl.id))
    return out


def card(session, level_id):
    lvl = BY_ID[level_id]
    return render.level_card(lvl, level_number(level_id), shots_left(session, level_id),
                             session["best"].get(level_id))


def progress(session):
    return render.progress(LEVELS, session["best"], session["stars"], session["unlocked"])


def chars_label(text, level_id):
    lvl = BY_ID.get(level_id, LEVELS[0])
    n = len(text or "")
    cls = "over" if n > lvl.max_chars else ""
    return f'<div class="counter {cls}">{n} / {lvl.max_chars}</div>'


# ---------------------------------------------------------------- play handlers

def on_load(session):
    session = repair(session)
    first = LEVELS[0].id
    return (session, gr.update(choices=level_choices(session), value=first), card(session, first),
            "", progress(session), status_html(), chars_label("", first))


def on_pick(level_id, session):
    session = repair(session)
    if level_id not in BY_ID:
        level_id = LEVELS[0].id
    return card(session, level_id), "", chars_label("", level_id), gr.update(value="")


def on_submit(text, level_id, session):
    session = repair(session)
    try:
        turn = play(session, level_id, text, ENGINE)
    except EngineNotReady as exc:
        return (session, render.notice(str(exc) or "The engine is still loading.", "info"),
                card(session, level_id), progress(session), gr.update(), status_html(), gr.update(visible=False))
    except Exception:
        log.exception("turn failed")
        return (session, render.notice("Something went wrong while judging. Your shot was not counted.", "error"),
                card(session, level_id), progress(session), gr.update(), status_html(), gr.update(visible=False))

    if turn.kind != "judged":
        return (session, render.notice(turn.message, "warn"), card(session, level_id), progress(session),
                gr.update(), status_html(), gr.update(visible=False))

    v = turn.verdict
    html = render.verdict(BY_ID[level_id], turn.outcome, v.latency_ms, v.routing, v.input_tokens,
                          shots_left(session, level_id), turn.new_best)
    has_next = turn.outcome.passed and level_number(level_id) < len(LEVELS)
    return (session, html, card(session, level_id), progress(session),
            gr.update(choices=level_choices(session), value=level_id), status_html(), gr.update(visible=has_next))


def on_next(level_id, session):
    session = repair(session)
    idx = level_number(level_id)
    nxt = LEVELS[min(idx, len(LEVELS) - 1)].id
    return (gr.update(choices=level_choices(session), value=nxt), card(session, nxt), "", gr.update(value=""),
            gr.update(visible=False), chars_label("", nxt))


def on_reset(level_id, session):
    session = repair(session)
    if level_id in BY_ID:
        reset_level(session, level_id)
    return session, card(session, level_id), render.notice("Case reset. Five fresh shots.", "info")


def on_restart():
    session = new_session()
    first = LEVELS[0].id
    return (session, gr.update(choices=level_choices(session), value=first), card(session, first), "",
            progress(session), gr.update(value=""))


def on_share(session):
    return gr.update(value=share_text(repair(session)), visible=True)


# ---------------------------------------------------------------- lab handlers

def on_preset(name):
    key, questions, example = preset(name)
    return questions, example, key


def on_lab_run(text, questions_json, key):
    try:
        questions = validate_questions(questions_json)
        state = parse_state(text, key or "message")
        v = ENGINE.predict(state, questions)
    except LabError as exc:
        return render.notice(str(exc), "warn"), None
    except EngineNotReady as exc:
        return render.notice(str(exc) or "The engine is still loading.", "info"), None
    except ValueError as exc:  # laya: options exceed the head token budget
        return render.notice(f"The engine rejected these questions: {exc}. Shorten option text or use fewer options.",
                             "warn"), None
    except Exception:
        log.exception("lab run failed")
        return render.notice("Something went wrong running these questions.", "error"), None
    raw = {"answers": v.answers, "routing": v.routing, "latency_ms": round(v.latency_ms, 1),
           "input_tokens": v.input_tokens}
    return render.lab(v.answers, v.latency_ms, v.routing, v.input_tokens), raw


# ---------------------------------------------------------------- layout

ABOUT = """
### What you are playing against

[Laya](https://github.com/NandhaKishorM/laya) is an open, non-autoregressive decision engine. It does not write text.
You give it an input and a set of typed questions, and it answers all of them in one forward pass:

- `choice` picks one option and returns a probability for every option,
- `score` places the input on an ordered rubric and returns the expected level,
- `noul` returns a calibrated probability that a statement is true.

Because nothing is generated, there is nothing to parse and nothing to hallucinate. The same call that took
tens of milliseconds here is what a support desk, a moderation queue, an LLM guardrail or a model router would
run on every request.

### How the game works

Each case gives you a goal on the engine's answers and a few rules: a character limit, banned words, sometimes a
language requirement. You get five shots per case. A shot blocked by a rule is free. Scores reward passing with
margin; three stars need a strong pass on the first shot.

Non-English text is sent to the multilingual checkpoint by the router, which detects script and language before
the model runs. The reason is shown under every verdict.

### Limits worth knowing

The engine is a fast base model, not an oracle. Its scores come from one forward pass over your words, so
phrasing matters, and some quirks are part of the puzzle. Nothing you type is stored by this app.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="SWAY · a word game against a decision engine") as demo:
        session = gr.BrowserState(new_session(), storage_key="sway-session-v1")

        gr.HTML("""
<header class="masthead">
  <h1>SWAY</h1>
  <p>Write the message. The decision engine judges it in one pass. Make it decide what you need.</p>
</header>""")
        status = gr.HTML(status_html())

        with gr.Tabs():
            with gr.Tab("Play"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=7):
                        level = gr.Dropdown(label="Case", choices=[], interactive=True)
                        case_html = gr.HTML()
                        text = gr.Textbox(label="Your message", lines=4, max_lines=8,
                                          placeholder="Type the message the engine will read…",
                                          max_length=400, autofocus=True)
                        counter = gr.HTML()
                        with gr.Row():
                            send = gr.Button("Send shot", variant="primary")
                            reset = gr.Button("Reset case", variant="secondary")
                            nxt = gr.Button("Next case", variant="primary", visible=False)
                        verdict = gr.HTML()
                    with gr.Column(scale=4):
                        prog = gr.HTML()
                        with gr.Row():
                            share = gr.Button("Share result", size="sm")
                            restart = gr.Button("Start over", size="sm", variant="stop")
                        share_box = gr.Textbox(label="Copy and share", visible=False, lines=3,
                                               buttons=["copy"], interactive=False)

            with gr.Tab("Lab"):
                gr.Markdown("Run any production question set on any input. Pick a preset or edit the JSON.")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        preset_pick = gr.Dropdown(PRESET_NAMES, value=PRESET_NAMES[0], label="Preset")
                        key0, q0, ex0 = preset(PRESET_NAMES[0])
                        lab_key = gr.State(key0)
                        lab_text = gr.Textbox(label="Input (plain text, or a JSON object)", value=ex0,
                                              lines=5, max_lines=12, max_length=2000)
                        lab_q = gr.Code(label="Questions", value=q0, language="json", lines=16)
                        run = gr.Button("Run decisions", variant="primary")
                    with gr.Column(scale=6):
                        lab_out = gr.HTML()
                        with gr.Accordion("Raw response", open=False):
                            lab_raw = gr.JSON()

            with gr.Tab("How it works"):
                gr.Markdown(ABOUT)

        gr.HTML(f'<footer class="foot">SWAY {__version__} · built on '
                '<a href="https://huggingface.co/convaiinnovations/laya" target="_blank" rel="noopener">Laya</a>'
                ' (Apache 2.0)</footer>')

        # wiring
        demo.load(on_load, [session], [session, level, case_html, verdict, prog, status, counter], api_visibility=HIDDEN)
        level.input(on_pick, [level, session], [case_html, verdict, counter, text], api_visibility=HIDDEN)
        text.change(chars_label, [text, level], [counter], show_progress="hidden", queue=False, api_visibility=HIDDEN)
        submit_io = dict(inputs=[text, level, session],
                         outputs=[session, verdict, case_html, prog, level, status, nxt],
                         concurrency_id="engine", api_visibility=HIDDEN)
        send.click(on_submit, **submit_io)
        text.submit(on_submit, **submit_io)
        nxt.click(on_next, [level, session], [level, case_html, verdict, text, nxt, counter], api_visibility=HIDDEN)
        reset.click(on_reset, [level, session], [session, case_html, verdict], api_visibility=HIDDEN)
        restart.click(on_restart, None, [session, level, case_html, verdict, prog, share_box], api_visibility=HIDDEN)
        share.click(on_share, [session], [share_box], api_visibility=HIDDEN)
        preset_pick.change(on_preset, [preset_pick], [lab_q, lab_text, lab_key], api_visibility=HIDDEN)
        run.click(on_lab_run, [lab_text, lab_q, lab_key], [lab_out, lab_raw], concurrency_id="engine",
                  api_name="decide", api_description="Run typed Laya questions on an input.")
    return demo


demo = build()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=int(os.environ.get("SWAY_CONCURRENCY", "2")), max_size=64)
    demo.launch(
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        ),
        css=CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860"))),
        show_error=False,
        ssr_mode=False,
    )
