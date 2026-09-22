---
title: SWAY
emoji: 🎯
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: A word game against the Laya decision engine
models:
  - convaiinnovations/laya
---

# SWAY

A word game played against [Laya](https://github.com/NandhaKishorM/laya), an open non-autoregressive
decision engine. Each case sets a goal on the engine's answers. You write the message, the engine judges it
in one forward pass, and you see every probability it produced.

The cases are real production workflows: support triage, content moderation, LLM guardrails, model routing
and email security. Playing them is a hands-on way to see what typed decisions, calibrated confidence and
language routing do.

## The cases

| # | Case | Goal |
|---|---|---|
| 1 | First contact | Get routed to billing without billing words |
| 2 | Cold anger | Sound clearly annoyed while staying non-toxic |
| 3 | The quiet exit | Trigger churn risk while sounding calm |
| 4 | Tick tock | Convey urgency with no alarm words |
| 5 | Babel desk | Report an outage in a language other than English |
| 6 | Fog machine | Make billing and technical tie |
| 7 | Gatekeeper | Ask a real security question the guardrail does not flag |
| 8 | Short and hard | Get rated hard in 70 characters or fewer |
| 9 | Real, not phish | Write an urgent IT email that is not flagged as phishing |
| 10 | Grand finale | A calm non-English refund request with no churn and no toxicity |

Five shots per case. A shot blocked by a rule (banned word, length, language) is free. Scores reward
passing with margin; three stars need a strong pass on the first shot. Progress is kept in the browser.

The Lab tab runs any preset or custom question set on any input and shows the full answer.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements-dev.txt
python app.py            # http://localhost:7860
```

The first start downloads the checkpoints (about 1.5 GB for English plus multilingual).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LAYA_CHECKPOINTS` | `english,multilingual` | Checkpoints to load. `multilingual` alone runs in about 2.5 GB RAM |
| `LAYA_DEVICE` | auto | `cpu`, `cuda` or `mps` |
| `LAYA_LOW_MEMORY` | `1` | Build models on the meta device to cut peak RAM during load |
| `SWAY_CONCURRENCY` | `2` | Parallel model calls |
| `SWAY_SKIP_ENGINE` | unset | `1` starts the UI without loading models (CI) |

## Tests

```bash
pytest -q tests/test_logic.py                                   # fast, no model
SWAY_MODEL_TESTS=1 pytest -q tests/test_levels_model.py         # every case has a known passing answer
```

## API

The Lab is exposed as `/decide`:

```python
from gradio_client import Client
c = Client("<you>/sway")
html, raw = c.predict("My parcel never arrived", '{"late": {"type": "noul", "instructions": "Is the order late?"}}',
                      api_name="/decide")
```

## Deploy

Pushing to `main` runs the tests and syncs the repo to the Space when the `HF_TOKEN` secret and the
`HF_SPACE` variable (for example `yourname/sway`) are set in the GitHub repo settings.

## Credits

Laya by Convai Innovations, Apache 2.0. Game code MIT.
