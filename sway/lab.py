"""Free-play lab: run any production question set on any input."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

MAX_QUESTIONS = 12
MAX_OPTIONS = 20
MAX_STATE_CHARS = 2000
MAX_QUESTIONS_JSON = 8000
_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")


def _presets() -> Dict[str, Tuple[str, Dict[str, Any], str]]:
    import laya

    return {
        "Support triage": ("message", laya.triage_questions(),
                           "I was billed twice this month. Please send the extra money back, I need it before rent is due."),
        "LLM guardrail": ("prompt", laya.guard_questions(),
                          "Summarise our refund policy for a customer who bought the annual plan."),
        "Content moderation": ("post", laya.moderation_questions(),
                               "Great write-up, though I think the second benchmark table is missing the CPU numbers."),
        "Model routing": ("request", laya.router_questions(),
                          "Refactor this Flask service to use dependency injection and add tests."),
        "Email security": ("body", laya.email_questions(),
                           "Your mailbox is full. Confirm your account details within 24 hours to avoid suspension."),
    }


PRESET_NAMES = ["Support triage", "LLM guardrail", "Content moderation", "Model routing", "Email security"]


def preset(name: str) -> Tuple[str, str, str]:
    """Return (state_key, questions_json, example_text) for a preset name."""
    table = _presets()
    key, questions, example = table.get(name, table[PRESET_NAMES[0]])
    return key, json.dumps(questions, indent=2, ensure_ascii=False), example


class LabError(ValueError):
    pass


def validate_questions(raw: str) -> Dict[str, Any]:
    if not raw or not raw.strip():
        raise LabError("Add at least one question.")
    if len(raw) > MAX_QUESTIONS_JSON:
        raise LabError(f"Question JSON is too long (limit {MAX_QUESTIONS_JSON} characters).")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LabError(f"Question JSON does not parse: line {exc.lineno}, column {exc.colno}: {exc.msg}.") from None
    if not isinstance(data, dict) or not data:
        raise LabError("Questions must be a JSON object mapping an id to a question.")
    if len(data) > MAX_QUESTIONS:
        raise LabError(f"Up to {MAX_QUESTIONS} questions per call.")
    clean: Dict[str, Any] = {}
    for qid, q in data.items():
        where = f"Question {qid!r}"
        if not _ID.match(str(qid)):
            raise LabError(f"{where}: ids use letters, digits and underscores, 40 characters max.")
        if not isinstance(q, dict):
            raise LabError(f"{where} must be an object.")
        qtype = q.get("type")
        if qtype not in ("choice", "score", "noul"):
            raise LabError(f"{where}: type must be choice, score or noul.")
        ins = q.get("instructions")
        if not isinstance(ins, str) or not ins.strip():
            raise LabError(f"{where}: instructions must be a non-empty string.")
        if len(ins) > 400:
            raise LabError(f"{where}: instructions are limited to 400 characters.")
        item: Dict[str, Any] = {"type": qtype, "instructions": ins.strip()}
        crit = q.get("criteria")
        if qtype == "choice":
            if isinstance(crit, list):
                crit = {str(c): None for c in crit}
            if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_OPTIONS:
                raise LabError(f"{where}: choice needs 2 to {MAX_OPTIONS} options in criteria.")
            opts = {}
            for label, desc in crit.items():
                label = str(label).strip()
                if not label or len(label) > 60:
                    raise LabError(f"{where}: option labels must be 1 to 60 characters.")
                if desc is not None and (not isinstance(desc, str) or len(desc) > 200):
                    raise LabError(f"{where}: option descriptions must be text up to 200 characters.")
                opts[label] = desc
            item["criteria"] = opts
        elif qtype == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10 or not all(isinstance(c, str) and c.strip() for c in crit):
                raise LabError(f"{where}: score needs a list of 2 to 10 level descriptions in criteria.")
            item["criteria"] = [c.strip()[:200] for c in crit]
        elif crit is not None:
            item["criteria"] = crit  # the email preset passes true/false descriptions; laya ignores them safely
        clean[str(qid)] = item
    return clean


def parse_state(text: str, key: str) -> Any:
    """Plain text is wrapped as {key: text}. A JSON object is passed through as-is."""
    text = (text or "").strip()
    if not text:
        raise LabError("Type some input for the engine to read.")
    if len(text) > MAX_STATE_CHARS:
        raise LabError(f"Input is limited to {MAX_STATE_CHARS} characters.")
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj:
            return {str(k): v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False)
                    for k, v in obj.items()}
    return {key: text}
