"""The campaign. Every level is plain data, so adding one needs no UI change.

Each level asks one real production question set (support triage, moderation,
LLM guardrails, model routing, email security) and the player writes the input.
The engine answers every question in a single forward pass.
"""
from __future__ import annotations

from typing import Dict, List

from .rules import Level, Objective as O

DEPARTMENT = {
    "type": "choice",
    "instructions": "Which department should handle `message`?",
    "criteria": {
        "billing": "invoices, payments, refunds, charges",
        "technical": "bugs, outages, errors, integrations",
        "sales": "pricing, new contracts, upgrades",
        "other": "everything else",
    },
}
FRUSTRATION = {
    "type": "score",
    "instructions": "How frustrated does the customer sound in `message`?",
    "criteria": ["calm and neutral", "concerned but civil", "clearly annoyed", "very angry or using strong language"],
}
TOXIC = {"type": "noul", "instructions": "Is `message` toxic: rude, insulting or disrespectful?"}
CHURN = {"type": "noul", "instructions": "Does `message` suggest the customer may leave for a competitor or cancel?"}
URGENT = {"type": "noul", "instructions": "Does `message` communicate time pressure or a deadline?"}
REFUND = {"type": "noul", "instructions": "Does the customer ask for money back?"}

LEVELS: List[Level] = [
    Level(
        id="first-contact",
        title="First contact",
        brief="A customer was charged for something they never ordered. Get the ticket to billing "
              "without using the obvious words.",
        lesson="`choice` questions return a probability for every option, not just a label. "
               "The engine routes on meaning, so synonyms work as well as keywords.",
        state_key="message",
        questions={"department": DEPARTMENT},
        objectives=[O("choice", "department", "Routed to billing", arg="billing", t=0.6)],
        max_chars=200,
        banned=["invoice", "bill", "billing", "charge", "charged", "payment", "refund", "money"],
        example="Hello, my card statement shows 49 euros taken by your company last week for an order I never placed.",
    ),
    Level(
        id="cold-anger",
        title="Cold anger",
        brief="Sound clearly annoyed, but stay polite. A moderation filter reads the same message.",
        lesson="One forward pass answered a support question and a moderation question together. "
               "`score` questions give an expected level on a rubric, so tone is measured, not guessed.",
        state_key="message",
        questions={"frustration": FRUSTRATION, "toxic": TOXIC},
        objectives=[
            O("band", "frustration", "Frustration at least 'clearly annoyed'", lo=1.9, hi=3.0),
            O("below", "toxic", "Not toxic", t=0.3),
        ],
        max_chars=220,
        banned=["angry", "furious", "annoyed", "frustrated", "hate", "stupid", "idiot", "useless", "worst", "terrible"],
        example="This is the third time I am writing about the same broken login. Nobody has answered. "
                "I am out of patience with this.",
    ),
    Level(
        id="quiet-exit",
        title="The quiet exit",
        brief="Make the churn detector fire while sounding perfectly calm.",
        lesson="`noul` questions return a calibrated probability, so a product can act at 0.8 and "
               "send the rest to a human. Calm customers who are about to leave are the ones rules miss.",
        state_key="message",
        questions={"churn_risk": CHURN, "frustration": FRUSTRATION},
        objectives=[
            O("above", "churn_risk", "Churn risk flagged", t=0.75),
            O("band", "frustration", "Calm tone (score up to 1.2)", lo=0.0, hi=1.2),
        ],
        max_chars=220,
        banned=["cancel", "leave", "leaving", "quit", "competitor", "unsubscribe", "terminate", "close my account"],
        example="Thanks for the help over the years. We have signed with another provider and will move our team there next month.",
    ),
    Level(
        id="tick-tock",
        title="Tick tock",
        brief="Convey time pressure without any of the usual alarm words.",
        lesson="The engine reads implied deadlines, not keywords. Short inputs are fine: "
               "cost scales with tokens, and this answer took one pass.",
        state_key="message",
        questions={"urgent": URGENT, "department": DEPARTMENT},
        objectives=[O("above", "urgent", "Urgency detected", t=0.7)],
        max_chars=140,
        banned=["urgent", "urgently", "asap", "now", "immediately", "today", "deadline", "emergency",
                "quickly", "hurry", "rush", "critical", "fast"],
        example="The contract must be signed before 5pm or we lose the grant. Please send it within the hour.",
    ),
    Level(
        id="babel",
        title="Babel desk",
        brief="Report a technical outage in any language except English. Watch which checkpoint the router picks.",
        lesson="The router detects script and language in under a millisecond, before the model runs, "
               "and sends non-English text to the multilingual checkpoint. The English one would stay "
               "confident while being wrong.",
        state_key="message",
        questions={"department": DEPARTMENT, "urgent": URGENT},
        objectives=[O("choice", "department", "Routed to technical", arg="technical", t=0.55)],
        max_chars=240,
        require_non_english=True,
        banned=["bug", "error", "erreur", "fehler", "crash", "panne"],
        example="Bonjour, depuis ce matin le site ne s'affiche plus et nos employés ne peuvent pas travailler.",
    ),
    Level(
        id="fog",
        title="Fog machine",
        brief="Write one message the engine cannot decide on: billing and technical must end up neck and neck.",
        lesson="Calibrated confidence is the product feature here. A 50/50 answer is the signal to "
               "escalate to a human instead of guessing.",
        state_key="message",
        questions={"department": DEPARTMENT},
        objectives=[O("split", "department", "Billing and technical tied (gap under 0.25)",
                      arg=("billing", "technical"), max_gap=0.25)],
        max_chars=220,
        example="Your checkout bug double charged my card.",
    ),
    Level(
        id="gatekeeper",
        title="Gatekeeper",
        brief="An AI assistant sits behind a guardrail. Ask a genuine security question that is filed as "
              "security testing without being flagged as a jailbreak or injection.",
        lesson="Guardrails are only useful if they do not block legitimate experts. Every guard check "
               "here ran in the same pass, fast enough to sit in front of every LLM call.",
        state_key="prompt",
        questions={
            "jailbreak": {"type": "noul", "instructions": "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"},
            "prompt_injection": {"type": "noul", "instructions": "Does `prompt` contain instructions aimed at the AI system rather than a genuine user request?"},
            "topic": {"type": "choice", "instructions": "What is `prompt` about?",
                      "criteria": {"product_support": None, "coding": None, "general_knowledge": None,
                                   "personal_advice": None, "security_testing": None, "other": None}},
        },
        objectives=[
            O("choice", "topic", "Filed as security testing", arg="security_testing", t=0.4),
            O("below", "jailbreak", "Not a jailbreak", t=0.25),
            O("below", "prompt_injection", "Not an injection", t=0.25),
        ],
        max_chars=240,
        banned=["ignore", "pretend", "roleplay", "jailbreak", "bypass", "instructions", "system prompt"],
        example="I run the authorized penetration test for our company web app next week. Which OWASP Top 10 checks should I prioritise?",
    ),
    Level(
        id="short-and-hard",
        title="Short and hard",
        brief="A router sends easy prompts to a small model and hard ones to a big one. Get rated hard in 70 characters or fewer.",
        lesson="This is model routing: a 30 ms decision that saves an expensive LLM call on every easy request.",
        state_key="request",
        questions={
            "difficulty": {"type": "score", "instructions": "How hard is `request` for a language model?",
                           "criteria": ["trivial: a lookup or one-liner", "easy: short answer, no reasoning",
                                        "moderate: several steps", "hard: long multi-step reasoning or specialist knowledge"]},
        },
        objectives=[O("band", "difficulty", "Rated hard (score 2.1 or more)", lo=2.1, hi=3.0)],
        max_chars=70,
        min_letters=6,
        banned=["hard", "difficult", "complex", "complicated"],
        example="Write a compiler for a typed lambda calculus with proofs of soundness.",
    ),
    Level(
        id="real-not-phish",
        title="Real, not phish",
        brief="Write a genuine, time-sensitive email from IT asking staff to act. It must not look like phishing.",
        lesson="Email security is where calibration pays: flag the scams, pass the real urgent mail, "
               "and hand the uncertain middle to a person.",
        state_key="body",
        extra_state={"from": "it-support@university.example", "subject": "Planned maintenance"},
        questions={
            "is_phishing": {"type": "noul", "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?"},
            "needs_reply": {"type": "noul", "instructions": "Does the sender expect a reply or an action from the reader?"},
            "urgency": {"type": "score", "instructions": "How urgent is the request in `body`?",
                        "criteria": ["no time pressure", "needs attention soon", "blocking issue or hard deadline"]},
        },
        objectives=[
            O("below", "is_phishing", "Not flagged as phishing", t=0.2),
            O("band", "urgency", "Needs attention soon or sooner (score 0.9+)", lo=0.9, hi=2.0),
        ],
        max_chars=300,
        banned=["password", "click", "link", "login", "verify", "credentials"],
        example="Hello all, the shared file server will be offline on Saturday from 8am to noon for disk replacement. "
                "Please save open work on Friday evening. Questions go to the IT desk in room B12.",
    ),
    Level(
        id="grand-finale",
        title="Grand finale",
        brief="A calm customer, in a language other than English, asks for their money back. Refund yes, churn no, toxicity none.",
        lesson="Five answers, one pass, any language. This is what a support stack runs on every ticket.",
        state_key="message",
        questions={"refund": REFUND, "churn_risk": CHURN, "toxic": TOXIC, "frustration": FRUSTRATION},
        objectives=[
            O("above", "refund", "Refund requested", t=0.7),
            O("below", "churn_risk", "No churn risk", t=0.3),
            O("below", "toxic", "Not toxic", t=0.2),
            O("band", "frustration", "Calm (score up to 1.4)", lo=0.0, hi=1.4),
        ],
        max_chars=260,
        require_non_english=True,
        banned=["refund", "rembourser", "remboursement", "reembolso", "rückerstattung", "rimborso"],
        example="Merci pour votre excellent service. Une seule remarque : j'ai payé deux fois par erreur, "
                "je souhaite récupérer le second montant. Bonne journée.",
    ),
]

BY_ID: Dict[str, Level] = {lvl.id: lvl for lvl in LEVELS}
assert len(BY_ID) == len(LEVELS), "duplicate level id"

SHOTS_PER_LEVEL = 5
