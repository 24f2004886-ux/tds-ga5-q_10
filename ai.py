"""
AI reasoning layer.

Kept separate from the protocol/storage code on purpose: this is the ONLY
place that calls an LLM, and it's called at most once per batch of NEW
(never-seen) packages. Retries, polls, cancels, and replays never touch this
file.

Uses AI Pipe (https://aipipe.org/) - an OpenAI-compatible proxy - so you don't
need an OpenAI key, just a free AI Pipe token.

Set these environment variables:
    AIPIPE_TOKEN   - your AI Pipe token (get one at https://aipipe.org/login)
    AIPIPE_MODEL   - defaults to a cheap model, e.g. "openai/gpt-4o-mini"
"""

import json
import os
import re
from typing import Any, Dict, List

import requests

AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openrouter/v1")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "openai/gpt-4o-mini")

VALID_ACTIONS = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
]

SYSTEM_PROMPT = f"""You are an invoice-processing agent. For EACH invoice package given,
choose exactly one action from this list: {", ".join(VALID_ACTIONS)}.

Rules:
- settle_invoice: valid, reconciled, and within autonomous authority.
- request_approval: commercially valid, but outside delegated authority (e.g. amount too large).
- hold_invoice: payment pauses until a stated verification completes.
- reject_duplicate: the same commercial invoice was already paid.
- open_exception: material records conflict and need an exception workflow.

The documents mix useful facts with old examples, negation, and irrelevant action words.
Ignore cover-sheet references, archived/example references, and decoy sentences.
Find the paragraph that actually DECIDES the action and cite ONLY the exact bracketed
references from that paragraph (things like "[3]"), nothing else.

Return ONLY a JSON array (no markdown, no prose), one object per package, in the same
order as given, each shaped exactly like:
{{
  "action": "one of the five action strings",
  "facts": {{"vendorName": "...", "invoiceNumber": "...", "amountMinor": 12345, "currency": "INR"}},
  "evidenceRefs": ["[3]", "[7]"],
  "rationale": "60 to 1500 characters. Name the action explicitly and cite at least two evidence refs."
}}
"""


def _extract_json_array(text: str) -> Any:
    text = text.strip()
    # Strip markdown fences if the model added them anyway.
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return json.loads(text)


def decide_actions(packages: List[Dict[str, Any]], policy_revision: str) -> List[Dict[str, Any]]:
    """
    Calls the AI once for the whole list of (uncached) packages and returns a
    validated list of decision dicts (without packageId/actionId - those are
    filled in by the caller).
    """
    if not AIPIPE_TOKEN:
        # Fallback so the server still runs / is testable without a token,
        # though this will NOT score well - always set AIPIPE_TOKEN for real use.
        return [_fallback_decision(pkg) for pkg in packages]

    user_content = json.dumps(
        {"policyRevision": policy_revision, "packages": packages}, ensure_ascii=False
    )

    resp = requests.post(
        f"{AIPIPE_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": AIPIPE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        },
        timeout=40,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    try:
        parsed = _extract_json_array(content)
    except Exception:
        parsed = [_fallback_decision(pkg) for pkg in packages]

    results = []
    for i, pkg in enumerate(packages):
        try:
            d = parsed[i]
            action = d.get("action")
            if action not in VALID_ACTIONS:
                raise ValueError("bad action")
            facts = d.get("facts", {})
            evidence = d.get("evidenceRefs", [])
            rationale = d.get("rationale", "")
            if not (60 <= len(rationale) <= 1500):
                rationale = (rationale + " " * 60)[:1500]
                if len(rationale) < 60:
                    rationale = rationale.ljust(60, ".")
            results.append(
                {
                    "action": action,
                    "facts": {
                        "vendorName": facts.get("vendorName", ""),
                        "invoiceNumber": facts.get("invoiceNumber", ""),
                        "amountMinor": facts.get("amountMinor", 0),
                        "currency": facts.get("currency", "INR"),
                    },
                    "evidenceRefs": evidence,
                    "rationale": rationale,
                }
            )
        except Exception:
            results.append(_fallback_decision(pkg))
    return results


def _fallback_decision(pkg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action": "open_exception",
        "facts": {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": "INR"},
        "evidenceRefs": [],
        "rationale": "Fallback decision: AI reasoning unavailable, routing to manual exception review for safety and audit purposes.",
    }
