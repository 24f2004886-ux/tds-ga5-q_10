"""
A2A Invoice Action Agent
========================

Implements the A2A 1.0 HTTP+JSON surface described in the assignment:
  GET  /.well-known/agent-card.json
  POST {base}/message:send
  GET  {base}/tasks/{id}
  GET  {base}/tasks
  POST {base}/tasks/{id}:cancel

Design (kept in 3 layers, as the assignment recommends):
  1. Protocol layer   -> FastAPI routes, header checks, envelopes (this file, top half)
  2. Storage layer    -> in-memory TaskStore with locks, idempotency, isolation
  3. AI reasoning      -> ai.py (separate file) - one batched call, cached by content

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Set BASE_URL env var to the exact public https URL you will submit, e.g.
    export BASE_URL="https://your-domain.com/a2a"
This value must appear inside the Agent Card's supportedInterfaces.
"""

import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from ai import decide_actions  # our AI reasoning layer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_URL = os.environ.get("BASE_URL", "https://example.invalid/a2a").rstrip("/")
A2A_VERSION = "1.0"
MEDIA_TYPE = "application/a2a+json"

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MODE = "application/vnd.ga5.invoice-action-receipts+json"

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

app = FastAPI()

# --------------------------------------------------------------------------
# In-memory storage (thread-safe)
# --------------------------------------------------------------------------
_lock = threading.RLock()

# tasks[task_id] = task dict
tasks: Dict[str, Dict[str, Any]] = {}

# per-task lock, to make the cancel-vs-result race resolve to exactly one winner
task_locks: Dict[str, threading.Lock] = {}

# message idempotency: (principal, messageId) -> {"hash": ..., "task_id": ...}
message_index: Dict[tuple, Dict[str, Any]] = {}

# principal -> set of task_ids they own
principal_tasks: Dict[str, set] = {}

# proposal cache by canonical package content -> proposal dict (without actionId)
package_decision_cache: Dict[str, Dict[str, Any]] = {}


def get_task_lock(task_id: str) -> threading.Lock:
    with _lock:
        if task_id not in task_locks:
            task_locks[task_id] = threading.Lock()
        return task_locks[task_id]


def canonical_json(obj: Any) -> str:
    """Recursively key-sorted, compact JSON string."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# --------------------------------------------------------------------------
# Agent Card (public, no auth)
# --------------------------------------------------------------------------
@app.get("/.well-known/agent-card.json")
def agent_card():
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one business action per invoice with cited evidence, and executes only accepted proposals.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "description": "Chooses one of settle_invoice, request_approval, hold_invoice, reject_duplicate, or open_exception for each invoice package, citing exact evidence, then executes only accepted proposals.",
                "tags": ["invoice", "finance", "a2a", "agent"],
            }
        ],
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [INPUT_MODE, RESULT_MODE],
        "defaultOutputModes": [PROPOSALS_MODE, RECEIPTS_MODE],
    }
    return JSONResponse(content=card, media_type="application/json")


# --------------------------------------------------------------------------
# Header / auth helpers
# --------------------------------------------------------------------------
def check_headers(a2a_version: Optional[str], content_type: Optional[str], authorization: Optional[str]):
    if authorization is None or not authorization.startswith("Bearer ") or len(authorization) <= 7:
        raise HTTPException(status_code=401, detail="Missing or malformed bearer token")

    if a2a_version is None:
        raise HTTPException(status_code=401, detail="Missing A2A-Version header")
    if a2a_version != A2A_VERSION:
        raise HTTPException(status_code=400, detail="Unsupported A2A-Version")

    # Content-Type only strictly required on requests with a body (message:send)
    if content_type is not None and MEDIA_TYPE not in content_type:
        raise HTTPException(status_code=400, detail="Unsupported media type")


def get_principal(authorization: str) -> str:
    # Every distinct bearer token is treated as a separate authenticated user.
    token = authorization[len("Bearer "):]
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")
    return token


def owns_task(principal: str, task_id: str) -> bool:
    return task_id in principal_tasks.get(principal, set())


def generic_not_found():
    # Never reveal whether a task exists for another principal.
    raise HTTPException(status_code=404, detail="Task not found")


# --------------------------------------------------------------------------
# Task helpers
# --------------------------------------------------------------------------
def make_task_snapshot(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": task["id"],
        "contextId": task["contextId"],
        "status": {"state": task["state"]},
        "history": task["history"],
        "artifacts": task["artifacts"],
    }


def add_history(task: Dict[str, Any], message: Dict[str, Any]):
    task["history"].append(message)


# --------------------------------------------------------------------------
# POST {base}/message:send
# --------------------------------------------------------------------------
@app.post("/a2a/message:send")
async def message_send(
    request: Request,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None),
    content_type: Optional[str] = Header(None, alias="Content-Type"),
):
    check_headers(a2a_version, content_type, authorization)
    principal = get_principal(authorization)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON body")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Malformed body")

    message = body.get("message")
    configuration = body.get("configuration", {})
    if (
        not isinstance(message, dict)
        or "messageId" not in message
        or "parts" not in message
        or not isinstance(message.get("parts"), list)
        or len(message["parts"]) == 0
    ):
        raise HTTPException(status_code=400, detail="Malformed message")

    message_id = message["messageId"]
    msg_hash = content_hash(message)
    dedupe_key = (principal, message_id)

    with _lock:
        existing = message_index.get(dedupe_key)
        if existing is not None:
            if existing["hash"] == msg_hash:
                # Same message replayed (maybe reordered keys / different `configuration`
                # / concurrent duplicate). Return the same stored task, no new work.
                task = tasks[existing["task_id"]]
                return {"task": make_task_snapshot(task)}
            else:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "IDEMPOTENCY_CONFLICT", "message": "messageId reused with different content"},
                )

    part = message["parts"][0]
    media_type = part.get("mediaType")
    data = part.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Malformed part data")

    try:
        if media_type == INPUT_MODE:
            return handle_new_batch(principal, message, message_id, msg_hash, data)
        elif media_type == RESULT_MODE:
            return handle_results(principal, message, message_id, msg_hash, data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported part mediaType")
    except HTTPException:
        raise
    except Exception as e:
        # Never let an unexpected bug surface as a bare 500 - treat malformed/
        # unexpected input as a client error instead of crashing the task store.
        raise HTTPException(status_code=400, detail=f"Malformed request: {e}")


def handle_new_batch(principal, message, message_id, msg_hash, data):
    batch_id = data["batchId"]
    packages = data["packages"]

    task_id = new_id("task")
    context_id = new_id("ctx")

    proposals = []
    for pkg in packages:
        pkg_key = content_hash(pkg)
        cached = package_decision_cache.get(pkg_key)
        if cached is None:
            cached = None  # filled below after batched AI call
        proposals.append((pkg, pkg_key, cached))

    # Only call the AI for packages we haven't seen before (canonical content cache).
    uncached_packages = [pkg for (pkg, key, cached) in proposals if cached is None]
    if uncached_packages:
        ai_results = decide_actions(uncached_packages, data.get("policyRevision"))
        for pkg, decision in zip(uncached_packages, ai_results):
            pkg_key = content_hash(pkg)
            package_decision_cache[pkg_key] = decision

    final_proposals = []
    seen_action_ids = set()
    for pkg, pkg_key, _ in proposals:
        decision = dict(package_decision_cache[pkg_key])  # copy
        # actionId must be unique per batch even if content is cached/reused.
        action_id = new_id("act")
        while action_id in seen_action_ids:
            action_id = new_id("act")
        seen_action_ids.add(action_id)
        decision["actionId"] = action_id
        decision["packageId"] = pkg.get("packageId") or pkg.get("id")
        if decision["action"] not in VALID_ACTIONS:
            decision["action"] = "open_exception"
        final_proposals.append(decision)

    now = time.time()
    task = {
        "id": task_id,
        "contextId": context_id,
        "principal": principal,
        "state": "TASK_STATE_INPUT_REQUIRED",
        "history": [message],
        "artifacts": [
            {
                "parts": [
                    {
                        "mediaType": PROPOSALS_MODE,
                        "data": {"batchId": batch_id, "proposals": final_proposals},
                    }
                ]
            }
        ],
        "batchId": batch_id,
        "proposals_by_key": {
            (p["packageId"], p["actionId"]): p for p in final_proposals
        },
        "created": now,
    }

    with _lock:
        tasks[task_id] = task
        principal_tasks.setdefault(principal, set()).add(task_id)
        message_index[(principal, message_id)] = {"hash": msg_hash, "task_id": task_id}

    return {"task": make_task_snapshot(task)}


def handle_results(principal, message, message_id, msg_hash, data):
    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if not task_id or task_id not in tasks:
        generic_not_found()

    lock = get_task_lock(task_id)
    with lock:
        task = tasks[task_id]

        if task["principal"] != principal:
            generic_not_found()
        if task["contextId"] != context_id:
            raise HTTPException(status_code=400, detail="Context mismatch")
        if data.get("batchId") != task["batchId"]:
            raise HTTPException(status_code=400, detail="Batch mismatch")

        if task["state"] != "TASK_STATE_INPUT_REQUIRED":
            # Already terminal (COMPLETED/CANCELED) or otherwise not awaiting results.
            raise HTTPException(status_code=409, detail="Task not awaiting results")

        executions = []
        for result in data["results"]:
            key = (result["packageId"], result["actionId"])
            proposal = task["proposals_by_key"].get(key)
            if (
                proposal is None
                or proposal["action"] != result.get("action")
            ):
                # Doesn't match a stored proposal -> ignore (not executed).
                continue
            if result.get("outcome") == "ACCEPTED":
                executions.append(
                    {
                        "packageId": proposal["packageId"],
                        "actionId": proposal["actionId"],
                        "action": proposal["action"],
                        "receiptNonce": result["receiptNonce"],
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    }
                )
            # REJECTED proposals stay in history, are not executed.

        task["artifacts"].append(
            {
                "parts": [
                    {
                        "mediaType": RECEIPTS_MODE,
                        "data": {"batchId": task["batchId"], "executions": executions},
                    }
                ]
            }
        )
        add_history(task, message)
        task["state"] = "TASK_STATE_COMPLETED"

        message_index[(principal, message_id)] = {"hash": msg_hash, "task_id": task_id}

        return {"task": make_task_snapshot(task)}


# --------------------------------------------------------------------------
# GET {base}/tasks/{id}
# --------------------------------------------------------------------------
@app.get("/a2a/tasks/{task_id}")
def get_task(
    task_id: str,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None),
):
    check_headers(a2a_version, None, authorization)
    principal = get_principal(authorization)

    task = tasks.get(task_id)
    if task is None or task["principal"] != principal:
        generic_not_found()

    return make_task_snapshot(task)


# --------------------------------------------------------------------------
# GET {base}/tasks
# --------------------------------------------------------------------------
@app.get("/a2a/tasks")
def list_tasks(
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None),
):
    check_headers(a2a_version, None, authorization)
    principal = get_principal(authorization)

    ids = principal_tasks.get(principal, set())
    return {"tasks": [make_task_snapshot(tasks[i]) for i in ids]}


# --------------------------------------------------------------------------
# POST {base}/tasks/{id}:cancel
# --------------------------------------------------------------------------
@app.post("/a2a/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    a2a_version: Optional[str] = Header(None, alias="A2A-Version"),
    authorization: Optional[str] = Header(None),
):
    check_headers(a2a_version, None, authorization)
    principal = get_principal(authorization)

    if task_id not in tasks:
        generic_not_found()

    lock = get_task_lock(task_id)
    with lock:
        task = tasks[task_id]
        if task["principal"] != principal:
            generic_not_found()

        if task["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            raise HTTPException(status_code=409, detail="Task already terminal")

        task["state"] = "TASK_STATE_CANCELED"
        return make_task_snapshot(task)
