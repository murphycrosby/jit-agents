"""Example: a small site built on the jit_agents library.

Run from the repository root:

    uv run --extra example uvicorn app:app --app-dir examples/webapp --reload

Chat with the agent as a customer, confirm the proposed tests, and watch the
generated rule run in a simulated event stream.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from jit_agents import BuildError, Proposal, Runtime, SqliteStore
from jit_agents.generator import decision_schema
from jit_agents.models import ExpectedCall
from jit_agents.tools import Call, calls_match

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from providers import available, default_name  # noqa: E402

# Model providers, configured by environment variables (see examples/providers.py).
# Switch at runtime from the page header, or set JIT_PROVIDER.
PROVIDERS = available()
state = {"provider": default_name(PROVIDERS)}


def page(**context: Any) -> dict[str, Any]:
    """Template context shared by every page."""
    return context | {
        "provider": state["provider"],
        "providers": {name: m.model for name, m in PROVIDERS.items()},
    }

TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

# ---------------------------------------------------------------------------
# The host application's own models, tools and policy
# ---------------------------------------------------------------------------

Warehouse = Literal["Dallas", "Austin", "Denver", "Chicago"]


class ShipmentEvent(BaseModel):
    """A shipment status update."""
    customer_id: str
    order_id: str = Field(description="e.g. ORD-48213")
    warehouse: Warehouse
    carrier: Literal["UPS", "FedEx", "DHL"]
    hours_late: float = Field(0, ge=0, description="hours past the promised time; 0 when on time")


class InvoiceEvent(BaseModel):
    """An invoice received from a vendor."""
    customer_id: str
    invoice_id: str = Field(description="e.g. INV-2091")
    vendor: str
    trusted_vendor: bool = Field(description="whether the vendor is on the trusted list")
    amount: float = Field(ge=0, description="USD")


class TicketEvent(BaseModel):
    """A customer support ticket."""
    customer_id: str
    ticket_id: str = Field(description="e.g. TCK-771")
    sentiment: Literal["positive", "neutral", "negative"]
    region: Literal["US-South", "US-West", "EU"]
    is_vip: bool = False


effects: deque[dict[str, Any]] = deque(maxlen=200)


def rule_key(customer_id: str, event_type: str) -> str:
    return f"{customer_id}/{event_type}"


def active_rules(customer_id: str) -> dict[str, Any]:
    """The live rule for each event type, for one customer."""
    found = {name: rt.rule_for(rule_key(customer_id, name)) for name in rt.events}
    return {name: rule for name, rule in found.items() if rule is not None}


def all_versions(customer_id: str) -> list[Any]:
    rules = [r for name in rt.events for r in rt.history(rule_key(customer_id, name))]
    return sorted(rules, key=lambda r: r.created_at, reverse=True)

rt = Runtime(
    model=PROVIDERS[state["provider"]],
    store=SqliteStore(os.environ.get("JIT_DB", HERE / "rules.db")),
    # One rule per customer and event type. Keys are opaque to the library.
    key=lambda event: rule_key(event.customer_id, type(event).__name__),
)
for model in (ShipmentEvent, InvoiceEvent, TicketEvent):
    rt.register_event(model)


def _effect(tool: str, **args: Any) -> None:
    effects.append({"tool": tool, "args": args, "ts": time.time()})


@rt.tool
def notify_sms(to: str, message: str) -> None:
    """Send an SMS. `to` is a contact name or a phone number."""
    _effect("notify_sms", to=to, message=message)


@rt.tool
def notify_email(to: str, subject: str, body: str) -> None:
    """Send an email."""
    _effect("notify_email", to=to, subject=subject, body=body)


@rt.tool
def route(queue: Literal["fraud_review", "priority", "standard", "escalation"]) -> None:
    """Route the event to a work queue."""
    _effect("route", queue=queue)


@rt.tool
def set_priority(level: Literal["low", "normal", "high", "urgent"]) -> None:
    """Set the priority of the event."""
    _effect("set_priority", level=level)


@rt.tool
def flag(reason: str) -> None:
    """Flag the event for human attention."""
    _effect("flag", reason=reason)


# -- lookups: read-only, return a value, run in the hot path -----------------
# Invoices are written to a database table as they arrive. The lookup queries it.

events_db = sqlite3.connect(os.environ.get("JIT_EVENTS_DB", HERE / "events.db"), check_same_thread=False)
events_db.execute("PRAGMA journal_mode=WAL")
events_db.execute("PRAGMA synchronous=NORMAL")
events_db.execute(
    "CREATE TABLE IF NOT EXISTS invoices ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, customer_id TEXT NOT NULL, vendor TEXT NOT NULL, "
    "amount REAL NOT NULL, received_at REAL NOT NULL)"
)
events_db.execute("CREATE INDEX IF NOT EXISTS invoices_recent ON invoices (customer_id, vendor, id)")
events_db.commit()


def record(event: BaseModel) -> None:
    """Store an invoice after its rule has run, so a rule only ever sees earlier invoices."""
    if not isinstance(event, InvoiceEvent):
        return
    cur = events_db.execute(
        "INSERT INTO invoices (customer_id, vendor, amount, received_at) VALUES (?,?,?,?)",
        (event.customer_id, event.vendor, event.amount, time.time()),
    )
    if cur.lastrowid % 500 == 0:  # keep the demo table small
        events_db.execute("DELETE FROM invoices WHERE id < ?", (cur.lastrowid - 5000,))
    events_db.commit()


@rt.lookup
def average_invoice_amount(customer_id: str, vendor: str | None = None, last: int = 10) -> float | None:
    """Average amount of this customer's most recent invoices received before the current one.
    Pass vendor to include only that vendor's invoices, or omit it for all vendors.
    Returns None when there are no earlier invoices."""
    last = max(1, min(int(last), 1000))
    if vendor is None:
        query = "SELECT AVG(amount) FROM (SELECT amount FROM invoices WHERE customer_id=? ORDER BY id DESC LIMIT ?)"
        row = events_db.execute(query, (customer_id, last)).fetchone()
    else:
        query = ("SELECT AVG(amount) FROM (SELECT amount FROM invoices "
                 "WHERE customer_id=? AND vendor=? ORDER BY id DESC LIMIT ?)")
        row = events_db.execute(query, (customer_id, vendor, last)).fetchone()
    return row[0]


@rt.invariant
def at_most_one_sms_per_event(event, calls):
    return sum(c.tool == "notify_sms" for c in calls) <= 1


@rt.invariant
def trusted_vendors_skip_fraud_review(event, calls):
    if getattr(event, "trusted_vendor", False):
        return not any(c.tool == "route" and c.args.get("queue") == "fraud_review" for c in calls)
    return True


# ---------------------------------------------------------------------------
# Demo state (in memory) and the simulated stream
# ---------------------------------------------------------------------------

customers: dict[str, str] = {"acme": "Acme Logistics", "globex": "Globex Retail"}
contacts: dict[str, str] = {}   # who "me" is for each customer; defaults to CONTACT
CONTACT = os.environ.get("JIT_CONTACT", "Murphy")
messages: dict[str, list[dict[str, str]]] = {}
proposals: dict[str, Proposal] = {}
failures: dict[str, dict[str, Any]] = {}
benchmarks: dict[str, dict[str, Any]] = {}
log: deque[dict[str, Any]] = deque(maxlen=1000)
stream_state = {"running": False, "total": 0}
VENDORS = [("Acme Supply", True), ("Globex", False), ("Initech", True), ("Umbrella Corp", False)]


def random_event(customer_id: str) -> BaseModel:
    kind = random.choice(["shipment", "invoice", "ticket"])
    if kind == "shipment":
        return ShipmentEvent(
            customer_id=customer_id, order_id=f"ORD-{random.randint(10000, 99999)}",
            warehouse=random.choice(["Dallas", "Austin", "Denver", "Chicago"]),
            carrier=random.choice(["UPS", "FedEx", "DHL"]),
            hours_late=random.choice([0, 0, 0.5, 1.5, 2, 2.5, 4, 8]),
        )
    if kind == "invoice":
        vendor, trusted = random.choice(VENDORS)
        return InvoiceEvent(
            customer_id=customer_id, invoice_id=f"INV-{random.randint(1000, 9999)}", vendor=vendor,
            trusted_vendor=trusted, amount=round(random.choice([45, 120, 480, 500, 520, 700, 9800]) * random.uniform(0.95, 1.05), 2),
        )
    return TicketEvent(
        customer_id=customer_id, ticket_id=f"TCK-{random.randint(100, 999)}",
        sentiment=random.choice(["positive", "neutral", "negative"]),
        region=random.choice(["US-South", "US-West", "EU"]), is_vip=random.random() < 0.2,
    )


def push(event: BaseModel) -> dict[str, Any]:
    result = rt.dispatch(event)
    entry = result.to_dict() | {
        "customer": event.customer_id, "event_type": type(event).__name__,
        "event": event.model_dump(), "ts": time.time(),
    }
    log.append(entry)
    stream_state["total"] += 1
    record(event)
    return entry


async def run_stream() -> None:
    while stream_state["running"]:
        for cid in list(customers):
            push(random_event(cid))
        await asyncio.sleep(0.25)


def start_stream() -> None:
    if not stream_state["running"]:
        stream_state["running"] = True
        asyncio.create_task(run_stream())


def stats(customer_id: str | None = None) -> dict[str, Any]:
    ran = [e for e in log if e["ran"] and (customer_id is None or e["customer"] == customer_id)]
    lat = sorted(e["latency_ns"] for e in ran)
    out: dict[str, Any] = {
        "running": stream_state["running"], "total": stream_state["total"], "dispatched": len(ran),
        "errors": sum(1 for e in ran if e["error"]), "p50_us": None, "p99_us": None, "max_us": None,
    }
    if lat:
        out["p50_us"] = round(statistics.median(lat) / 1000, 2)
        out["p99_us"] = round(lat[max(0, int(len(lat) * 0.99) - 1)] / 1000, 2)
        out["max_us"] = round(lat[-1] / 1000, 2)
    return out


def say(cid: str, role: str, content: str) -> None:
    messages.setdefault(cid, []).append({"role": role, "content": content})


@asynccontextmanager
async def lifespan(_: FastAPI):
    for key in rt.store.keys():
        if "/" not in key:  # rules saved before keys included the event type
            for rule in rt.store.list(key):
                rule.key = rule_key(key, rule.event_type)
                rt.store.save(rule)
        customers.setdefault(key.split("/")[0], key.split("/")[0])
    start_stream()   # rules are lazy loaded by the first event for each customer
    yield
    stream_state["running"] = False


app = FastAPI(title="JIT Agents example", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    rows = []
    for cid, name in customers.items():
        loaded = sum(1 for k in rt.loaded_keys() if k.startswith(cid + "/"))
        rows.append({"id": cid, "name": name, "rules": active_rules(cid), "stats": stats(cid), "loaded": loaded})
    return TEMPLATES.TemplateResponse(request, "index.html", page(customers=rows))


@app.post("/customers")
async def create_customer(name: str = Form(...)):
    cid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "customer"
    customers.setdefault(cid, name.strip())
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.get("/customers/{cid}", response_class=HTMLResponse)
async def customer_page(request: Request, cid: str):
    if cid not in customers:
        raise HTTPException(404)
    return TEMPLATES.TemplateResponse(
        request, "customer.html",
        page(
            cid=cid, name=customers[cid],
            messages=messages.get(cid, []), proposal=proposals.get(cid),
            active=active_rules(cid), rules=all_versions(cid), event_types=list(rt.events),
            old_counts={name: sum(1 for r in rt.history(rule_key(cid, name)) if r.status != "active") for name in rt.events},
            failure=failures.get(cid), benchmark=benchmarks.get(cid),
        ),
    )


def sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"


def live(generator) -> StreamingResponse:
    return StreamingResponse(
        generator, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def build_stream(cid: str, proposal: Proposal, auto: bool):
    """Build and inject, yielding each step as it happens.

    If no rule can satisfy the tests, the agent revises the tests once. With
    ``auto`` it then builds again. Otherwise the revised tests wait for the customer.
    """
    failures.pop(cid, None)
    for round_number in (1, 2):
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def progress(kind: str, data: dict[str, Any]) -> None:
            if kind == "generating":
                queue.put_nowait({"stage": "generating", "attempt": data["attempt"], "max": rt.max_attempts})
            else:
                report = data["report"]
                queue.put_nowait({
                    "stage": "checked", "attempt": data["attempt"], "ok": report.ok, "source": data["source"],
                    "events_checked": report.events_checked, "feedback": None if report.ok else report.feedback(),
                })

        task = asyncio.create_task(rt.build(proposal, on_progress=progress))
        while True:
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield sse(getter.result())
                continue
            getter.cancel()
            while not queue.empty():
                yield sse(queue.get_nowait())
            break

        try:
            rule = task.result()
        except BuildError as exc:
            if round_number == 1:
                yield sse({"stage": "revising", "attempts": len(exc.attempts)})
                try:
                    revised = await rt.revise(proposal, exc)
                except Exception as err:
                    say(cid, "system", f"Agent error: {err}")
                    yield sse({"stage": "error", "message": str(err)})
                    return
                if revised.ready:
                    revised.key = rule_key(cid, revised.event_type)
                    proposals[cid] = proposal = revised
                    say(cid, "assistant", f"No rule could pass my first tests, so I corrected them. "
                                          f"{proposal.event_type}: {proposal.summary}")
                    yield sse(proposal_event(proposal) | {"revised": True})
                    if auto:
                        continue
                    yield sse({"stage": "awaiting_confirmation"})
                    return
            failures[cid] = {"source": exc.source, "report": exc.report.model_dump() if exc.report else None, "attempts": len(exc.attempts)}
            say(cid, "assistant", "I could not produce a rule that passes the gate. Try restating the rule more precisely.")
            yield sse({"stage": "failed", "attempts": len(exc.attempts)})
            return
        except Exception as exc:
            say(cid, "system", f"Agent error: {exc}")
            yield sse({"stage": "error", "message": str(exc)})
            return

        rule = rt.inject(rule)
        proposals.pop(cid, None)
        r = rule.report
        say(cid, "assistant",
            f"{rule.event_type} rule v{rule.version} is live. {len(r.behavior)} behaviour tests, {len(r.builtin)} "
            f"built-in checks and {len(r.invariants)} invariants pass across {r.events_checked} events.")
        yield sse({
            "stage": "injected", "version": rule.version, "event_type": rule.event_type, "rule_id": rule.id,
            "attempts": rule.attempts, "source": rule.source, "p50_us": round(r.latency_p50_ns / 1000, 2),
            "events_checked": r.events_checked,
        })
        return


def proposal_event(proposal: Proposal) -> dict[str, Any]:
    replacing = rt.rule_for(proposal.key)
    return {
        "stage": "proposal", "summary": proposal.summary, "event_type": proposal.event_type,
        "replaces": replacing.version if replacing else None,
        "tests": [t.model_dump() for t in proposal.tests],
    }


@app.post("/customers/{cid}/update")
async def update(cid: str, text: str = Form(...), auto: bool = Form(False)):
    """Take an instruction from the text box and stream every step back to the page."""
    if cid not in customers:
        raise HTTPException(404)
    text = text.strip()

    async def steps():
        if not text:
            return
        history = [m for m in messages.get(cid, []) if m["role"] in ("user", "assistant")][-12:]
        current = active_rules(cid)
        contact = contacts.get(cid, CONTACT)
        context = (
            f"CONTEXT: I am {contact} at {customers[cid]}, customer_id {cid!r}. When I say me or my, use the "
            f"contact name {contact!r} as the recipient. Contact names are valid recipients for SMS and email.\n"
            "I can have one rule per event type. "
        )
        if current:
            listing = "\n".join(f"- {name} (v{r.version}): {r.summary}" for name, r in current.items())
            context += (
                "My live rules are:\n" + listing + "\n"
                "A request about an event type with no rule adds a new rule and leaves the others alone. "
                "A change to an existing rule must give the complete updated rule for that event type."
            )
        else:
            context += "I have no rules yet."
        history.insert(0, {"role": "user", "content": context})
        say(cid, "user", text)
        yield sse({"stage": "understanding"})
        try:
            proposal = await rt.propose(cid, text, history=history)
        except Exception as exc:
            say(cid, "system", f"Agent error: {exc}")
            yield sse({"stage": "error", "message": str(exc)})
            return
        if not proposal.ready:
            proposals.pop(cid, None)
            say(cid, "assistant", proposal.question or "Could you say more?")
            yield sse({"stage": "question", "text": proposal.question})
            return
        proposal.key = rule_key(cid, proposal.event_type)
        proposals[cid] = proposal
        say(cid, "assistant", f"{proposal.event_type}: {proposal.summary}")
        yield sse(proposal_event(proposal))
        if not auto:
            yield sse({"stage": "awaiting_confirmation"})
            return
        async for chunk in build_stream(cid, proposal, auto=True):
            yield chunk

    return live(steps())


@app.post("/customers/{cid}/build")
async def build(cid: str):
    """Build the pending proposal after the customer confirms it."""
    proposal = proposals.get(cid)
    if proposal is None:
        raise HTTPException(404)
    return live(build_stream(cid, proposal, auto=False))


@app.post("/customers/{cid}/reject")
async def reject(cid: str):
    proposals.pop(cid, None)
    say(cid, "assistant", "Discarded. Tell me what to change.")
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.post("/customers/{cid}/rollback")
async def rollback(cid: str, event_type: str = Form(...)):
    prev = rt.rollback(rule_key(cid, event_type))
    say(cid, "system", f"{event_type} rolled back to v{prev.version}." if prev else f"No earlier {event_type} rule to roll back to.")
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.post("/customers/{cid}/rules/{rule_id}/delete")
async def delete_version(cid: str, rule_id: str):
    rule = rt.store.get(rule_id)
    if rule is None or not rule.key.startswith(cid + "/"):
        raise HTTPException(404)
    try:
        rt.delete(rule_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    say(cid, "system", f"{rule.event_type} v{rule.version} deleted.")
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.post("/customers/{cid}/prune")
async def prune(cid: str, event_type: str = Form(...)):
    count = rt.prune(rule_key(cid, event_type))
    say(cid, "system", f"Deleted {count} old {event_type} version(s).")
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.post("/customers/{cid}/remove")
async def remove(cid: str, event_type: str = Form(...)):
    rt.remove(rule_key(cid, event_type))
    say(cid, "system", f"{event_type} rule removed. Its history is kept.")
    return RedirectResponse(f"/customers/{cid}", status_code=303)


@app.post("/customers/{cid}/benchmark")
async def benchmark(cid: str, event_type: str = Form(...), n: int = Form(5)):
    """The same events through the injected rule and through a model called per event."""
    rule = rt.rule_for(rule_key(cid, event_type))
    if rule is None:
        raise HTTPException(400, "no active rule")
    event_model = rt.events[rule.event_type]
    system = (
        "You are a rule engine. Apply the customer's rule to the event and return the tool calls to make.\n"
        f"AVAILABLE TOOLS:\n{rt.toolset.schema_text()}\n"
        'Respond with JSON {"calls": [{"tool": str, "args": {...}}]}. Return an empty list if the rule does not apply.'
    )
    rows = []
    for _ in range(max(1, min(n, 20))):
        event = random_event(cid)
        while not isinstance(event, event_model):
            event = random_event(cid)
        injected = rt.dispatch(event, execute=False)
        facts = ""
        if isinstance(event, InvoiceEvent):
            facts = (
                "\n\nLOOKUP VALUES:\n"
                f"average of this vendor's last 10 invoices: {average_invoice_amount(cid, event.vendor)}\n"
                f"average of the last 10 invoices from all vendors: {average_invoice_amount(cid)}"
            )
        t0 = time.perf_counter()
        try:
            data = await rt.model.json(system, [{"role": "user", "content": f"RULE:\n{rule.summary}\n\nEVENT:\n{event.model_dump_json()}{facts}"}], decision_schema(rt.toolset))
            llm_calls, err = data.get("calls") or [], None
        except Exception as exc:
            llm_calls, err = [], str(exc)
        secs = time.perf_counter() - t0
        agree = err is None and calls_match(
            [ExpectedCall(tool=c.tool, args=c.args) for c in injected.calls],
            [Call(c.get("tool", ""), c.get("args") or {}) for c in llm_calls],
        ) is None
        rows.append({
            "event": event.model_dump(exclude={"customer_id"}), "rule_calls": [c.to_dict() for c in injected.calls],
            "rule_us": round(injected.latency_ns / 1000, 2), "llm_calls": llm_calls,
            "llm_ms": round(secs * 1000, 1), "llm_error": err, "agree": agree,
        })
    rule_us = [r["rule_us"] for r in rows]
    llm_ms = [r["llm_ms"] for r in rows if not r["llm_error"]]
    avg_us = sum(rule_us) / len(rule_us)
    avg_ms = sum(llm_ms) / len(llm_ms) if llm_ms else None
    benchmarks[cid] = {
        "model": f"{state['provider']} · {rt.model.model}", "event_type": event_type,
        "rows": rows, "rule_avg_us": round(avg_us, 2), "llm_avg_ms": round(avg_ms, 1) if avg_ms else None,
        "speedup": round(avg_ms * 1000 / avg_us) if avg_ms and avg_us else None,
        "agreement": f"{sum(1 for r in rows if r['agree'])}/{len(rows)}",
    }
    return RedirectResponse(f"/customers/{cid}#benchmark", status_code=303)


# ---------------------------------------------------------------------------
# JSON for the live view
# ---------------------------------------------------------------------------


@app.post("/api/model")
async def set_model(provider: str = Form(...)):
    """Switch the model used for proposing and building. Live rules are unaffected."""
    if provider not in PROVIDERS:
        raise HTTPException(400, f"unknown provider {provider!r}")
    state["provider"] = provider
    rt.model = PROVIDERS[provider]
    return JSONResponse({"provider": provider, "model": rt.model.model})


@app.get("/api/stream")
async def api_stream(customer_id: str | None = None, limit: int = 25):
    items = [e for e in reversed(log) if customer_id is None or e["customer"] == customer_id][:limit]
    return JSONResponse(json.loads(json.dumps({"stats": stats(customer_id), "events": items}, default=str)))


@app.post("/api/stream/{action}")
async def api_stream_control(action: str):
    if action == "start":
        start_stream()
    elif action == "stop":
        stream_state["running"] = False
    else:
        raise HTTPException(404)
    return JSONResponse(stats())
