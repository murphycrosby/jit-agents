"""End-to-end run against one model. Use it to check that a provider works.

    uv run python examples/quickstart.py
    JIT_PROVIDER=claude uv run python examples/quickstart.py
    JIT_PROVIDER=databricks uv run python examples/quickstart.py

See examples/providers.py for the environment variables each provider needs.
"""
from __future__ import annotations

import asyncio
import os
import statistics
from typing import Literal

from pydantic import BaseModel, Field

from jit_agents import BuildError, ModelError, Runtime, SqliteStore
from providers import from_env


class ShipmentEvent(BaseModel):
    customer_id: str
    order_id: str
    warehouse: Literal["Dallas", "Austin", "Denver"]
    hours_late: float = Field(0, ge=0, description="hours past the promised time")


provider, model = from_env()

rt = Runtime(
    model=model,
    store=SqliteStore(os.environ.get("JIT_DB", "quickstart.db")),
    key=lambda event: event.customer_id,
)
rt.register_event(ShipmentEvent)


@rt.tool
def notify_sms(to: str, message: str) -> None:
    """Send an SMS to a person."""
    print(f"   SMS to {to}: {message}")


@rt.invariant
def at_most_one_sms_per_event(event, calls):
    return sum(c.tool == "notify_sms" for c in calls) <= 1


async def main() -> None:
    intent = (
        "If a shipment from our Dallas warehouse is more than two hours late, send an SMS. "
        'The recipient is the contact name "Murphy", no phone number needed. '
        "The message should say which order is late and by how many hours."
    )
    print(f"provider: {provider}\nmodel: {model.model}\nintent: {intent}\n")

    try:
        proposal = await rt.propose("acme", intent)
    except ModelError as exc:
        print(f"FAILED at propose: {exc}")
        return
    if not proposal.ready:
        print(f"the agent asked: {proposal.question}")
        return
    print(f"summary: {proposal.summary}")
    for t in proposal.tests:
        print(f" - {t.name}: {[c.tool for c in t.expected_calls] or 'nothing'}")

    try:
        rule = await rt.build(proposal)
    except ModelError as exc:
        print(f"FAILED at build: {exc}")
        return
    except BuildError as exc:
        print(f"\nbuild failed after {len(exc.attempts)} attempts:\n{exc.report.feedback()}\n\n{exc.source}")
        return
    rt.inject(rule)
    print(f"\ninjected v{rule.version} (id {rule.id}) after {rule.attempts} attempt(s), "
          f"{rule.report.events_checked} events checked\n\n{rule.source}")

    print("dispatching:")
    for hours in (0, 2, 2.5, 6):
        event = ShipmentEvent(customer_id="acme", order_id=f"ORD-{int(hours * 10)}", warehouse="Dallas", hours_late=hours)
        result = rt.dispatch(event)
        print(f" {hours}h late -> {len(result.calls)} call(s) in {result.latency_ns / 1000:.1f}µs")

    event = ShipmentEvent(customer_id="acme", order_id="ORD-1", warehouse="Austin", hours_late=1)
    samples = sorted(rt.dispatch(event).latency_ns for _ in range(20000))
    print(f"\n20000 dispatches: p50 {statistics.median(samples) / 1000:.2f}µs, p99 {samples[19800] / 1000:.2f}µs")


if __name__ == "__main__":
    asyncio.run(main())
