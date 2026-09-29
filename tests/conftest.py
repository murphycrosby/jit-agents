from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel

from jit_agents import Case, Runtime


class Vendor(BaseModel):
    name: str
    trusted: bool = False


class ShipmentEvent(BaseModel):
    customer_id: str
    order_id: str
    warehouse: Literal["Dallas", "Austin", "Denver"]
    hours_late: float | None = 0
    vendor: Vendor | None = None


class InvoiceEvent(BaseModel):
    customer_id: str
    amount: float
    trusted_vendor: bool = False


GOOD = '''
def handle(event, tools):
    hours = event.hours_late if event.hours_late is not None else 0
    if event.warehouse == "Dallas" and hours > 2:
        tools.notify_sms(to="Murphy", message=f"Shipment {event.order_id} from Dallas is {hours}h late")
'''

GOOD_V2 = GOOD.replace('to="Murphy"', 'to="Ops"')

TESTS = [
    Case(
        name="late Dallas shipment",
        event={"customer_id": "acme", "order_id": "ORD-1", "warehouse": "Dallas", "hours_late": 3.0},
        expected_calls=[{"tool": "notify_sms", "args": {"to": "Murphy", "message": "Shipment ORD-1 from Dallas is 3.0h late"}}],
    ),
    Case(
        name="exactly two hours is not late enough",
        event={"customer_id": "acme", "order_id": "ORD-2", "warehouse": "Dallas", "hours_late": 2.0},
        expected_calls=[],
    ),
    Case(
        name="late Austin shipment is ignored",
        event={"customer_id": "acme", "order_id": "ORD-3", "warehouse": "Austin", "hours_late": 5},
        expected_calls=[],
    ),
]

TESTS_V2 = [
    t.model_copy(update={"expected_calls": [c.model_copy(update={"args": {**c.args, "to": "Ops"}}) for c in t.expected_calls]})
    for t in TESTS
]


def make_runtime(**kwargs) -> tuple[Runtime, list]:
    sent: list = []
    rt = Runtime(key=lambda e: e["customer_id"] if isinstance(e, dict) else e.customer_id, **kwargs)
    rt.register_event(ShipmentEvent)
    rt.register_event(InvoiceEvent)

    @rt.tool
    def notify_sms(to: str, message: str) -> None:
        """Send an SMS."""
        sent.append(("sms", to, message))

    @rt.tool
    def route(queue: Literal["fraud_review", "standard"]) -> None:
        """Route the event to a queue."""
        sent.append(("route", queue))

    return rt, sent


@pytest.fixture
def rt_and_sent():
    return make_runtime()


@pytest.fixture
def rt(rt_and_sent):
    return rt_and_sent[0]
