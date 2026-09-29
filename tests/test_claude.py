"""The Claude provider, exercised with a stand-in client. No network calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from jit_agents import Claude, ModelError
from jit_agents.generator import CODE_SCHEMA, decision_schema, proposal_schema

from .conftest import GOOD, TESTS, make_runtime

anthropic = pytest.importorskip("anthropic")


class FakeMessages:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.requests = reply, stop_reason, []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        schema = kwargs["output_config"]["format"]["schema"]
        body = self.reply(schema) if callable(self.reply) else self.reply
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps(body))],
        )


def fake_client(reply, stop_reason="end_turn"):
    ga, beta = FakeMessages(reply, stop_reason), FakeMessages(reply, stop_reason)
    return SimpleNamespace(messages=ga, beta=SimpleNamespace(messages=beta)), ga, beta


def objects(schema):
    """Every object schema nested anywhere inside ``schema``."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from objects(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from objects(item)


def test_request_shape_with_fallbacks():
    client, ga, beta = fake_client({"source": "x"})
    model = Claude(client=client, effort="high")
    out = asyncio.run(model.json("be precise", [{"role": "user", "content": "hi"}], CODE_SCHEMA))

    assert out == {"source": "x"} and not ga.requests
    req = beta.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["system"] == "be precise"
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["betas"] == ["server-side-fallback-2026-07-01"] and req["fallbacks"] == "default"
    assert req["output_config"]["effort"] == "high"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in req and "temperature" not in req


def test_fallbacks_can_be_turned_off():
    client, ga, beta = fake_client({"source": "x"})
    asyncio.run(Claude(client=client, fallbacks=False).json("s", [{"role": "user", "content": "hi"}], CODE_SCHEMA))
    assert len(ga.requests) == 1 and not beta.requests
    assert "betas" not in ga.requests[0] and "fallbacks" not in ga.requests[0]


@pytest.mark.parametrize("stop_reason, match", [("refusal", "declined"), ("max_tokens", "output tokens")])
def test_bad_stop_reasons_raise(stop_reason, match):
    client, _, _ = fake_client({"source": "x"}, stop_reason=stop_reason)
    with pytest.raises(ModelError, match=match):
        asyncio.run(Claude(client=client).json("s", [{"role": "user", "content": "hi"}], CODE_SCHEMA))


def test_schemas_meet_structured_output_rules(rt):
    """Every object is closed and nothing is free-form, before and after the SDK transform."""
    for schema in (proposal_schema(rt.events, rt.toolset), decision_schema(rt.toolset), CODE_SCHEMA):
        sent = anthropic.transform_schema(schema)
        found = list(objects(sent))
        assert found
        for obj in found:
            assert obj.get("additionalProperties") is False, obj
            assert "properties" in obj, obj
        assert "minimum" not in json.dumps(sent)


def test_proposal_schema_names_every_event_and_tool(rt):
    text = json.dumps(proposal_schema(rt.events, rt.toolset))
    for name in ("ShipmentEvent", "InvoiceEvent", "notify_sms", "route", "fraud_review", "Vendor"):
        assert name in text


def test_full_flow_through_the_claude_provider():
    def reply(schema):
        if "tests" in schema["properties"]:
            return {
                "ready": True, "question": None, "event_type": "ShipmentEvent",
                "summary": "Text Murphy when a Dallas shipment is more than 2 hours late.",
                "tests": [t.model_dump() for t in TESTS],
            }
        return {"source": GOOD}

    client, _, beta = fake_client(reply)
    rt, sent = make_runtime(model=Claude(client=client))

    async def flow():
        return await rt.build(await rt.propose("acme", "Text Murphy if a Dallas shipment is over 2 hours late"))

    rule = rt.inject(asyncio.run(flow()))
    assert rule.attempts == 1 and len(beta.requests) == 2
    from .conftest import ShipmentEvent

    res = rt.dispatch(ShipmentEvent(customer_id="acme", order_id="ORD-9", warehouse="Dallas", hours_late=4))
    assert res.executed and len(sent) == 1
