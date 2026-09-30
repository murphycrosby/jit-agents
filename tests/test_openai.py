"""The OpenAI-format provider and its schema dialects. No network calls."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from jit_agents import ModelError, OpenAICompatible
from jit_agents.generator import CODE_SCHEMA, decision_schema, proposal_schema
from jit_agents.schema import count_keys, decode, to_simple, to_strict

from .conftest import GOOD, TESTS, ShipmentEvent, make_runtime

pytest.importorskip("openai")
FORBIDDEN_SIMPLE = ("anyOf", "oneOf", "allOf", "$ref", "$defs", "prefixItems", "pattern")


def walk(node):
    if isinstance(node, dict):
        yield node
        for k, v in node.items():
            if k == "properties":
                for sub in v.values():
                    yield from walk(sub)
            elif k not in ("enum", "const", "required", "description"):
                yield from walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def schema_keywords(schema):
    return {k for node in walk(schema) for k in node}


# -- dialects ----------------------------------------------------------------

def test_strict_requires_every_field_and_closes_objects(rt):
    sent = to_strict(proposal_schema(rt.events, rt.toolset))
    objects = [n for n in walk(sent) if n.get("type") == "object" or "properties" in n]
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    assert "anyOf" in schema_keywords(sent) and "$defs" in sent          # kept: the host supports them
    assert not schema_keywords(sent) & {"minimum", "maximum", "default", "pattern"}


def test_strict_keeps_constraints_as_hints():
    sent = to_strict({"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "default": 3}}})
    assert sent["properties"]["n"] == {"type": "integer", "description": "{minimum: 1}"}
    assert sent["required"] == ["n"]


def test_strict_does_not_mistake_property_names_for_keywords():
    sent = to_strict({"type": "object", "properties": {"format": {"type": "string"}, "default": {"type": "string"}}})
    assert list(sent["properties"]) == ["format", "default"]


def test_simple_has_nothing_databricks_rejects(rt):
    for schema in (proposal_schema(rt.events, rt.toolset), decision_schema(rt.toolset), CODE_SCHEMA):
        sent, _ = to_simple(schema)
        assert not schema_keywords(sent) & set(FORBIDDEN_SIMPLE), schema_keywords(sent)
        assert count_keys(sent) <= 64
        for node in walk(sent):
            kind = node.get("type")
            assert isinstance(kind, str) or (isinstance(kind, list) and len(kind) == 2 and "null" in kind), node


def test_simple_turns_optional_into_a_type_list():
    sent, plan = to_simple({"type": "object", "properties": {"q": {"anyOf": [{"type": "string"}, {"type": "null"}]}}})
    assert sent["properties"]["q"]["type"] == ["string", "null"] and plan is None


def test_simple_inlines_references():
    schema = {
        "type": "object", "properties": {"v": {"$ref": "#/$defs/Vendor"}},
        "$defs": {"Vendor": {"type": "object", "properties": {"name": {"type": "string"}}}},
    }
    sent, plan = to_simple(schema)
    assert sent["properties"]["v"]["properties"] == {"name": {"type": "string"}} and plan is None


def test_simple_respects_the_key_budget():
    wide = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(30)}}
    schema = {"type": "object", "properties": {"a": wide, "b": wide, "note": {"type": "string"}}}
    sent, plan = to_simple(schema, max_keys=40)
    assert count_keys(sent) <= 40
    assert sorted(plan["props"].values()) == ["json"]                      # one side was enough
    reply = {k: (json.dumps({"f0": "x"}) if plan["props"].get(k) else v) for k, v in {"a": {"f0": "y"}, "b": {"f0": "y"}, "note": "n"}.items()}
    assert {"f0": "x"} in decode(reply, plan).values()


def test_simple_round_trip_through_a_proposal(rt):
    sent, plan = to_simple(proposal_schema(rt.events, rt.toolset))
    item = sent["properties"]["tests"]["items"]["properties"]
    assert item["event"]["type"] == "string" and "ShipmentEvent" in item["event"]["description"]
    assert item["expected_calls"]["items"]["type"] == "string"
    wire = {
        "ready": True, "question": None, "event_type": "ShipmentEvent", "summary": "s",
        "tests": [{"name": t.name, "event": json.dumps(t.event),
                   "expected_calls": [json.dumps(c.model_dump()) for c in t.expected_calls]} for t in TESTS],
    }
    back = decode(wire, plan)
    assert back["tests"][0]["event"] == TESTS[0].event
    assert back["tests"][0]["expected_calls"] == [c.model_dump() for c in TESTS[0].expected_calls]


def test_decode_reports_bad_json():
    with pytest.raises(ValueError, match="did not parse"):
        decode({"e": "{not json"}, {"props": {"e": "json"}})


# -- provider ----------------------------------------------------------------

class FakeCompletions:
    def __init__(self, replies, finish="stop", refusal=None):
        self.replies, self.finish, self.refusal, self.requests = list(replies), finish, refusal, []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        reply = reply(kwargs) if callable(reply) else reply
        content = reply if isinstance(reply, (str, list)) else json.dumps(reply)
        message = SimpleNamespace(content=content, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish)])


def provider(replies, **kwargs):
    completions = FakeCompletions(replies, **{k: kwargs.pop(k) for k in ("finish", "refusal") if k in kwargs})
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAICompatible("m", client=client, **kwargs), completions


def ask(model, schema=CODE_SCHEMA):
    return asyncio.run(model.json("be precise", [{"role": "user", "content": "hi"}], schema))


def test_request_shape():
    model, sent = provider([{"source": "x"}], max_tokens=500, reasoning_effort="low")
    assert ask(model) == {"source": "x"}
    req = sent.requests[0]
    assert req["model"] == "m" and req["messages"][0] == {"role": "system", "content": "be precise"}
    fmt = req["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["required"] == ["source"]
    assert req["max_completion_tokens"] == 500 and req["reasoning_effort"] == "low"
    assert "temperature" not in req   # some models reject it, so it is only sent when set


def test_reply_as_a_list_of_blocks():
    blocks = [{"type": "reasoning", "summary": "thinking"}, {"type": "text", "text": '{"source": "x"}'}]
    model, _ = provider([blocks])
    assert ask(model) == {"source": "x"}


def test_unreadable_reply_is_retried_with_the_reason():
    model, sent = provider(["{not json", {"source": "x"}])
    assert ask(model) == {"source": "x"}
    assert "could not be read" in sent.requests[1]["messages"][-1]["content"]


def test_gives_up_after_retries():
    model, sent = provider(["nope"] * 3, retries=2)
    with pytest.raises(ModelError, match="unreadable reply after 3 attempts"):
        ask(model)
    assert len(sent.requests) == 3


@pytest.mark.parametrize("kwargs, match", [({"finish": "length"}, "output tokens"), ({"refusal": "no"}, "declined")])
def test_bad_outcomes_raise(kwargs, match):
    model, _ = provider([{"source": "x"}], **kwargs)
    with pytest.raises(ModelError, match=match):
        ask(model)


def test_host_helpers():
    azure = OpenAICompatible.azure("my-resource", "my-deployment", api_key="k")
    assert azure.base_url == "https://my-resource.openai.azure.com/openai/v1/"
    assert azure.model == "my-deployment" and azure.dialect == "openai"
    dbx = OpenAICompatible.databricks("adb-123.azuredatabricks.net", "my-endpoint", token="t")
    assert dbx.base_url == "https://adb-123.azuredatabricks.net/serving-endpoints"
    assert dbx.model == "my-endpoint" and dbx.dialect == "simple"
    with pytest.raises(ValueError):
        OpenAICompatible("m", dialect="loose")


@pytest.mark.parametrize("dialect", ["openai", "simple"])
def test_full_flow_in_both_dialects(dialect):
    def reply(request):
        schema = request["response_format"]["json_schema"]["schema"]
        if "tests" not in schema["properties"]:
            return {"source": GOOD}
        tests = [t.model_dump() for t in TESTS]
        if dialect == "simple":   # what a Databricks model would send: JSON inside strings
            tests = [{"name": t["name"], "event": json.dumps(t["event"]),
                      "expected_calls": [json.dumps(c) for c in t["expected_calls"]]} for t in tests]
        return {"ready": True, "question": None, "event_type": "ShipmentEvent", "summary": "s", "tests": tests}

    model, sent = provider([reply, reply], dialect=dialect)
    rt, texts = make_runtime(model=model)

    async def flow():
        return await rt.build(await rt.propose("acme", "Text Murphy if a Dallas shipment is over 2 hours late"))

    rt.inject(asyncio.run(flow()))
    res = rt.dispatch(ShipmentEvent(customer_id="acme", order_id="ORD-9", warehouse="Dallas", hours_late=4))
    assert res.executed and len(texts) == 1 and len(sent.requests) == 2


def test_propose_rejects_calls_to_unknown_tools():
    from jit_agents import CallableModel

    good = [t.model_dump() for t in TESTS]
    bad = [dict(good[0], expected_calls=[{"tool": "launch_missiles", "args": {}}])] + good[1:]
    replies, asks = [bad, good], []

    def fn(system, messages, schema):
        asks.append(messages[-1]["content"])
        return {"ready": True, "question": None, "event_type": "ShipmentEvent", "summary": "s", "tests": replies.pop(0)}

    rt, _ = make_runtime(model=CallableModel(fn))
    proposal = asyncio.run(rt.propose("acme", "text me"))
    assert len(asks) == 2 and "launch_missiles" in asks[1] and "not an available tool" in asks[1]
    assert proposal.tests[0].expected_calls[0].tool == "notify_sms"


def test_a_web_page_instead_of_an_api_reply_is_explained():
    class Page:
        async def create(self, **kwargs):
            return "<!doctype html><title>Sign in</title>"

    model = OpenAICompatible("m", base_url="https://wrong.example/x", client=SimpleNamespace(chat=SimpleNamespace(completions=Page())))
    with pytest.raises(ModelError, match="did not answer in the OpenAI format"):
        ask(model)
