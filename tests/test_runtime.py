import asyncio

import pytest

from jit_agents import BuildError, CallableModel, Case, SqliteStore

from .conftest import GOOD, GOOD_V2, TESTS, TESTS_V2, InvoiceEvent, ShipmentEvent, make_runtime

LATE = ShipmentEvent(customer_id="acme", order_id="ORD-9", warehouse="Dallas", hours_late=4)

INVOICE_RULE = '''
def handle(event, tools):
    if event.amount > 500 and not event.trusted_vendor:
        tools.route(queue="fraud_review")
'''
INVOICE_TESTS = [
    Case(name="big untrusted", event={"customer_id": "globex", "amount": 900.5}, expected_calls=[{"tool": "route", "args": {"queue": "fraud_review"}}]),
    Case(name="small", event={"customer_id": "globex", "amount": 100}, expected_calls=[]),
]


def test_inject_and_dispatch_executes_tools(rt_and_sent):
    rt, sent = rt_and_sent
    rule = rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS, intent="text me"))
    assert rule.status == "active" and rule.version == 1

    res = rt.dispatch(LATE)
    assert res.ran and res.executed and res.error is None
    assert [c.tool for c in res.calls] == ["notify_sms"]
    assert sent == [("sms", "Murphy", "Shipment ORD-9 from Dallas is 4.0h late")]


def test_dispatch_without_execute_returns_calls_only(rt_and_sent):
    rt, sent = rt_and_sent
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    res = rt.dispatch(LATE, execute=False)
    assert len(res.calls) == 1 and not res.executed and sent == []


def test_each_key_runs_only_its_own_rule(rt_and_sent):
    rt, sent = rt_and_sent
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    rt.inject(rt.build_from_source("globex", INVOICE_RULE, "InvoiceEvent", INVOICE_TESTS))

    assert [c.tool for c in rt.dispatch(LATE).calls] == ["notify_sms"]
    assert [c.tool for c in rt.dispatch(InvoiceEvent(customer_id="globex", amount=900)).calls] == ["route"]
    # no rule for this key
    assert not rt.dispatch(LATE.model_copy(update={"customer_id": "nobody"})).ran
    # acme's rule is for shipments, so an acme invoice is not run through it
    assert not rt.dispatch(InvoiceEvent(customer_id="acme", amount=900)).ran


def test_dict_events_are_validated(rt):
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    ok = rt.dispatch({"customer_id": "acme", "order_id": "O", "warehouse": "Dallas", "hours_late": 9}, execute=False)
    assert ok.ran and len(ok.calls) == 1
    bad = rt.dispatch({"customer_id": "acme", "order_id": "O", "warehouse": "Mars"}, execute=False)
    assert not bad.ran and "not a valid ShipmentEvent" in bad.error


def test_composite_keys_are_opaque(rt):
    rt.inject(rt.build_from_source(("acme", "ShipmentEvent"), GOOD, "ShipmentEvent", TESTS))
    assert rt.dispatch(LATE, key=("acme", "ShipmentEvent"), execute=False).ran
    assert not rt.dispatch(LATE, key=("acme", "InvoiceEvent"), execute=False).ran


def test_new_rule_replaces_old_and_rollback_restores(rt):
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    v2 = rt.inject(rt.build_from_source("acme", GOOD_V2, "ShipmentEvent", TESTS_V2))
    assert v2.version == 2
    assert rt.dispatch(LATE, execute=False).calls[0].args["to"] == "Ops"
    assert [r.status for r in rt.history("acme")] == ["active", "superseded"]

    restored = rt.rollback("acme")
    assert restored is not None and restored.version == 1
    assert rt.dispatch(LATE, execute=False).calls[0].args["to"] == "Murphy"
    assert [r.status for r in rt.history("acme")] == ["rolled_back", "active"]
    assert rt.rollback("acme") is None


def test_source_that_fails_the_gate_cannot_be_injected(rt):
    with pytest.raises(BuildError):
        rt.build_from_source("acme", GOOD.replace("hours > 2", "hours > 1"), "ShipmentEvent", TESTS)
    assert rt.rule_for("acme") is None


def test_rules_survive_restart_and_lazy_load(tmp_path):
    db = tmp_path / "rules.db"
    rt1, _ = make_runtime(store=SqliteStore(db))
    rule = rt1.inject(rt1.build_from_source("acme", GOOD, "ShipmentEvent", TESTS, intent="text me when late"))

    rt2, sent = make_runtime(store=SqliteStore(db))  # a fresh process
    assert rt2.loaded_keys() == []                   # nothing loaded at startup
    res = rt2.dispatch(LATE)                         # first event pulls the rule in
    assert res.ran and res.rule_id == rule.id and res.version == 1
    assert rt2.loaded_keys() == ["acme"]
    assert sent == [("sms", "Murphy", "Shipment ORD-9 from Dallas is 4.0h late")]
    assert rt2.rule_for("acme").intent == "text me when late"


def test_warm_preloads(tmp_path):
    db = tmp_path / "rules.db"
    rt1, _ = make_runtime(store=SqliteStore(db))
    rt1.inject(rt1.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    rt2, _ = make_runtime(store=SqliteStore(db))
    assert rt2.warm() == 1 and rt2.loaded_keys() == ["acme"]


def test_missing_keys_are_cached(rt):
    calls = {"n": 0}
    original = rt.store.list

    def counting(key):
        calls["n"] += 1
        return original(key)

    rt.store.list = counting
    for _ in range(5):
        assert not rt.dispatch(LATE, execute=False).ran
    assert calls["n"] == 1
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    assert rt.dispatch(LATE, execute=False).ran  # inject clears the cached miss


def test_changed_tools_make_a_rule_stale_on_load(tmp_path):
    db = tmp_path / "rules.db"
    rt1, _ = make_runtime(store=SqliteStore(db))
    rule = rt1.inject(rt1.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))

    from jit_agents import Runtime

    stale: list = []
    rt2 = Runtime(store=SqliteStore(db), key=lambda e: e.customer_id, on_stale=lambda r, rep: stale.append(r.id))
    rt2.register_event(ShipmentEvent)

    @rt2.tool
    def notify_sms(phone: str, message: str) -> None:  # parameter renamed since the rule was built
        """Send an SMS."""

    assert not rt2.dispatch(LATE).ran
    assert stale == [rule.id]
    assert rt2.store.get(rule.id).status == "stale"


def test_compatible_change_revalidates_instead_of_going_stale(tmp_path):
    db = tmp_path / "rules.db"
    rt1, _ = make_runtime(store=SqliteStore(db))
    rule = rt1.inject(rt1.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))

    rt2, _ = make_runtime(store=SqliteStore(db))

    @rt2.tool
    def flag(reason: str) -> None:  # a new tool changes the fingerprint but not the rule's behaviour
        """Flag the event."""

    assert rt2.dispatch(LATE, execute=False).ran
    reloaded = rt2.store.get(rule.id)
    assert reloaded.status == "active" and reloaded.fingerprint != rule.fingerprint


def test_invariant_blocks_execution_at_stream_time(rt_and_sent):
    rt, sent = rt_and_sent
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    rt.invariants["quiet_hours"] = lambda event, calls: not calls  # added after injection, bypassing the gate
    res = rt.dispatch(LATE)
    assert res.blocked and not res.executed and "quiet_hours" in res.error
    assert sent == []


def test_max_loaded_evicts_and_reloads(rt):
    rt.max_loaded = 1
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    rt.inject(rt.build_from_source("globex", INVOICE_RULE, "InvoiceEvent", INVOICE_TESTS))
    assert rt.loaded_keys() == ["globex"]
    assert rt.dispatch(LATE, execute=False).ran  # acme reloads on demand
    assert rt.loaded_keys() == ["acme"]


def test_async_tools_use_adispatch():
    from jit_agents import Runtime

    sent: list = []
    rt = Runtime(key=lambda e: e.customer_id)
    rt.register_event(ShipmentEvent)

    @rt.tool
    async def notify_sms(to: str, message: str) -> None:
        """Send an SMS."""
        sent.append(to)

    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    assert "adispatch" in rt.dispatch(LATE).error
    res = asyncio.run(rt.adispatch(LATE))
    assert res.executed and sent == ["Murphy"]


def test_dispatch_latency_is_microseconds(rt):
    rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    lat = sorted(rt.dispatch(LATE, execute=False).latency_ns for _ in range(2000))
    assert lat[len(lat) // 2] < 50_000


# -- the model-driven path, with a scripted model ---------------------------

def scripted_model(sources: list[str]):
    remaining = list(sources)
    seen: list[str] = []

    def fn(system, messages, schema):
        if "tests" in schema["properties"]:
            return {
                "ready": True, "question": None, "event_type": "ShipmentEvent",
                "summary": "Text Murphy when a Dallas shipment is more than 2 hours late.",
                "tests": [t.model_dump() for t in TESTS],
            }
        seen.append(messages[-1]["content"])
        return {"source": remaining.pop(0)}

    return CallableModel(fn), seen


def test_propose_build_inject_with_retry():
    broken = GOOD.replace("hours > 2", "hours >= 2")
    model, seen = scripted_model([f"```python\n{broken}```", GOOD])
    rt, sent = make_runtime(model=model)

    async def flow():
        proposal = await rt.propose("acme", "Text me if a Dallas shipment is over 2 hours late")
        assert proposal.ready and proposal.event_type == "ShipmentEvent" and len(proposal.tests) == 3
        return await rt.build(proposal)

    rule = asyncio.run(flow())
    assert rule.attempts == 2
    assert "IT FAILED THE GATE WITH" in seen[1] and "exactly two hours" in seen[1]
    rt.inject(rule)
    assert rt.dispatch(LATE).executed and len(sent) == 1


def test_build_gives_up_after_max_attempts():
    model, _ = scripted_model(["import os\ndef handle(event, tools):\n    pass\n"] * 4)
    rt, _ = make_runtime(model=model, max_attempts=3)

    async def flow():
        return await rt.build(await rt.propose("acme", "anything"))

    with pytest.raises(BuildError) as err:
        asyncio.run(flow())
    assert len(err.value.attempts) == 3
    assert "Import" in err.value.report.validation_error


def test_rebuild_regenerates_from_stored_intent():
    model, _ = scripted_model([GOOD])
    rt, _ = make_runtime(model=model)
    rule = rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS, intent="text me"))
    rebuilt = asyncio.run(rt.rebuild(rule.id))
    assert rebuilt.id != rule.id and rebuilt.intent == "text me" and rebuilt.report.ok


# -- lookups: read-only tools that return a value -----------------------------

AVG_RULE = '''
def handle(event, tools):
    avg = tools.average_amount(customer_id=event.customer_id, last=10)
    if avg is not None and avg > 0 and event.amount > avg * 1.1:
        tools.route(queue="fraud_review")
'''
AVG_TESTS = [
    Case(name="over", event={"customer_id": "globex", "amount": 600.5}, lookups={"average_amount": 500.0},
         expected_calls=[{"tool": "route", "args": {"queue": "fraud_review"}}]),
    Case(name="exactly 10% over is not enough", event={"customer_id": "globex", "amount": 550}, lookups={"average_amount": 500.0}),
    Case(name="no history", event={"customer_id": "globex", "amount": 600}, lookups={"average_amount": None}),
]


def runtime_with_lookup():
    rt, sent = make_runtime()
    history = {"globex": [100.0, 300.0]}
    asked: list = []

    @rt.lookup
    def average_amount(customer_id: str, last: int = 10) -> float | None:
        """Average amount of the customer's most recent invoices."""
        asked.append((customer_id, last))
        amounts = history.get(customer_id, [])[-last:]
        return sum(amounts) / len(amounts) if amounts else None

    return rt, sent, history, asked


def test_lookup_value_drives_the_rule_in_production():
    rt, sent, history, asked = runtime_with_lookup()
    rt.inject(rt.build_from_source("globex", AVG_RULE, "InvoiceEvent", AVG_TESTS))
    assert asked == []  # the gate used the test values, never the real lookup

    assert rt.dispatch(InvoiceEvent(customer_id="globex", amount=230)).calls != []   # average is 200
    assert rt.dispatch(InvoiceEvent(customer_id="globex", amount=210)).calls == []
    history["globex"] = []
    assert rt.dispatch(InvoiceEvent(customer_id="globex", amount=9999)).calls == []  # no history
    assert asked == [("globex", 10)] * 3
    assert sent == [("route", "fraud_review")]


def test_gate_varies_lookup_values():
    rt, *_ = runtime_with_lookup()
    unguarded = AVG_RULE.replace("avg is not None and avg > 0 and ", "")
    report = rt.check(unguarded, "InvoiceEvent", AVG_TESTS)
    assert not report.ok
    assert any("TypeError" in c.detail for c in report.behavior + report.builtin)

    no_none = [t for t in AVG_TESTS if t.lookups["average_amount"] is not None]
    report = rt.check(unguarded, "InvoiceEvent", no_none)  # tests alone would pass
    assert all(c.passed for c in report.behavior)
    assert not next(c for c in report.builtin if c.name == "never raises").passed


def test_lookup_arguments_are_validated():
    rt, *_ = runtime_with_lookup()
    bad = AVG_RULE.replace("last=10", 'last="ten"')
    report = rt.check(bad, "InvoiceEvent", AVG_TESTS)
    assert not next(c for c in report.builtin if c.name == "tool arguments valid").passed


def test_lookup_rules():
    rt, *_ = runtime_with_lookup()
    with pytest.raises(TypeError, match="return annotation"):
        @rt.lookup
        def nothing(x: int):
            """No return type."""
    with pytest.raises(TypeError, match="synchronous"):
        @rt.lookup
        async def slow(x: int) -> int:
            """Async."""
    report = rt.check(AVG_RULE.replace("average_amount", "median_amount"), "InvoiceEvent", AVG_TESTS)
    assert "unknown tool" in report.validation_error


def test_lookups_appear_in_prompt_and_schema():
    from jit_agents.generator import _generate_system, _propose_system, proposal_schema

    rt, *_ = runtime_with_lookup()
    assert "tools.average_amount(customer_id: str, last: int = 10)" in _propose_system(rt.events, rt.toolset)
    assert "tools.average_amount" in _generate_system(InvoiceEvent, rt.toolset)
    schema = proposal_schema(rt.events, rt.toolset)
    lookups = schema["properties"]["tests"]["items"]["properties"]["lookups"]
    assert lookups["required"] == ["average_amount"]
    assert "average_amount" not in str(schema["properties"]["tests"]["items"]["properties"]["expected_calls"])


def test_contradictory_tests_are_sent_back_before_building():
    good = [t.model_dump() for t in AVG_TESTS]
    clash = good + [dict(good[0], name="same inputs, opposite answer", expected_calls=[])]
    replies, asks = [clash, good], []

    def fn(system, messages, schema):
        asks.append(messages[-1]["content"])
        return {"ready": True, "question": None, "event_type": "InvoiceEvent", "summary": "s", "tests": replies.pop(0)}

    rt, *_ = runtime_with_lookup()
    rt.model = CallableModel(fn)
    proposal = asyncio.run(rt.propose("globex", "flag invoices over average"))
    assert len(asks) == 2 and "identical inputs" in asks[1]
    assert proposal.ready and len(proposal.tests) == 3


def test_revise_corrects_tests_no_rule_could_pass():
    good = [t.model_dump() for t in AVG_TESTS]
    # Same relevant inputs as the first test, different id-like field, opposite expectation.
    clash = good + [dict(good[0], name="clash", event=dict(good[0]["event"], trusted_vendor=True), expected_calls=[])]
    proposals_out, seen = [clash, good], []

    def fn(system, messages, schema):
        if "tests" in schema["properties"]:
            seen.append(messages[-1]["content"])
            return {"ready": True, "question": None, "event_type": "InvoiceEvent", "summary": "s", "tests": proposals_out.pop(0)}
        return {"source": AVG_RULE}

    rt, *_ = runtime_with_lookup()
    rt.model, rt.max_attempts = CallableModel(fn), 2

    async def flow():
        proposal = await rt.propose("globex", "route invoices over average")
        with pytest.raises(BuildError) as err:
            await rt.build(proposal)
        revised = await rt.revise(proposal, err.value)
        return revised, await rt.build(revised)

    revised, rule = asyncio.run(flow())
    assert "GATE FEEDBACK" in seen[1] and "clash" in seen[1]
    assert revised.intent == "route invoices over average" and len(revised.tests) == 3
    assert rule.report.ok


def test_floating_point_boundary_is_explained_and_fixed_by_rounding():
    at_threshold = [Case(
        name="exactly 10% over counts", event={"customer_id": "globex", "amount": 110}, lookups={"average_amount": 100.0},
        expected_calls=[{"tool": "route", "args": {"queue": "fraud_review"}}],
    )]
    rt, *_ = runtime_with_lookup()
    unrounded = AVG_RULE.replace("event.amount > avg * 1.1", "event.amount >= avg * 1.1")
    report = rt.check(unrounded, "InvoiceEvent", at_threshold)
    assert not report.ok and "110.00000000000001" in report.behavior[0].detail

    rounded = AVG_RULE.replace("event.amount > avg * 1.1", "event.amount >= round(avg * 1.1, 2)")
    assert rt.check(rounded, "InvoiceEvent", at_threshold).ok


@pytest.mark.parametrize("use_sqlite", [False, True])
def test_delete_and_prune_old_versions(tmp_path, use_sqlite):
    rt, _ = make_runtime(store=SqliteStore(tmp_path / "r.db")) if use_sqlite else make_runtime()
    v1 = rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    v2 = rt.inject(rt.build_from_source("acme", GOOD_V2, "ShipmentEvent", TESTS_V2))
    v3 = rt.inject(rt.build_from_source("acme", GOOD, "ShipmentEvent", TESTS))
    other = rt.inject(rt.build_from_source("globex", INVOICE_RULE, "InvoiceEvent", INVOICE_TESTS))

    with pytest.raises(ValueError, match="live"):
        rt.delete(v3.id)
    with pytest.raises(KeyError):
        rt.delete("missing")

    assert rt.delete(v1.id).version == 1
    assert [r.version for r in rt.history("acme")] == [3, 2]

    assert rt.prune("acme", keep=1) == 0   # one old version, keep one
    assert rt.prune("acme") == 1
    assert [(r.version, r.status) for r in rt.history("acme")] == [(3, "active")]
    assert rt.dispatch(LATE, execute=False).ran      # the live rule still runs
    assert rt.rollback("acme") is None               # nothing left to roll back to
    assert rt.rule_for("globex").id == other.id      # other keys untouched


def test_tests_must_exercise_a_fractional_value_for_float_fields():
    from jit_agents.generator import test_problems

    whole = [t.model_copy(update={"event": dict(t.event, hours_late=3)}) for t in TESTS]
    problems = test_problems(whole, ShipmentEvent)
    assert len(problems) == 1 and "hours_late" in problems[0] and "2.5" in problems[0]
    assert test_problems(TESTS, ShipmentEvent) == []          # TESTS has a 3.5
