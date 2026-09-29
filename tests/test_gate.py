from jit_agents import Case, GateConfig

from .conftest import GOOD, TESTS


def test_good_rule_passes_all_layers(rt):
    report = rt.check(GOOD, "ShipmentEvent", TESTS)
    assert report.ok, report.feedback()
    assert all(c.passed for c in report.behavior)
    assert {c.name for c in report.builtin} >= {"never raises", "deterministic", "tool arguments valid", "latency budget"}
    assert report.events_checked > len(TESTS)  # synthesised events were added
    assert report.latency_p50_ns is not None


def test_behavior_mismatch_fails(rt):
    report = rt.check(GOOD.replace('to="Murphy"', 'to="Nobody"'), "ShipmentEvent", TESTS)
    assert not report.ok
    assert not report.behavior[0].passed
    assert "Nobody" in report.behavior[0].detail


def test_off_by_one_threshold_is_caught_by_boundary_test(rt):
    report = rt.check(GOOD.replace("hours > 2", "hours >= 2"), "ShipmentEvent", TESTS)
    assert not report.ok
    assert not report.behavior[1].passed


def test_unguarded_optional_fails_never_raises(rt):
    fragile = (
        "def handle(event, tools):\n"
        "    if event.warehouse == 'Dallas' and event.hours_late > 2:\n"
        "        tools.notify_sms(to='Murphy', message=f'Shipment {event.order_id} from Dallas is {event.hours_late}h late')\n"
    )
    report = rt.check(fragile, "ShipmentEvent", TESTS)
    assert not report.ok
    never_raises = next(c for c in report.builtin if c.name == "never raises")
    assert not never_raises.passed and "TypeError" in never_raises.detail


def test_invalid_tool_arguments_fail(rt):
    source = "def handle(event, tools):\n    tools.route(queue='nowhere')\n"
    tests = [Case(name="routes", event=TESTS[0].event, expected_calls=[{"tool": "route", "args": {"queue": "nowhere"}}])]
    report = rt.check(source, "ShipmentEvent", tests)
    assert not report.ok
    assert not next(c for c in report.builtin if c.name == "tool arguments valid").passed


def test_invalid_test_event_fails(rt):
    bad = [TESTS[0].model_copy(update={"event": {"customer_id": "acme", "warehouse": "Mars"}})]
    report = rt.check(GOOD, "ShipmentEvent", bad)
    assert not report.ok
    assert "not a valid ShipmentEvent" in report.behavior[0].detail


def test_invariant_blocks_rule_at_the_gate(rt):
    @rt.invariant
    def never_text_murphy(event, calls):
        return not any(c.tool == "notify_sms" and c.args.get("to") == "Murphy" for c in calls)

    report = rt.check(GOOD, "ShipmentEvent", TESTS)
    assert not report.ok
    assert report.invariants[0].name == "never_text_murphy" and not report.invariants[0].passed


def test_latency_budget_is_enforced(rt):
    rt.gate = GateConfig(latency_budget_us=0.0001, bench_iterations=50)
    report = rt.check(GOOD, "ShipmentEvent", TESTS)
    assert not report.ok
    assert not next(c for c in report.builtin if c.name == "latency budget").passed


def test_rule_with_no_tests_is_rejected(rt):
    assert not rt.check(GOOD, "ShipmentEvent", []).ok
