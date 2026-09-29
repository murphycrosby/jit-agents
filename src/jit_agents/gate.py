"""The gate. Nothing is injected until it passes all three layers.

1. Behaviour tests   confirmed by the customer: this event, these calls.
2. Built-in checks   enforced by the library on every rule.
3. Invariants        policy set by the developer that no rule may violate.

Layers 2 and 3 run over the test events plus events synthesised from the
Pydantic model (see ``jit_agents.synth``). Lookups never run for real here.
They return the values each test provides, and those values are varied too.
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from jit_agents.models import Case, Check, Report
from jit_agents.synth import candidates, variants
from jit_agents.tools import Call, ToolSet, calls_match
from jit_agents.validator import RuleError, collect_constants, load

Invariant = Callable[[Any, list[Call]], bool]
Stubs = dict[str, Any]


@dataclass
class GateConfig:
    latency_budget_us: float = 200.0   # median rule evaluation time
    max_calls_per_event: int = 10
    bench_iterations: int = 2000
    max_synth_events: int = 300


def evaluate(fn: Callable[[Any, Any], Any], event: BaseModel, toolset: ToolSet, stubs: Stubs | None = None) -> Any:
    """Run a rule against a recording proxy and return the proxy."""
    proxy = toolset.recorder({} if stubs is None else stubs)
    fn(event, proxy)
    return proxy


def _short(event: BaseModel, stubs: Stubs) -> str:
    text = event.model_dump_json()
    text = text if len(text) <= 400 else text[:400] + "…"
    return f"{text} with lookups {json.dumps(stubs, default=str)}" if stubs else text


class _Tally:
    """Collects failures for one named check, keeping a few examples."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.examples: list[str] = []

    def fail(self, detail: str) -> None:
        self.count += 1
        if len(self.examples) < 3:
            self.examples.append(detail)

    def check(self) -> Check:
        if not self.count:
            return Check(name=self.name, passed=True, detail="ok")
        return Check(name=self.name, passed=False, detail=f"{self.count} failure(s). " + " | ".join(self.examples))


def _stubs_for(case: Case, toolset: ToolSet) -> Stubs:
    """Lookup values a test provides. A null for a lookup that cannot return None means 'not provided'."""
    lookups = toolset.lookups
    return {
        name: value
        for name, value in case.lookups.items()
        if name in lookups and (value is not None or lookups[name].returns_optional)
    }


def run_gate(
    source: str,
    event_model: type[BaseModel],
    toolset: ToolSet,
    tests: list[Case],
    invariants: dict[str, Invariant],
    config: GateConfig | None = None,
) -> Report:
    cfg = config or GateConfig()
    try:
        fn = load(source, event_model, toolset.names, toolset.returns)
    except RuleError as exc:
        return Report(ok=False, validation_error=str(exc))

    if not tests:
        return Report(ok=False, builtin=[Check(name="has tests", passed=False, detail="a rule needs at least one test")])

    # Values from any test fill in for tests that leave a lookup out.
    defaults: Stubs = {}
    for case in tests:
        for name, value in _stubs_for(case, toolset).items():
            defaults.setdefault(name, value)

    # -- layer 1: behaviour tests -------------------------------------------
    behavior: list[Check] = []
    bases: list[tuple[BaseModel, Stubs]] = []
    for case in tests:
        try:
            event = event_model.model_validate(case.event)
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
            behavior.append(Check(name=case.name, passed=False, detail=f"test event is not a valid {event_model.__name__}: {errs}"))
            continue
        stubs = defaults | _stubs_for(case, toolset)
        bases.append((event, stubs))
        try:
            calls = evaluate(fn, event, toolset, stubs)._calls
        except Exception as exc:
            behavior.append(Check(name=case.name, passed=False, detail=f"raised {type(exc).__name__}: {exc}"))
            continue
        mismatch = calls_match(case.expected_calls, calls)
        if mismatch and stubs:
            mismatch += f" (lookups: {json.dumps(stubs, default=str)})"
        if mismatch and ("*" in source or "/" in source) and "round(" not in source:
            mismatch += (
                ". If this test sits exactly on a computed threshold, floating point is the likely cause: "
                "100 * 1.1 is 110.00000000000001. Round the threshold, e.g. round(avg * 1.1, 2)"
            )
        behavior.append(Check(name=case.name, passed=mismatch is None, detail=mismatch or "ok"))

    # -- layers 2 and 3 over test events plus synthesised events -----------
    nums, strs = collect_constants(source)
    probes: list[tuple[BaseModel, Stubs]] = list(bases)
    per_base = max(1, cfg.max_synth_events // max(1, len(bases)))
    for event, stubs in bases:
        probes += [(v, stubs) for v in variants(event_model, [event], nums, strs, per_base)]
        for name in stubs:  # vary what each lookup returns
            for value in candidates(toolset.lookups[name].returns, nums, strs):
                if value != stubs[name]:
                    probes.append((event, stubs | {name: value}))

    raises = _Tally("never raises")
    determinism = _Tally("deterministic")
    call_cap = _Tally(f"at most {cfg.max_calls_per_event} calls per event")
    args_valid = _Tally("tool arguments valid")
    unmodified = _Tally("event not modified")
    inv_tallies = {name: _Tally(name) for name in invariants}

    for event, stubs in probes:
        before = event.model_dump_json()
        try:
            proxy = evaluate(fn, event, toolset, stubs)
            again = evaluate(fn, event, toolset, stubs)
        except Exception as exc:
            raises.fail(f"{type(exc).__name__}: {exc} on event {_short(event, stubs)}")
            continue
        calls = proxy._calls
        if calls != again._calls:
            determinism.fail(f"two runs differed on event {_short(event, stubs)}")
        if len(calls) > cfg.max_calls_per_event:
            call_cap.fail(f"{len(calls)} calls on event {_short(event, stubs)}")
        for call in (*calls, *proxy._reads):
            problem = toolset.validate_call(call)
            if problem:
                args_valid.fail(f"{problem} on event {_short(event, stubs)}")
        if event.model_dump_json() != before:
            unmodified.fail(f"event changed: {_short(event, stubs)}")
        for name, inv in invariants.items():
            try:
                held = bool(inv(event, calls))
            except Exception as exc:
                inv_tallies[name].fail(f"invariant raised {type(exc).__name__}: {exc}")
                continue
            if not held:
                inv_tallies[name].fail(f"calls {[c.to_dict() for c in calls]} on event {_short(event, stubs)}")

    builtin = [t.check() for t in (raises, determinism, call_cap, args_valid, unmodified)]
    inv_checks = [t.check() for t in inv_tallies.values()]

    # -- latency budget -------------------------------------------------------
    p50 = p99 = None
    so_far = all(c.passed for c in (*behavior, *builtin, *inv_checks))
    if so_far and bases:
        event, stubs = bases[0]
        samples: list[int] = []
        for _ in range(cfg.bench_iterations):
            t0 = time.perf_counter_ns()
            evaluate(fn, event, toolset, stubs)
            samples.append(time.perf_counter_ns() - t0)
        samples.sort()
        p50 = int(statistics.median(samples))
        p99 = samples[max(0, int(len(samples) * 0.99) - 1)]
        within = p50 / 1000 <= cfg.latency_budget_us
        builtin.append(
            Check(
                name="latency budget",
                passed=within,
                detail="ok" if within else f"median {p50 / 1000:.1f}µs exceeds budget {cfg.latency_budget_us}µs",
            )
        )

    ok = all(c.passed for c in (*behavior, *builtin, *inv_checks))
    return Report(
        ok=ok, behavior=behavior, builtin=builtin, invariants=inv_checks,
        events_checked=len(probes), latency_p50_ns=p50, latency_p99_ns=p99,
    )
