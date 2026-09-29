"""The runtime: register, propose, build, inject, dispatch."""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from jit_agents import generator
from jit_agents.gate import GateConfig, Invariant, run_gate
from jit_agents.llm import Model
from jit_agents.models import Case, Proposal, Report, Rule
from jit_agents.store import MemoryStore, Store
from jit_agents.tools import Call, ToolSet
from jit_agents.validator import RULE_LANGUAGE_VERSION, RuleError, is_model, load

log = logging.getLogger("jit_agents")


class BuildError(Exception):
    """No generated rule passed the gate within the allowed attempts."""

    def __init__(self, message: str, attempts: list[tuple[str, Report]]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.report = attempts[-1][1] if attempts else None
        self.source = attempts[-1][0] if attempts else ""


@dataclass(slots=True)
class Dispatch:
    key: str
    ran: bool
    rule_id: str | None = None
    version: int | None = None
    calls: list[Call] = field(default_factory=list)
    latency_ns: int = 0
    executed: bool = False
    blocked: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "ran": self.ran, "rule_id": self.rule_id, "version": self.version,
            "calls": [c.to_dict() for c in self.calls], "latency_ns": self.latency_ns,
            "latency_us": round(self.latency_ns / 1000, 2), "executed": self.executed,
            "blocked": self.blocked, "error": self.error,
        }


@dataclass(slots=True)
class _Loaded:
    rule_id: str
    version: int
    event_model: type[BaseModel]
    fn: Callable[[Any, Any], Any]


class Runtime:
    def __init__(
        self,
        model: Model | None = None,
        store: Store | None = None,
        key: Callable[[Any], Any] | None = None,
        gate: GateConfig | None = None,
        max_attempts: int = 4,
        enforce_invariants: bool = True,
        max_loaded: int | None = None,
        on_stale: Callable[[Rule, Report], None] | None = None,
    ) -> None:
        self.model = model
        self.store: Store = store or MemoryStore()
        self.key_fn = key
        self.gate = gate or GateConfig()
        self.max_attempts = max_attempts
        self.enforce_invariants = enforce_invariants
        self.max_loaded = max_loaded
        self.on_stale = on_stale

        self.events: dict[str, type[BaseModel]] = {}
        self.toolset = ToolSet()
        self.invariants: dict[str, Invariant] = {}

        self._active: OrderedDict[str, _Loaded] = OrderedDict()
        self._missing: set[str] = set()
        self._lock = threading.Lock()
        self._key_cache: dict[Any, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_event(self, model: type[BaseModel]) -> type[BaseModel]:
        if not is_model(model):
            raise TypeError("events must be Pydantic models")
        self.events[model.__name__] = model
        return model

    def tool(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        self.toolset.add(fn)
        return fn

    def lookup(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a read-only function whose return value a rule can use.

        Lookups run during evaluation, in the hot path. Keep them fast and
        free of side effects. In the gate they are replaced by test values.
        """
        self.toolset.add(fn, kind="lookup")
        return fn

    def invariant(self, fn: Invariant) -> Invariant:
        self.invariants[fn.__name__] = fn
        return fn

    # ------------------------------------------------------------------
    # Keys and fingerprints
    # ------------------------------------------------------------------
    def normalize_key(self, key: Any) -> str:
        """Keys are opaque. Strings pass through; anything else is serialised."""
        if isinstance(key, str):
            return key
        try:
            cached = self._key_cache.get(key)
        except TypeError:
            return json.dumps(key, default=str, sort_keys=True)
        if cached is None:
            cached = self._key_cache[key] = json.dumps(key, default=str, sort_keys=True)
        return cached

    def fingerprint(self, event_type: str) -> str:
        """Hash of everything a rule was built against."""
        invariants = {}
        for name, fn in sorted(self.invariants.items()):
            try:
                invariants[name] = inspect.getsource(fn)
            except (OSError, TypeError):
                invariants[name] = name
        payload = {
            "language": RULE_LANGUAGE_VERSION,
            "event": self._event_model(event_type).model_json_schema(),
            "tools": self.toolset.json_schema(),
            "invariants": invariants,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _event_model(self, event_type: str) -> type[BaseModel]:
        try:
            return self.events[event_type]
        except KeyError:
            raise KeyError(f"event type {event_type!r} is not registered") from None

    # ------------------------------------------------------------------
    # Conversation time: the model is involved
    # ------------------------------------------------------------------
    def _require_model(self) -> Model:
        if self.model is None:
            raise RuntimeError("this Runtime has no model; pass model=... to propose or build")
        return self.model

    async def propose(self, key: Any, intent: str, history: list[dict[str, str]] | None = None) -> Proposal:
        if not self.events:
            raise RuntimeError("register at least one event model before proposing")
        return await generator.propose(
            self._require_model(), key=self.normalize_key(key), intent=intent,
            history=history, events=self.events, toolset=self.toolset,
        )

    def check(self, source: str, event_type: str, tests: list[Case]) -> Report:
        """Run a rule through the gate without saving or injecting it."""
        return run_gate(source, self._event_model(event_type), self.toolset, tests, self.invariants, self.gate)

    async def build(
        self, proposal: Proposal, on_progress: Callable[[str, dict[str, Any]], None] | None = None
    ) -> Rule:
        """Generate a rule for a confirmed proposal and pass it through the gate.

        ``on_progress`` is called with ("generating", {"attempt"}) before each model
        call and ("checked", {"attempt", "source", "report"}) after each gate run.
        """
        if not proposal.ready or not proposal.event_type:
            raise ValueError("proposal is not ready; answer its question and propose again")
        model = self._require_model()
        event_model = self._event_model(proposal.event_type)
        attempts: list[tuple[str, Report]] = []
        feedback = previous = None
        for n in range(1, self.max_attempts + 1):
            if on_progress:
                on_progress("generating", {"attempt": n})
            source = await generator.generate(
                model, summary=proposal.summary, event_model=event_model, toolset=self.toolset,
                tests=proposal.tests, feedback=feedback, previous=previous,
            )
            report = self.check(source, proposal.event_type, proposal.tests)
            attempts.append((source, report))
            if on_progress:
                on_progress("checked", {"attempt": n, "source": source, "report": report})
            if report.ok:
                return Rule(
                    key=proposal.key, intent=proposal.intent, summary=proposal.summary,
                    event_type=proposal.event_type, source=source, tests=proposal.tests,
                    fingerprint=self.fingerprint(proposal.event_type), report=report, attempts=len(attempts),
                )
            feedback, previous = report.feedback(), source
        raise BuildError(f"no rule passed the gate in {len(attempts)} attempt(s)", attempts)

    async def revise(self, proposal: Proposal, error: BuildError) -> Proposal:
        """Ask the model to correct a proposal whose tests no rule could satisfy.

        The revised tests differ from what the customer confirmed, so show them
        to the customer again before building, unless you build unattended.
        """
        return await generator.revise(
            self._require_model(), proposal=proposal,
            feedback=error.report.feedback() if error.report else str(error),
            source=error.source, events=self.events, toolset=self.toolset,
        )

    def build_from_source(
        self, key: Any, source: str, event_type: str, tests: list[Case], intent: str = "", summary: str = ""
    ) -> Rule:
        """Build a rule from hand-written or previously generated source. Still gated."""
        report = self.check(source, event_type, tests)
        if not report.ok:
            raise BuildError("source did not pass the gate", [(source, report)])
        return Rule(
            key=self.normalize_key(key), intent=intent, summary=summary, event_type=event_type,
            source=source, tests=tests, fingerprint=self.fingerprint(event_type), report=report,
        )

    async def rebuild(self, rule_id: str) -> Rule:
        """Regenerate a rule from its stored intent and tests, e.g. after it went stale."""
        old = self.store.get(rule_id)
        if old is None:
            raise KeyError(rule_id)
        proposal = Proposal(
            key=old.key, intent=old.intent, ready=True, event_type=old.event_type,
            summary=old.summary or old.intent, tests=old.tests,
        )
        return await self.build(proposal)

    # ------------------------------------------------------------------
    # Injection and lifecycle
    # ------------------------------------------------------------------
    def inject(self, rule: Rule, key: Any | None = None) -> Rule:
        """Save the rule and make it live for its key. One rule per key."""
        if key is not None:
            rule.key = self.normalize_key(key)
        current = self.fingerprint(rule.event_type)
        if rule.report is None or not rule.report.ok or rule.fingerprint != current:
            report = self.check(rule.source, rule.event_type, rule.tests)
            if not report.ok:
                raise BuildError("rule did not pass the gate", [(rule.source, report)])
            rule.report, rule.fingerprint = report, current
        loaded = self._load(rule)

        with self._lock:
            versions = self.store.list(rule.key)
            for other in versions:
                if other.status == "active" and other.id != rule.id:
                    other.status = "superseded"
                    self.store.save(other)
            if rule.version == 0:
                rule.version = max((r.version for r in versions), default=0) + 1
                loaded.version = rule.version
            rule.status = "active"
            self.store.save(rule)
            self._put(rule.key, loaded)
        return rule

    def rollback(self, key: Any) -> Rule | None:
        """Re-activate the newest superseded version for this key."""
        k = self.normalize_key(key)
        versions = self.store.list(k)
        previous = next((r for r in versions if r.status == "superseded"), None)
        if previous is None:
            return None
        current = next((r for r in versions if r.status == "active"), None)
        restored = self._restore(previous)
        if restored is None:
            return None
        with self._lock:
            if current is not None:
                current.status = "rolled_back"
                self.store.save(current)
            previous.status = "active"
            self.store.save(previous)
            self._put(k, restored)
        return previous

    def remove(self, key: Any) -> None:
        """Deactivate the rule for this key. History is kept."""
        k = self.normalize_key(key)
        with self._lock:
            for r in self.store.list(k):
                if r.status == "active":
                    r.status = "superseded"
                    self.store.save(r)
            self._active.pop(k, None)
            self._missing.add(k)

    def delete(self, rule_id: str) -> Rule:
        """Permanently delete one stored version. The live version cannot be deleted."""
        rule = self.store.get(rule_id)
        if rule is None:
            raise KeyError(rule_id)
        if rule.status == "active":
            raise ValueError("this version is live; remove or replace it before deleting")
        self.store.delete(rule_id)
        return rule

    def prune(self, key: Any, keep: int = 0) -> int:
        """Permanently delete old versions for a key, keeping the live one and the newest ``keep`` others.

        Deleted versions can no longer be rolled back to. Returns how many were deleted.
        """
        old = [r for r in self.store.list(self.normalize_key(key)) if r.status != "active"]
        doomed = old[max(0, keep):]
        for rule in doomed:
            self.store.delete(rule.id)
        return len(doomed)

    def evict(self, key: Any) -> None:
        """Drop the loaded function from memory. It reloads on the next event."""
        with self._lock:
            self._active.pop(self.normalize_key(key), None)

    def warm(self, keys: list[Any] | None = None) -> int:
        """Load rules ahead of traffic. Returns how many are loaded."""
        targets = [self.normalize_key(k) for k in keys] if keys is not None else self.store.keys()
        return sum(1 for k in targets if self._resolve(k) is not None)

    def rule_for(self, key: Any) -> Rule | None:
        return next((r for r in self.store.list(self.normalize_key(key)) if r.status == "active"), None)

    def history(self, key: Any) -> list[Rule]:
        return self.store.list(self.normalize_key(key))

    def loaded_keys(self) -> list[str]:
        return list(self._active)

    # -- loading ------------------------------------------------------------
    def _load(self, rule: Rule) -> _Loaded:
        event_model = self._event_model(rule.event_type)
        fn = load(rule.source, event_model, self.toolset.names, self.toolset.returns)
        return _Loaded(rule.id, rule.version, event_model, fn)

    def _put(self, key: str, loaded: _Loaded) -> None:
        self._active[key] = loaded
        self._active.move_to_end(key)
        self._missing.discard(key)
        if self.max_loaded is not None:
            while len(self._active) > self.max_loaded:
                self._active.popitem(last=False)

    def _restore(self, rule: Rule) -> _Loaded | None:
        """Load a stored rule. Re-run the gate only if what it was built against changed."""
        try:
            current = self.fingerprint(rule.event_type)
            if rule.fingerprint != current:
                report = self.check(rule.source, rule.event_type, rule.tests)
                if not report.ok:
                    self._mark_stale(rule, report)
                    return None
                rule.report, rule.fingerprint = report, current
                self.store.save(rule)
            return self._load(rule)
        except (RuleError, KeyError) as exc:
            self._mark_stale(rule, Report(ok=False, validation_error=str(exc)))
            return None

    def _mark_stale(self, rule: Rule, report: Report) -> None:
        rule.status, rule.report = "stale", report
        self.store.save(rule)
        log.warning("rule %s for key %s is stale: %s", rule.id, rule.key, report.feedback())
        if self.on_stale:
            self.on_stale(rule, report)

    def _resolve(self, key: str) -> _Loaded | None:
        """Lazy load: the first event for a key pulls its rule from the store."""
        with self._lock:
            loaded = self._active.get(key)
            if loaded is not None:
                return loaded
            if key in self._missing:
                return None
            rule = next((r for r in self.store.list(key) if r.status == "active"), None)
            loaded = self._restore(rule) if rule is not None else None
            if loaded is None:
                self._missing.add(key)
                return None
            self._put(key, loaded)
            return loaded

    # ------------------------------------------------------------------
    # Stream time: no model involved
    # ------------------------------------------------------------------
    def _evaluate(self, event: Any, key: Any | None) -> tuple[Dispatch, bool]:
        """Run the rule. Returns the dispatch and whether its calls may be executed."""
        if key is None:
            if self.key_fn is None:
                raise ValueError("pass key=... to dispatch or give the Runtime a key function")
            key = self.key_fn(event)
        k = key if isinstance(key, str) else self.normalize_key(key)

        loaded = self._active.get(k)
        if loaded is None:
            loaded = self._resolve(k)
            if loaded is None:
                return Dispatch(k, ran=False), False
        elif self.max_loaded is not None:
            self._active.move_to_end(k)

        if not isinstance(event, loaded.event_model):
            if not isinstance(event, dict):
                return Dispatch(k, ran=False, rule_id=loaded.rule_id, version=loaded.version), False
            try:
                event = loaded.event_model.model_validate(event)
            except ValidationError as exc:
                return Dispatch(
                    k, ran=False, rule_id=loaded.rule_id, version=loaded.version,
                    error=f"event is not a valid {loaded.event_model.__name__}: {exc.error_count()} error(s)",
                ), False

        proxy = self.toolset.recorder()
        t0 = time.perf_counter_ns()
        try:
            loaded.fn(event, proxy)
        except Exception as exc:
            return Dispatch(
                k, ran=True, rule_id=loaded.rule_id, version=loaded.version,
                latency_ns=time.perf_counter_ns() - t0, error=f"{type(exc).__name__}: {exc}",
            ), False
        result = Dispatch(
            k, ran=True, rule_id=loaded.rule_id, version=loaded.version,
            calls=proxy._calls, latency_ns=time.perf_counter_ns() - t0,
        )

        if self.enforce_invariants and self.invariants:
            for name, inv in self.invariants.items():
                try:
                    held = bool(inv(event, result.calls))
                except Exception as exc:
                    held = False
                    name = f"{name} ({type(exc).__name__}: {exc})"
                if not held:
                    result.blocked = True
                    result.error = f"invariant violated: {name}"
                    return result, False
        return result, True

    def dispatch(self, event: Any, key: Any | None = None, execute: bool = True) -> Dispatch:
        """Evaluate the rule for this event's key, then execute its tool calls."""
        result, may_execute = self._evaluate(event, key)
        if execute and may_execute:
            try:
                for call in result.calls:
                    self.toolset.execute(call)
                result.executed = True
            except Exception as exc:
                result.error = f"tool failed: {type(exc).__name__}: {exc}"
        return result

    async def adispatch(self, event: Any, key: Any | None = None, execute: bool = True) -> Dispatch:
        """Like dispatch, for async tools."""
        result, may_execute = self._evaluate(event, key)
        if execute and may_execute:
            try:
                for call in result.calls:
                    await self.toolset.aexecute(call)
                result.executed = True
            except Exception as exc:
                result.error = f"tool failed: {type(exc).__name__}: {exc}"
        return result
