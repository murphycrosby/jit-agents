"""Data models: proposals, test cases, reports and stored rules."""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


class ExpectedCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    """One behaviour test: this event must produce exactly these tool calls."""

    name: str
    event: dict[str, Any]
    lookups: dict[str, Any] = Field(default_factory=dict)  # value each lookup returns in this test
    expected_calls: list[ExpectedCall] = Field(default_factory=list)


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Report(BaseModel):
    """Result of running a rule through the gate."""

    ok: bool
    validation_error: str | None = None
    behavior: list[Check] = Field(default_factory=list)
    builtin: list[Check] = Field(default_factory=list)
    invariants: list[Check] = Field(default_factory=list)
    events_checked: int = 0
    latency_p50_ns: int | None = None
    latency_p99_ns: int | None = None

    def failures(self) -> list[Check]:
        return [c for c in (*self.behavior, *self.builtin, *self.invariants) if not c.passed]

    def feedback(self) -> str:
        """Text handed back to the model when a rule fails the gate."""
        parts: list[str] = []
        if self.validation_error:
            parts.append(f"STATIC VALIDATION FAILED: {self.validation_error}")
        for c in self.behavior:
            if not c.passed:
                parts.append(f"TEST FAILED [{c.name}]: {c.detail}")
        for c in self.builtin:
            if not c.passed:
                parts.append(f"BUILT-IN CHECK FAILED [{c.name}]: {c.detail}")
        for c in self.invariants:
            if not c.passed:
                parts.append(f"INVARIANT VIOLATED [{c.name}]: {c.detail}")
        return "\n".join(parts) or "unknown failure"


class Proposal(BaseModel):
    """What the agent understood, expressed as tests the customer can confirm."""

    id: str = Field(default_factory=_new_id)
    key: str
    intent: str
    ready: bool
    question: str | None = None
    event_type: str | None = None
    summary: str = ""
    tests: list[Case] = Field(default_factory=list)


RuleStatus = Literal["built", "active", "superseded", "rolled_back", "stale"]


class Rule(BaseModel):
    """A generated rule. The source and the intent are what get persisted."""

    id: str = Field(default_factory=_new_id)
    key: str
    version: int = 0
    intent: str = ""
    summary: str = ""
    event_type: str
    source: str
    tests: list[Case] = Field(default_factory=list)
    fingerprint: str = ""
    status: RuleStatus = "built"
    report: Report | None = None
    attempts: int = 1
    created_at: float = Field(default_factory=time.time)
