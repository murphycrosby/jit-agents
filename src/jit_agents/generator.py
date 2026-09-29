"""The two model calls. Neither of them is in the hot path.

``propose`` turns plain-language intent into tests the customer can confirm.
``generate`` writes the rule in the restricted subset.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from jit_agents.llm import Model
from jit_agents.models import Case, Proposal
from jit_agents.tools import ToolSet
from jit_agents.validator import ALLOWED_METHODS, SAFE_BUILTINS

CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"source": {"type": "string"}},
    "required": ["source"],
    "additionalProperties": False,
}


def _hoist(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Move a Pydantic schema's $defs to the shared root so it can be nested."""
    schema = copy.deepcopy(schema)
    defs.update(schema.pop("$defs", {}))
    return schema


def call_schema(toolset: ToolSet, defs: dict[str, Any]) -> dict[str, Any]:
    """Schema for one tool call: a known tool name with that tool's exact arguments."""
    options = [
        {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": [name]},
                "args": _hoist(spec.args_model.model_json_schema(), defs),
            },
            "required": ["tool", "args"],
            "additionalProperties": False,
        }
        for name, spec in sorted(toolset.actions.items())
    ]
    if not options:
        return {"type": "object", "properties": {"tool": {"type": "string"}}, "required": ["tool"], "additionalProperties": False}
    return {"anyOf": options}


def proposal_schema(events: dict[str, type[BaseModel]], toolset: ToolSet) -> dict[str, Any]:
    """Exact schema for a proposal, built from the registered events and tools.

    No free-form objects: test events must match a registered event model and
    expected calls must match a registered tool's arguments.
    """
    defs: dict[str, Any] = {}
    event_options = [_hoist(m.model_json_schema(), defs) for m in events.values()]
    test_properties: dict[str, Any] = {
        "name": {"type": "string"},
        "event": {"anyOf": event_options},
        "expected_calls": {"type": "array", "items": call_schema(toolset, defs)},
    }
    lookups = toolset.lookups
    if lookups:
        # The value each lookup returns in this test. Null means the test does not use it.
        test_properties["lookups"] = {
            "type": "object",
            "properties": {
                name: {"anyOf": [_hoist(spec.returns_schema(), defs), {"type": "null"}]}
                for name, spec in sorted(lookups.items())
            },
            "required": sorted(lookups),
            "additionalProperties": False,
        }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ready": {"type": "boolean"},
            "question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "event_type": {"type": "string", "enum": list(events)},
            "summary": {"type": "string"},
            "tests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": test_properties,
                    "required": list(test_properties),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["ready", "question", "event_type", "summary", "tests"],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def decision_schema(toolset: ToolSet) -> dict[str, Any]:
    """Schema for a list of tool calls. Used when a model decides per event."""
    defs: dict[str, Any] = {}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"calls": {"type": "array", "items": call_schema(toolset, defs)}},
        "required": ["calls"],
        "additionalProperties": False,
    }
    if defs:
        schema["$defs"] = defs
    return schema


def _schema_json(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), indent=1)


def _propose_system(events: dict[str, type[BaseModel]], toolset: ToolSet) -> str:
    event_blocks = "\n\n".join(f"EVENT TYPE {name}:\n{_schema_json(m)}" for name, m in events.items())
    return f"""You turn a customer's plain-language request into a precise, testable rule for a real-time event stream.
Do NOT write code. Produce a one-paragraph summary and 3 to 5 concrete test cases the customer can confirm by reading them.

{event_blocks}

AVAILABLE TOOLS (the only actions a rule can take):
{toolset.schema_text()}

LOOKUPS (read-only; a rule can call these to get a value that is not on the event):
{toolset.schema_text("lookup")}

Respond with JSON: {{"ready": bool, "question": string or null, "event_type": string, "summary": string, "tests": [...]}}

Guidelines:
- Choose the single event_type the rule applies to.
- Prefer to proceed. Use what the customer said literally: a person's name is a valid recipient, so never ask
  for a phone number or email address when a name was given. Make reasonable choices for wording and state
  them in the summary.
- Ask only when you cannot write the tests without guessing something that changes behaviour, such as a missing
  threshold or no indication of what action to take. Then set ready=false, ask exactly ONE short question,
  and leave tests empty.
- When the conversation contains a current rule and the customer asks for a change, apply the change to that
  rule and keep everything else about it the same.
- Each test is {{"name": str, "event": {{...}}, "lookups": {{...}}, "expected_calls": [{{"tool": str, "args": {{...}}}}]}}.
- "lookups" (only when lookups are listed above) gives the value each lookup returns in that test, for example
  {{"average_amount": 500.0}}. Tests never call the real lookup. Choose values that make the arithmetic easy to
  check, and say in the summary exactly how the lookup is called (which arguments). Include a case where the
  lookup returns null if it can, and the rule should then do nothing. Only expected_calls lists actions.
- Every test event must be a valid instance of the chosen event type, with every required field present.
- Include at least one case where expected_calls is [] because the rule should do nothing.
- When a threshold is involved, include cases just below, exactly at, and just above it.
- Tool arguments are compared literally. Keep messages short and build them from event field values,
  so the same event always produces the same message.
- State in the summary whether each threshold is exclusive ("more than", >) or inclusive ("at least", "or more", >=),
  and make the test at the exact threshold match. "10% over average" on its own is exclusive: an amount of exactly
  110 against an average of 100 does nothing. Say in the summary that computed thresholds are rounded to 2 decimals.
- Tests must not contradict each other. If two tests have the same values in the fields the rule reads and the
  same lookup values, they must expect the same calls. For a threshold of "more than 10% over an average of 100":
  110 is exactly at the threshold and expects nothing, 120 is over it and expects the action.
- Never invent tools or fields."""


def _generate_system(event_model: type[BaseModel], toolset: ToolSet) -> str:
    return f"""You write ONE Python function that implements a customer's rule for a real-time event stream.
It runs on every event with no model involved, so it must be small, deterministic, and must never raise.

A static validator REJECTS anything outside this subset:
- Exactly one definition at module level: def handle(event, tools):
- Statements: if / elif / else, assignment to local variables, return, pass, calls to tools.
- Expressions: comparisons, and / or / not, + - * / % //, x if c else y, f-strings, literals.
- NOT allowed: for, while, comprehensions, lambda, try, with, import, del, assert, augmented assignment (+=),
  attribute or item assignment, slicing, nested functions, any name or attribute starting with an underscore.
- Builtins: {', '.join(sorted(SAFE_BUILTINS))}
- Methods on values: {', '.join(sorted(ALLOWED_METHODS))}
- f-string format specs are limited to forms like .1f, .2f, d.

event is a typed object. Read fields as attributes: event.field_name. Only fields in the schema exist.
Optional fields may be None. Guard them before comparing: value = event.field if event.field is not None else 0

EVENT TYPE {event_model.__name__}:
{_schema_json(event_model)}

tools has exactly these action methods. Call them with keyword arguments. They return nothing.
{toolset.schema_text()}

tools also has these read-only lookups. Call them with keyword arguments and use the value they return.
In the tests each lookup returns the value given in that test's "lookups". A lookup may return None or 0:
guard before comparing or dividing, and do nothing when the value is missing.
{toolset.schema_text("lookup")}

Floating point: 100 * 1.1 is 110.00000000000001, so an amount of 110 compares wrongly against it with both > and >=.
ALWAYS round a computed threshold before comparing: threshold = round(avg * 1.1, 2).
Use > when the summary says "more than" and >= when it says "at least" or "or more".

The function must produce EXACTLY the expected tool calls for every test, in order, with the exact argument strings.

Example of the style:
def handle(event, tools):
    hours = event.hours_late if event.hours_late is not None else 0
    if event.warehouse == "Dallas" and hours > 2:
        tools.notify_sms(to="Murphy", message=f"Shipment {{event.order_id}} from Dallas is {{hours}}h late")

Respond with JSON: {{"source": "<python source>"}}"""


PROPOSE_ATTEMPTS = 3


def test_problems(tests: list[Case], event_model: type[BaseModel]) -> list[str]:
    """Problems that make a set of tests impossible to satisfy, found before any code is written."""
    problems: list[str] = []
    seen: dict[str, Case] = {}
    for case in tests:
        try:
            event = event_model.model_validate(case.event)
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
            problems.append(f"test {case.name!r}: event is not a valid {event_model.__name__}: {errs}")
            continue
        inputs = event.model_dump_json() + json.dumps(case.lookups, sort_keys=True, default=str)
        other = seen.setdefault(inputs, case)
        if other is not case and other.expected_calls != case.expected_calls:
            problems.append(
                f"tests {other.name!r} and {case.name!r} have identical inputs but expect different calls. "
                "A rule gives one answer per input. Change the inputs of one of them so they differ, "
                "and check the arithmetic: 110 is exactly 10% over 100, not more than 10% over."
            )
    return problems


test_problems.__test__ = False  # not a pytest test


def _strip_fences(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip() + "\n"


async def propose(
    model: Model,
    *,
    key: str,
    intent: str,
    history: list[dict[str, str]] | None,
    events: dict[str, type[BaseModel]],
    toolset: ToolSet,
) -> Proposal:
    messages = list(history or [])
    if not messages or messages[-1].get("content") != intent:
        messages.append({"role": "user", "content": intent})
    system, schema = _propose_system(events, toolset), proposal_schema(events, toolset)

    for attempt in range(PROPOSE_ATTEMPTS):
        data = await model.json(system, messages, schema)
        tests = [Case.model_validate(t) for t in data.get("tests") or []]
        event_type = data.get("event_type")
        ready = bool(data.get("ready")) and bool(tests) and event_type in events
        problems = test_problems(tests, events[event_type]) if ready else []
        if not problems or attempt == PROPOSE_ATTEMPTS - 1:
            break
        messages = messages + [
            {"role": "assistant", "content": json.dumps(data)},
            {"role": "user", "content": "Those tests have problems. Fix them and send the complete proposal again:\n- "
                                        + "\n- ".join(problems)},
        ]
    return Proposal(
        key=key,
        intent=intent,
        ready=ready,
        question=None if ready else (data.get("question") or "Could you say more about what should happen?"),
        event_type=event_type if event_type in events else None,
        summary=str(data.get("summary", "")).strip(),
        tests=tests if ready else [],
    )


async def revise(
    model: Model,
    *,
    proposal: Proposal,
    feedback: str,
    source: str,
    events: dict[str, type[BaseModel]],
    toolset: ToolSet,
) -> Proposal:
    """Correct a proposal whose tests no rule could satisfy."""
    previous = proposal.model_dump(include={"ready", "question", "event_type", "summary", "tests"})
    complaint = (
        "No rule could satisfy those tests after several attempts, so the tests are probably wrong.\n\n"
        f"GATE FEEDBACK ON THE LAST ATTEMPT:\n{feedback}\n\nLAST ATTEMPT:\n{source}\n\n"
        "Look for tests that contradict each other or the summary: the same relevant inputs with different "
        "expected calls, arithmetic that is off, or a message format that differs between tests. "
        "Keep my intent. Correct the tests and send the complete proposal again."
    )
    history = [
        {"role": "user", "content": proposal.intent},
        {"role": "assistant", "content": json.dumps(previous, default=str)},
        {"role": "user", "content": complaint},
    ]
    revised = await propose(model, key=proposal.key, intent=complaint, history=history, events=events, toolset=toolset)
    revised.intent = proposal.intent
    return revised


async def generate(
    model: Model,
    *,
    summary: str,
    event_model: type[BaseModel],
    toolset: ToolSet,
    tests: list[Case],
    feedback: str | None = None,
    previous: str | None = None,
) -> str:
    user = (
        f"RULE SUMMARY:\n{summary}\n\n"
        f"TEST CASES (the function must pass all of them):\n"
        f"{json.dumps([t.model_dump() for t in tests], indent=1)}"
    )
    if feedback:
        user += (
            f"\n\nYOUR PREVIOUS ATTEMPT:\n{previous}\n\nIT FAILED THE GATE WITH:\n{feedback}\n\n"
            "Fix the function. Stay inside the allowed subset."
        )
    data = await model.json(_generate_system(event_model, toolset), [{"role": "user", "content": user}], CODE_SCHEMA)
    return _strip_fences(str(data.get("source", "")))
