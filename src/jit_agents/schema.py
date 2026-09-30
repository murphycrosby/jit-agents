"""Adapt the library's JSON schemas to what a model host accepts.

Hosts that speak the OpenAI format disagree about which parts of JSON Schema
they allow in structured outputs. Two dialects cover the ones we target.

``to_strict``  OpenAI and Azure Foundry. Every property is required, every
               object is closed, and validation keywords the host rejects are
               moved into the description. ``anyOf`` and ``$ref`` are kept.

``to_simple``  Databricks. No ``anyOf``, ``oneOf``, ``allOf`` or ``$ref``, and
               a cap on the number of keys. Anything that cannot be expressed
               is sent as a string holding JSON, with its schema written into
               the description. ``decode`` turns those strings back into values.
"""
from __future__ import annotations

import copy
import json
from typing import Any

# Keywords OpenAI-format hosts reject in strict mode. They are kept as hints in the description.
UNSUPPORTED = frozenset({
    "minLength", "maxLength", "pattern", "format",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "patternProperties", "unevaluatedProperties", "propertyNames", "minProperties", "maxProperties",
    "unevaluatedItems", "contains", "minContains", "maxContains", "minItems", "maxItems", "uniqueItems",
})
DROPPED = frozenset({"default", "title"})
COMPOSITION = ("anyOf", "oneOf", "allOf")

Plan = Any  # None, "json", {"props": {name: Plan}}, or {"items": Plan}


def _clean(node: dict[str, Any]) -> dict[str, Any]:
    """Copy a schema node without rejected keywords, noting them in the description."""
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in node.items():
        if key in DROPPED:
            continue
        if key in UNSUPPORTED:
            notes.append(f"{key}: {json.dumps(value)}")
            continue
        out[key] = value
    if notes:
        out["description"] = (out.get("description", "") + " {" + ", ".join(notes) + "}").strip()
    return out


def _close(out: dict[str, Any]) -> None:
    if "properties" in out or out.get("type") == "object":
        out.setdefault("properties", {})
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False


# ---------------------------------------------------------------------------
# Strict dialect
# ---------------------------------------------------------------------------

def to_strict(schema: dict[str, Any]) -> dict[str, Any]:
    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = _clean(node)
        for key, value in list(out.items()):
            if key in ("properties", "$defs"):
                out[key] = {name: walk(sub) for name, sub in value.items()}
            elif key not in ("enum", "const", "required", "description"):
                out[key] = walk(value)
        _close(out)
        return out

    return walk(copy.deepcopy(schema))


# ---------------------------------------------------------------------------
# Simple dialect
# ---------------------------------------------------------------------------

def _inline(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace every $ref with the definition it points to."""
    defs = schema.get("$defs", {})

    def walk(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [walk(n, stack) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            if name in stack or name not in defs:
                return {"description": f"a {name} value"}  # recursive or unknown: left free-form, then encoded
            rest = {k: v for k, v in node.items() if k != "$ref"}
            return {**walk(defs[name], stack + (name,)), **walk(rest, stack)}
        return {
            k: ({n: walk(s, stack) for n, s in v.items()} if k == "properties" else walk(v, stack))
            for k, v in node.items() if k != "$defs"
        }

    return walk(copy.deepcopy(schema), ())


def _encoded(node: dict[str, Any]) -> tuple[dict[str, Any], Plan]:
    spec = json.dumps(_clean(node), separators=(",", ":"))
    lead = (node.get("description", "") + " ").lstrip()
    return {
        "type": "string",
        "description": f"{lead}A JSON value written as a string. It must match this JSON schema: {spec}",
    }, "json"


def _convert(node: Any, path: tuple[Any, ...], force: set[tuple[Any, ...]]) -> tuple[Any, Plan]:
    if not isinstance(node, dict):
        return node, None
    if path in force:
        return _encoded(node)

    if any(k in node for k in COMPOSITION):
        options = node.get("anyOf") or node.get("oneOf") or []
        real = [o for o in options if o.get("type") != "null"]
        if "allOf" not in node and len(real) == 1 and len(real) < len(options):
            inner, plan = _convert(real[0], path, force)   # Optional[X] becomes a [type, "null"] list
            if plan != "json" and isinstance(inner.get("type"), str):
                inner = {**inner, "type": [inner["type"], "null"]}
                if "description" in node:
                    inner["description"] = node["description"]
                if "enum" in inner and None not in inner["enum"]:
                    inner["enum"] = [*inner["enum"], None]
                return inner, plan
        return _encoded(node)

    kind = node.get("type")
    if isinstance(kind, list) and not (len(kind) == 2 and "null" in kind):
        return _encoded(node)
    base = kind if isinstance(kind, str) else next((t for t in kind or [] if t != "null"), None)

    out = _clean(node)
    if base == "object" or "properties" in node:
        if not node.get("properties"):
            return _encoded(node)  # free-form object
        props: dict[str, Any] = {}
        plans: dict[str, Plan] = {}
        for name, sub in node["properties"].items():
            props[name], plan = _convert(sub, path + (name,), force)
            if plan is not None:
                plans[name] = plan
        out["properties"] = props
        _close(out)
        return out, ({"props": plans} if plans else None)
    if base == "array":
        if not isinstance(node.get("items"), dict):
            return _encoded(node)
        out["items"], plan = _convert(node["items"], path + (0,), force)
        return out, ({"items": plan} if plan is not None else None)
    if base is None and "enum" not in node and "const" not in node:
        return _encoded(node)  # untyped: anything goes
    return out, None


def count_keys(schema: Any) -> int:
    if isinstance(schema, list):
        return sum(count_keys(s) for s in schema)
    if not isinstance(schema, dict):
        return 0
    total = 0
    for key, value in schema.items():
        if key == "properties":
            total += len(value) + sum(count_keys(v) for v in value.values())
        elif key not in ("enum", "const", "required", "description"):
            total += count_keys(value)
    return total


def _nested_objects(schema: Any, path: tuple[Any, ...] = ()) -> list[tuple[int, tuple[Any, ...]]]:
    """Every object below the root that could be encoded, with its key count."""
    found: list[tuple[int, tuple[Any, ...]]] = []
    if not isinstance(schema, dict):
        return found
    if path and schema.get("properties"):
        found.append((count_keys(schema), path))
    for name, sub in (schema.get("properties") or {}).items():
        found += _nested_objects(sub, path + (name,))
    if isinstance(schema.get("items"), dict):
        found += _nested_objects(schema["items"], path + (0,))
    return found


def to_simple(schema: dict[str, Any], max_keys: int = 64) -> tuple[dict[str, Any], Plan]:
    """Return the simplified schema and the plan ``decode`` needs to undo it."""
    flat = _inline(schema)
    force: set[tuple[Any, ...]] = set()
    simple, plan = _convert(flat, (), force)
    while count_keys(simple) > max_keys:
        candidates = [c for c in _nested_objects(simple) if c[1] not in force]
        if not candidates:
            break
        force.add(max(candidates, key=lambda c: (c[0], -len(c[1])))[1])  # largest subtree first
        simple, plan = _convert(flat, (), force)
    return simple, plan


def decode(data: Any, plan: Plan) -> Any:
    """Turn JSON-in-a-string values back into real values. Raises ValueError on bad JSON."""
    if plan is None or data is None:
        return data
    if plan == "json":
        if not isinstance(data, str):
            return data
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"a field that should hold JSON did not parse ({exc.msg}): {data[:200]}") from exc
    if isinstance(data, dict) and "props" in plan:
        return {k: decode(v, plan["props"].get(k)) for k, v in data.items()}
    if isinstance(data, list) and "items" in plan:
        return [decode(v, plan["items"]) for v in data]
    return data
