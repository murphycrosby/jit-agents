"""Generate extra events from a Pydantic model to probe a rule.

Starting from the confirmed test events, vary one field at a time: every
Literal and Enum option, both booleans, None for optionals, awkward strings,
and numbers just below, at, and just above each constant found in the rule.
Only variants that validate against the model are kept.
"""
from __future__ import annotations

import copy
import enum
import types
from typing import Annotated, Any, Iterator, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError

from jit_agents.validator import is_model, unwrap


def candidates(ann: Any, nums: list[float], strs: list[str]) -> list[Any]:
    """Probe values for a type annotation."""
    origin = get_origin(ann)
    if origin is Annotated:
        return candidates(get_args(ann)[0], nums, strs)
    if origin in (Union, types.UnionType):
        out: list[Any] = []
        for a in get_args(ann):
            out += [None] if a is type(None) else candidates(a, nums, strs)
        return out
    if origin is Literal:
        return list(get_args(ann))
    if not isinstance(ann, type):
        return []
    if issubclass(ann, enum.Enum):
        return list(ann)
    if ann is bool:
        return [True, False]
    if ann is int:
        out = [0, -1, 1, 10**9]
        for c in nums:
            out += [int(c) - 1, int(c), int(c) + 1]
        return out
    if ann is float:
        out = [0.0, -1.0, 1e9]
        for c in nums:
            out += [c - 1, c - 0.01, float(c), c + 0.01, c + 1]
        return out
    if ann is str:
        out = ["", " ", "zzz-unmatched-✓"]
        for s in strs:
            out += [s, s.lower(), s.upper()]
        return out
    return []


def _walk(
    model: type[BaseModel], data: dict[str, Any], path: tuple[str, ...], nums: list[float], strs: list[str]
) -> Iterator[tuple[tuple[str, ...], Any]]:
    for name, field in model.model_fields.items():
        inner = unwrap(field.annotation)
        if is_model(inner):
            if isinstance(data.get(name), dict):
                yield from _walk(inner, data[name], path + (name,), nums, strs)
            if type(None) in get_args(field.annotation):
                yield path + (name,), None
        else:
            for value in candidates(field.annotation, nums, strs):
                yield path + (name,), value


def _set(data: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    out = copy.deepcopy(data)
    cur = out
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value
    return out


def variants(
    model: type[BaseModel],
    base_events: list[BaseModel],
    nums: list[float],
    strs: list[str],
    limit: int = 300,
) -> list[BaseModel]:
    seen: set[str] = {e.model_dump_json() for e in base_events}
    out: list[BaseModel] = []
    for base in base_events:
        data = base.model_dump()
        for path, value in _walk(model, data, (), nums, strs):
            try:
                ev = model.model_validate(_set(data, path, value))
            except ValidationError:
                continue
            sig = ev.model_dump_json()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(ev)
            if len(out) >= limit:
                return out
    return out
