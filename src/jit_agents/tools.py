"""Tools: the only surface a generated rule can touch.

There are two kinds.

Actions change the world: send a message, route an event. A rule never
executes an action. It calls ``tools.<name>(...)`` on a recording proxy, which
captures the call, and the runtime executes the calls afterwards.

Lookups read the world and return a value: an average, a balance, a flag.
They run during evaluation, so they sit in the hot path and must be fast. In
the gate they are replaced by the values each test provides, which keeps
every rule a pure function of its event and its lookup values.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from jit_agents.models import ExpectedCall


@dataclass(slots=True)
class Call:
    tool: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    doc: str
    signature: inspect.Signature
    args_model: type[BaseModel]
    is_async: bool
    kind: str = "action"   # "action" or "lookup"
    returns: Any = None    # return annotation, lookups only

    @property
    def returns_optional(self) -> bool:
        return get_origin(self.returns) in (Union, types.UnionType) and type(None) in get_args(self.returns)

    def returns_schema(self) -> dict[str, Any]:
        return TypeAdapter(self.returns).json_schema()

    def json_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        out = {"name": self.name, "kind": self.kind, "description": self.doc, "parameters": schema}
        if self.kind == "lookup":
            out["returns"] = self.returns_schema()
        return out


def spec_from_function(fn: Callable[..., Any], kind: str = "action") -> ToolSpec:
    name = fn.__name__
    if name.startswith("_"):
        raise ValueError(f"tool name {name!r} may not start with an underscore")
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(f"tool {name!r}: *args and **kwargs are not supported")
        default = ... if p.default is p.empty else p.default
        fields[p.name] = (hints.get(p.name, Any), default)
    args_model = create_model(f"{name}_args", __config__=ConfigDict(extra="forbid"), **fields)
    returns = hints.get("return")
    if kind == "lookup":
        if inspect.iscoroutinefunction(fn):
            raise TypeError(f"lookup {name!r} must be synchronous; it runs in the hot path")
        if returns is None or returns is type(None):
            raise TypeError(f"lookup {name!r} needs a return annotation, e.g. -> float | None")
    return ToolSpec(
        kind=kind,
        returns=returns if kind == "lookup" else None,
        name=name,
        fn=fn,
        doc=(inspect.getdoc(fn) or "").strip(),
        signature=sig,
        args_model=args_model,
        is_async=inspect.iscoroutinefunction(fn),
    )


def _make_method(spec: ToolSpec) -> Callable[..., None]:
    name, sig = spec.name, spec.signature

    def method(self: Any, *a: Any, **kw: Any) -> None:
        if a:
            kw = dict(sig.bind(*a, **kw).arguments)
        self._calls.append(Call(name, kw))

    method.__name__ = name
    return method


def _make_lookup(spec: ToolSpec) -> Callable[..., Any]:
    name, sig, fn = spec.name, spec.signature, spec.fn

    def method(self: Any, *a: Any, **kw: Any) -> Any:
        if a:
            kw = dict(sig.bind(*a, **kw).arguments)
        self._reads.append(Call(name, kw))
        stubs = self._stubs
        if stubs is None:
            return fn(**kw)
        if name not in stubs:
            raise LookupError(f"no test provides a value for lookup {name!r}")
        return stubs[name]

    method.__name__ = name
    return method


class ToolSet:
    def __init__(self) -> None:
        self.specs: dict[str, ToolSpec] = {}
        self._proxy_cls: type = self._build_proxy()

    def add(self, fn: Callable[..., Any], kind: str = "action") -> ToolSpec:
        spec = spec_from_function(fn, kind)
        self.specs[spec.name] = spec
        self._proxy_cls = self._build_proxy()
        return spec

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.specs)

    @property
    def actions(self) -> dict[str, ToolSpec]:
        return {n: s for n, s in self.specs.items() if s.kind == "action"}

    @property
    def lookups(self) -> dict[str, ToolSpec]:
        return {n: s for n, s in self.specs.items() if s.kind == "lookup"}

    @property
    def returns(self) -> dict[str, Any]:
        """Return annotation of each lookup, for the validator."""
        return {n: s.returns for n, s in self.lookups.items()}

    def _build_proxy(self) -> type:
        def __init__(self: Any, stubs: dict[str, Any] | None = None) -> None:
            self._calls = []
            self._reads = []
            self._stubs = stubs

        ns: dict[str, Any] = {"__slots__": ("_calls", "_reads", "_stubs"), "__init__": __init__}
        for spec in self.specs.values():
            ns[spec.name] = _make_lookup(spec) if spec.kind == "lookup" else _make_method(spec)
        return type("Tools", (), ns)

    def recorder(self, stubs: dict[str, Any] | None = None) -> Any:
        """A fresh proxy. With ``stubs`` lookups return those values; without, they run for real."""
        return self._proxy_cls(stubs)

    # -- validation and execution -----------------------------------------
    def validate_call(self, call: Call) -> str | None:
        spec = self.specs.get(call.tool)
        if spec is None:
            return f"unknown tool {call.tool!r}"
        try:
            spec.args_model.model_validate(call.args)
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
            return f"tools.{call.tool} got invalid arguments: {errs}"
        return None

    def execute(self, call: Call) -> Any:
        spec = self.specs[call.tool]
        if spec.is_async:
            raise RuntimeError(f"tool {call.tool!r} is async; use Runtime.adispatch")
        return spec.fn(**call.args)

    async def aexecute(self, call: Call) -> Any:
        spec = self.specs[call.tool]
        result = spec.fn(**call.args)
        if inspect.isawaitable(result):
            result = await result
        return result

    # -- schema -------------------------------------------------------------
    def json_schema(self) -> list[dict[str, Any]]:
        return [self.specs[n].json_schema() for n in sorted(self.specs)]

    def schema_text(self, kind: str = "action") -> str:
        lines: list[str] = []
        for name in sorted(n for n, s in self.specs.items() if s.kind == kind):
            spec = self.specs[name]
            if kind == "lookup":
                ret = getattr(spec.returns, "__name__", None) or str(spec.returns).replace("typing.", "")
                args = ", ".join(str(p) for p in spec.signature.parameters.values())
                lines.append(f"- tools.{name}({args}) -> {ret}  # {spec.doc}")
                continue
            params = []
            props = spec.args_model.model_json_schema().get("properties", {})
            for p in spec.signature.parameters:
                info = props.get(p, {})
                typ = info.get("type") or ("|".join(map(str, info["enum"])) if "enum" in info else "any")
                params.append(f"{p}: {typ}")
            lines.append(f"- tools.{name}({', '.join(params)})  # {spec.doc}")
        return "\n".join(lines) or ("(none)" if kind == "lookup" else "(no tools registered)")


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def calls_match(expected: list[ExpectedCall], actual: list[Call]) -> str | None:
    """None when the calls match, otherwise a description of the mismatch.

    Strings compare case-insensitively with whitespace collapsed.
    """
    if len(expected) != len(actual):
        return f"expected {len(expected)} tool call(s), got {len(actual)}: {[c.to_dict() for c in actual]}"
    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp.tool != act.tool:
            return f"call {i}: expected tools.{exp.tool}, got tools.{act.tool}"
        if {k: _norm(v) for k, v in exp.args.items()} != {k: _norm(v) for k, v in act.args.items()}:
            return f"call {i} (tools.{act.tool}): expected args {exp.args}, got {act.args}"
    return None
