"""Static validation and loading of generated rules.

The rule language is a small subset of Python, enforced on the AST before any
code runs. There are no imports, loops, comprehensions, lambdas, nested
functions or underscore names, and only a handful of builtins. Because there
are no loops and no user-defined calls, every rule terminates in time linear
in its own size.

Field access is checked against the Pydantic event model, so a misspelled or
invented field is rejected here instead of failing on live traffic.
"""
from __future__ import annotations

import ast
import enum
import re
import types
from typing import Annotated, Any, Callable, Union, get_args, get_origin

from pydantic import BaseModel

RULE_LANGUAGE_VERSION = "1"

MAX_SOURCE_CHARS = 8000
MAX_NODES = 1500
MAX_INT = 10**12
MAX_STR = 500

SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "min": min, "max": max, "abs": abs, "round": round,
}
ALLOWED_METHODS = frozenset(
    {"lower", "upper", "strip", "startswith", "endswith", "isdigit", "replace", "get"}
)
ENUM_ATTRS = frozenset({"value", "name"})
_FORMAT_SPEC = re.compile(r"^,?(\.\d{1,2})?[dfeg%s]?$")

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Pass, ast.Expr,
    ast.If, ast.IfExp, ast.Assign,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.Attribute, ast.Subscript,
    ast.Call, ast.keyword, ast.JoinedStr, ast.FormattedValue,
    ast.Dict, ast.List, ast.Tuple,
)
_SEQUENCE_LITERALS = (ast.JoinedStr, ast.List, ast.Tuple)


class RuleError(Exception):
    """The source is outside the rule language."""


def unwrap(ann: Any) -> Any:
    """Optional[X] -> X, Annotated[X, ...] -> X. Ambiguous unions -> None."""
    origin = get_origin(ann)
    if origin is Annotated:
        return unwrap(get_args(ann)[0])
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(ann) if a is not type(None)]
        return unwrap(args[0]) if len(args) == 1 else None
    return ann


def is_model(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def _is_enum(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, enum.Enum)


class _Validator(ast.NodeVisitor):
    def __init__(
        self, event_model: type[BaseModel], tool_names: frozenset[str], returns: dict[str, Any] | None = None
    ) -> None:
        self.event_model = event_model
        self.tool_names = tool_names
        self.returns = returns or {}
        self.errors: list[str] = []
        self.local_types: dict[str, Any] = {}
        self.in_function = False

    def _err(self, node: ast.AST, msg: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODES):
            self._err(node, f"{type(node).__name__} is not allowed")
            return
        super().generic_visit(node)

    # -- type resolution ----------------------------------------------------
    def _type_of(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return self.event_model if node.id == "event" else self.local_types.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self._type_of(node.value)
            if is_model(base) and node.attr in base.model_fields:
                return unwrap(base.model_fields[node.attr].annotation)
            return None
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tools":
                return unwrap(self.returns.get(f.attr))
            return None
        if isinstance(node, ast.Subscript):
            base = self._type_of(node.value)
            origin, args = get_origin(base), get_args(base)
            if origin in (list, tuple, set, frozenset) and args:
                return unwrap(args[0])
            if origin is dict and len(args) == 2:
                return unwrap(args[1])
        return None

    # -- structure ----------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        defs = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        if len(defs) != 1 or defs[0].name != "handle":
            self._err(node, "module must define exactly one function named handle")
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                continue
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            self._err(n, "only the handle function may appear at module level")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.in_function:
            self._err(node, "nested functions are not allowed")
            return
        a = node.args
        if (
            [x.arg for x in a.args] != ["event", "tools"]
            or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or a.defaults or a.kw_defaults
        ):
            self._err(node, "signature must be exactly: def handle(event, tools)")
        if node.decorator_list:
            self._err(node, "decorators are not allowed")
        self.in_function = True
        for stmt in node.body:
            self.visit(stmt)
        self.in_function = False

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.local_types[node.targets[0].id] = self._type_of(node.value)

    # -- names and attributes -----------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if name.startswith("_"):
            self._err(node, f"name {name!r} may not start with an underscore")
        elif name == "tools":
            self._err(node, "tools may only be used as tools.<name>(...)")
        elif isinstance(node.ctx, ast.Store):
            if name == "event" or name in SAFE_BUILTINS:
                self._err(node, f"may not reassign {name!r}")
            self.local_types.setdefault(name, None)
        elif isinstance(node.ctx, ast.Load):
            if name != "event" and name not in SAFE_BUILTINS and name not in self.local_types:
                self._err(node, f"unknown name {name!r}")
        else:
            self._err(node, "del is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._err(node, "attribute assignment is not allowed")
        if node.attr.startswith("_"):
            self._err(node, f"attribute {node.attr!r} may not start with an underscore")
            return
        if isinstance(node.value, ast.Name) and node.value.id == "tools":
            if node.attr not in self.tool_names:
                self._err(node, f"unknown tool tools.{node.attr}; available: {sorted(self.tool_names)}")
            return
        base = self._type_of(node.value)
        if is_model(base):
            if node.attr not in base.model_fields:
                self._err(node, f"{base.__name__} has no field {node.attr!r}; fields: {sorted(base.model_fields)}")
        elif _is_enum(base):
            if node.attr not in ENUM_ATTRS:
                self._err(node, f"only .value and .name are allowed on {base.__name__}")
        elif node.attr not in ALLOWED_METHODS:
            self._err(node, f"method .{node.attr} is not allowed; allowed: {sorted(ALLOWED_METHODS)}")
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._err(node, "item assignment is not allowed")
        if isinstance(node.slice, ast.Slice):
            self._err(node, "slicing is not allowed")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name):
            if f.id not in SAFE_BUILTINS:
                self._err(node, f"calling {f.id!r} is not allowed")
        elif isinstance(f, ast.Attribute):
            self.visit(f)
        else:
            self._err(node, "only tool methods, value methods and safe builtins may be called")
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            if kw.arg is None:
                self._err(node, "**kwargs expansion is not allowed")
            self.visit(kw.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            for side in (node.left, node.right):
                if isinstance(side, _SEQUENCE_LITERALS) or (
                    isinstance(side, ast.Constant) and isinstance(side.value, str)
                ):
                    self._err(node, "multiplying a string or sequence is not allowed")
        self.generic_visit(node)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        spec = node.format_spec
        if spec is not None:
            parts = spec.values if isinstance(spec, ast.JoinedStr) else [spec]
            if not all(isinstance(p, ast.Constant) and isinstance(p.value, str) for p in parts):
                self._err(node, "dynamic format specs are not allowed")
            elif not _FORMAT_SPEC.match("".join(p.value for p in parts)):  # type: ignore[attr-defined]
                self._err(node, "format spec not allowed; use forms like .1f, .2f, d, or ,.2f")
        self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> None:
        v = node.value
        if not isinstance(v, (str, int, float, bool, type(None))):
            self._err(node, f"constant of type {type(v).__name__} is not allowed")
        elif isinstance(v, str) and len(v) > MAX_STR:
            self._err(node, f"string constant longer than {MAX_STR} characters")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) > MAX_INT:
            self._err(node, f"numeric constant larger than {MAX_INT}")


def validate(
    source: str, event_model: type[BaseModel], tool_names: frozenset[str], returns: dict[str, Any] | None = None
) -> ast.Module:
    """Parse and check ``source``. Raises RuleError if it is outside the rule language."""
    if len(source) > MAX_SOURCE_CHARS:
        raise RuleError(f"source exceeds {MAX_SOURCE_CHARS} characters")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise RuleError(f"syntax error: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise RuleError(f"rule has more than {MAX_NODES} AST nodes")
    v = _Validator(event_model, tool_names, returns)
    v.visit(tree)
    if v.errors:
        raise RuleError("; ".join(v.errors))
    return tree


def load(
    source: str, event_model: type[BaseModel], tool_names: frozenset[str], returns: dict[str, Any] | None = None
) -> Callable[[Any, Any], Any]:
    """Validate ``source`` and return its ``handle`` function."""
    tree = validate(source, event_model, tool_names, returns)
    ns: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS)}
    exec(compile(tree, "<jit-rule>", "exec"), ns)  # noqa: S102 - validated subset
    return ns["handle"]


def collect_constants(source: str) -> tuple[list[float], list[str]]:
    """Numeric and string constants in the rule, used to probe its boundaries."""
    nums: list[float] = []
    strs: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nums, strs
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and v not in nums:
                nums.append(v)
            elif isinstance(v, str) and v and len(v) <= 60 and v not in strs:
                strs.append(v)
    return nums[:20], strs[:20]
