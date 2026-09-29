"""Rule storage. Source and intent are persisted; functions never are."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

from jit_agents.models import Rule


@runtime_checkable
class Store(Protocol):
    def save(self, rule: Rule) -> None: ...
    def get(self, rule_id: str) -> Rule | None: ...
    def list(self, key: str) -> list[Rule]:
        """All versions for a key, newest version first."""
        ...
    def keys(self) -> list[str]: ...
    def delete(self, rule_id: str) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def save(self, rule: Rule) -> None:
        self._rules[rule.id] = rule.model_copy(deep=True)

    def get(self, rule_id: str) -> Rule | None:
        rule = self._rules.get(rule_id)
        return rule.model_copy(deep=True) if rule else None

    def list(self, key: str) -> list[Rule]:
        rules = [r.model_copy(deep=True) for r in self._rules.values() if r.key == key]
        return sorted(rules, key=lambda r: r.version, reverse=True)

    def keys(self) -> list[str]:
        return sorted({r.key for r in self._rules.values()})

    def delete(self, rule_id: str) -> None:
        self._rules.pop(rule_id, None)


class SqliteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rules ("
                "id TEXT PRIMARY KEY, key TEXT NOT NULL, version INTEGER NOT NULL, "
                "status TEXT NOT NULL, body TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS rules_key ON rules (key, version)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, rule: Rule) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO rules (id, key, version, status, body) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET key=excluded.key, version=excluded.version, "
                "status=excluded.status, body=excluded.body",
                (rule.id, rule.key, rule.version, rule.status, rule.model_dump_json()),
            )

    def get(self, rule_id: str) -> Rule | None:
        with self._conn() as conn:
            row = conn.execute("SELECT body FROM rules WHERE id=?", (rule_id,)).fetchone()
        return Rule.model_validate_json(row[0]) if row else None

    def list(self, key: str) -> list[Rule]:
        with self._conn() as conn:
            rows = conn.execute("SELECT body FROM rules WHERE key=? ORDER BY version DESC", (key,)).fetchall()
        return [Rule.model_validate_json(r[0]) for r in rows]

    def keys(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT DISTINCT key FROM rules ORDER BY key").fetchall()
        return [r[0] for r in rows]

    def delete(self, rule_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
