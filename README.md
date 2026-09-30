# JIT Agents

From user intent to real-time decisions. The model writes the code, then gets out of the way.

A customer says what they want in plain language. An agent turns that into tests, the customer confirms
the tests, and the agent writes a small Python function that passes them. The function is injected into
your event stream and runs on every event in microseconds. The model is involved once per rule, never
once per event.

## Quick start

```bash
uv sync --all-extras
uv run pytest
uv run python examples/quickstart.py
```

```python
from typing import Literal
from pydantic import BaseModel
from jit_agents import Runtime, Ollama, SqliteStore

class ShipmentEvent(BaseModel):
    customer_id: str
    order_id: str
    warehouse: Literal["Dallas", "Austin", "Denver"]
    hours_late: float = 0

rt = Runtime(
    model=Ollama("qwen3.8:27b-mlx"),
    store=SqliteStore("rules.db"),
    key=lambda event: event.customer_id,
)
rt.register_event(ShipmentEvent)

@rt.tool
def notify_sms(to: str, message: str) -> None:
    """Send an SMS to a person."""
    ...

@rt.invariant
def at_most_one_sms_per_event(event, calls):
    return sum(c.tool == "notify_sms" for c in calls) <= 1

# Conversation time: the model is involved
proposal = await rt.propose("acme", "Text Murphy if a Dallas shipment is over 2 hours late")
rule = await rt.build(proposal)     # generate, validate, run the gate
rt.inject(rule)                     # saved by id, live from the next event

# Stream time: no model involved
rt.dispatch(event)
```

## Models

| Provider | Use | Needs |
|---|---|---|
| `Ollama("qwen3.8:27b-mlx")` | Local, free | Ollama running |
| `Claude()` | Claude Opus 5.5 by default | `uv sync --extra claude` and `ANTHROPIC_API_KEY` in the environment |
| `OpenAICompatible(...)` | OpenAI, Azure Foundry, Databricks, and compatible servers | `uv sync --extra openai` and that host's key |
| `CallableModel(fn)` | Tests and other providers | A function returning a dict |

```python
from jit_agents import Claude

rt = Runtime(model=Claude(), ...)                                  # claude-opus-5-5
rt = Runtime(model=Claude("claude-sonnet-5-5", effort="low"), ...)  # cheaper and faster
```

### OpenAI-format hosts

```python
from jit_agents import OpenAICompatible

OpenAICompatible("gpt-5-mini")                                          # OpenAI, reads OPENAI_API_KEY
OpenAICompatible.azure("my-resource", "my-deployment")                  # reads AZURE_OPENAI_API_KEY
OpenAICompatible.databricks("adb-123.azuredatabricks.net", "my-endpoint")   # reads DATABRICKS_TOKEN
OpenAICompatible("llama3", base_url="http://localhost:11434/v1", api_key="ollama")
```

The model must support structured outputs. There is no fallback to free-form text.

Hosts accept different amounts of JSON Schema, so the provider has two dialects.

| Dialect | For | What it does |
|---|---|---|
| `openai` | OpenAI, Azure Foundry | Sends the full schema in strict mode |
| `simple` | Databricks | Removes `anyOf` and `$ref`, and stays under the host's key limit |

In the `simple` dialect, parts of a schema the host cannot express are sent as JSON held in a string
and decoded afterwards. Those parts are not enforced by the host, so the library checks them itself
and asks the model to correct any that are wrong.

To check that a provider works, run the quickstart against it:

```bash
JIT_PROVIDER=databricks uv run python examples/quickstart.py
```

`Claude` uses structured outputs, so replies always follow the schema. It also turns on server-side
fallbacks, which re-run a request on another model if a safety classifier declines it. Pass
`fallbacks=False` to turn that off.

The model only runs when a rule is proposed or built. Switching providers never affects rules that
are already live.

## How it works

| Step | What happens |
|---|---|
| `register_event`, `tool`, `invariant` | You describe your events, the actions a rule may take, and the policy no rule may break. |
| `propose` | The model turns intent into a summary and test cases, or asks one clarifying question. |
| `build` | The model writes `def handle(event, tools)`. Failures from the gate are fed back, up to `max_attempts`. |
| `revise` | If no rule can satisfy the tests, the model corrects the tests using the gate's feedback. |
| `inject` | The rule is saved and swapped in for its key. The previous version is kept for `rollback`. |
| `dispatch` | A dictionary lookup and a function call. |

A rule never executes a tool. It records calls on a proxy, which makes every rule a pure function from
an event to a list of tool calls. The runtime executes the calls afterwards. Use `dispatch(event, execute=False)`
to get the calls and run them yourself, and `adispatch` for async tools.

## Lookups

Actions change the world. Lookups read it and return a value the rule can use.

```python
@rt.lookup
def average_invoice_amount(customer_id: str, vendor: str | None = None, last: int = 10) -> float | None:
    """Average amount of this customer's most recent invoices."""
    return db.execute("SELECT AVG(amount) FROM (SELECT amount FROM invoices ... LIMIT ?)", ...).fetchone()[0]
```

```python
def handle(event, tools):
    if event.vendor == "Globex":
        avg = tools.average_invoice_amount(customer_id=event.customer_id, vendor="Globex", last=10)
        if avg is not None and event.amount > avg * 1.10:
            tools.notify_email(to="Murphy", subject="Invoice Alert", body=f"Invoice {event.invoice_id} is 10% over average")
```

- Lookups run during evaluation, so they are in the hot path. A database query costs far more than the rule itself.
- In the gate, lookups never run. Each test states what the lookup returns, and the gate varies those values.
- Lookups must be synchronous and must declare a return type.

## The gate

Nothing is injected until it passes all three layers.

| Layer | Defined by | Checks |
|---|---|---|
| Behaviour tests | The customer | This event produces exactly these calls |
| Built-in checks | The library | Never raises, deterministic, valid tool arguments, call cap, event not modified, latency budget |
| Invariants | You | Any `(event, calls) -> bool` policy |

Built-in checks and invariants run over the test events plus events synthesised from the Pydantic model:
every `Literal` and `Enum` option, `None` for optionals, and numbers just below, at, and above each
constant in the rule. Invariants are also enforced at stream time, and a violation blocks execution.

## The rule language

Generated code is checked on the syntax tree before it runs.

- One function, `def handle(event, tools)`. No imports, loops, comprehensions, lambdas, `try`, or nested functions.
- No name or attribute that starts with an underscore.
- `event.<field>` must be a field of the Pydantic model, including nested models.
- `tools.<name>` must be a registered tool.
- A handful of builtins and string methods.

With no loops and no user-defined calls, every rule terminates in time linear in its size.

## Persistence

Rules are stored as source, intent, and tests, under an id. Functions are never stored.

- **Lazy loading.** Nothing loads at startup. The first event for a key pulls its rule from the store. Use `warm()` to preload.
- **Fingerprints.** Each rule records a hash of the event schema, tools, and invariants it was built against. On load, a matching fingerprint skips the gate. A changed one re-runs it.
- **Stale rules.** A rule that no longer passes is marked `stale` and not injected. `rebuild(rule_id)` regenerates it from the stored intent.
- **Keys are opaque.** One rule per key. Use a string, or a tuple such as `("acme", "ShipmentEvent")`.

- **Cleaning up.** `delete(rule_id)` removes one old version and `prune(key, keep=2)` removes all but the newest few. Both are permanent, and neither will delete the live version.

Implement `Store` (`save`, `get`, `list`, `keys`, `delete`) to use another database.

## Security model

Read this before using it with untrusted users.

- Rules run in your process. The validator and the gate are the containment, not an operating system sandbox.
- Your registered tools are the trust boundary. A rule can call any tool you register, with any arguments that pass its type hints.
- Large allocations built from event data are not prevented. For hostile input, run dispatch in a worker with memory limits.
- This is a prototype and has not had a security review.

## Example site

```bash
uv run --extra example examples/webapp/app.py
```

Type a request into the update box and watch the rule get rebuilt and swapped into the running stream.
Switch between Claude and Ollama from the page header, or start with `JIT_PROVIDER=claude`.

## Not yet supported

- More than one rule per key
- Multiple processes sharing a store with live invalidation
- JavaScript

## License

Apache License 2.0. See [LICENSE](LICENSE).

If you distribute this software or a product that includes it, keep the [NOTICE](NOTICE) file with it.
That is how the project is credited.

Copyright 2026 Murphy Crosby.
