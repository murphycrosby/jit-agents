import pytest

from jit_agents.validator import RuleError, load, validate

from .conftest import GOOD, ShipmentEvent

TOOLS = frozenset({"notify_sms", "route"})


def check(source: str) -> None:
    validate(source, ShipmentEvent, TOOLS)


def test_good_rule_loads():
    assert callable(load(GOOD, ShipmentEvent, TOOLS))


def test_nested_model_fields_and_typed_locals():
    check(
        "def handle(event, tools):\n"
        "    v = event.vendor\n"
        "    if v is not None and v.trusted and event.vendor.name.lower() == 'acme':\n"
        "        tools.route(queue='standard')\n"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "import os\ndef handle(event, tools):\n    pass",
        "def handle(event, tools):\n    __import__('os')",
        "def handle(event, tools):\n    x = ().__class__.__base__.__subclasses__()",
        "def handle(event, tools):\n    x = event.__class__",
        "def handle(event, tools):\n    x = event.model_dump()",
        "def handle(event, tools):\n    x = event.model_fields",
        "def handle(event, tools):\n    while True:\n        pass",
        "def handle(event, tools):\n    for x in event:\n        pass",
        "def handle(event, tools):\n    x = [k for k in event]",
        "def handle(event, tools):\n    f = lambda: 1",
        "def handle(event, tools):\n    x = getattr(event, 'order_id')",
        "def handle(event, tools):\n    x = open('/etc/passwd')",
        "def handle(event, tools):\n    x = eval('1')",
        "def handle(event, tools):\n    tools._calls.append(1)",
        "def handle(event, tools):\n    tools.launch_missiles()",
        "def handle(event, tools):\n    t = tools\n    t.notify_sms(to='a', message='b')",
        "def handle(event, tools):\n    event.order_id = 'x'",
        "def handle(event, tools):\n    x = 2 ** 100000",
        "def handle(event, tools):\n    x = 'a' * 1000000000",
        "def handle(event, tools):\n    x = f'{event.order_id:>999999999}'",
        "def handle(event, tools):\n    x = 10 ** 3",
        "def handle(event, tools):\n    x = 99999999999999999999",
        "def handle(event, tools):\n    try:\n        pass\n    except Exception:\n        pass",
        "def handle(event):\n    pass",
        "def other(event, tools):\n    pass",
        "def handle(event, tools):\n    def inner():\n        pass",
        "x = 1\ndef handle(event, tools):\n    pass",
        "def handle(event, tools):\n    event = None",
        "def handle(event, tools):\n    x = undefined_name",
    ],
)
def test_escapes_are_rejected(bad):
    with pytest.raises(RuleError):
        check(bad)


def test_misspelled_field_is_rejected_statically():
    with pytest.raises(RuleError, match="has no field 'hours_lat'"):
        check("def handle(event, tools):\n    x = event.hours_lat")


def test_misspelled_nested_field_is_rejected_statically():
    with pytest.raises(RuleError, match="has no field 'nam'"):
        check("def handle(event, tools):\n    x = event.vendor.nam")
