from types import SimpleNamespace
import pytest
from scripts.twin.restock_manipulation.clearance import linear_tool_sweep, overlaps


def test_linear_sweep_checks_interior_and_rejects_rotating_endpoints():
    tool = SimpleNamespace(envelope_pad_min=(-.1, -.1, -.1),
                           envelope_pad_max=(.1, .1, .1))
    start = (0, 0, 0, 0, 0, 0, 1)
    end = (2, 0, 0, 0, 0, 0, 1)
    obstacle = ((.9, -.05, -.05), (1.1, .05, .05))
    assert not overlaps(linear_tool_sweep(tool, start, start), obstacle)
    assert not overlaps(linear_tool_sweep(tool, end, end), obstacle)
    assert overlaps(linear_tool_sweep(tool, start, end), obstacle)
    with pytest.raises(ValueError, match='rotating sweep'):
        linear_tool_sweep(tool, start, (2, 0, 0, 0, 0, 1, 0))
