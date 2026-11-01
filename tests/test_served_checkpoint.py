"""The promoted fitted planner is served from a committed, source-pinned checkpoint.

Any change under ``src/catan_ai`` unbinds it. Rebind with
``uv run python scripts/bind_parameter_checkpoint.py --write``, which refuses if
the change alters how the promoted bot plays.
"""
import json

import pytest

from catan_ai.engine import BACKEND
from catan_ai.parameter_planner import SERVED_CHECKPOINT, compatibility_digest, load_parameters

pytestmark = pytest.mark.skipif(BACKEND != 'python', reason='the served binding is for the pure-Python engine')

# Weights evaluated in the v13 external rerun and the sealed Tactical-control
# panels (checkpoint b1785ccb, same weights as 77c5ebfc). Promotion is of these
# numbers; changing them needs a new evaluation, not a rebind.
EVALUATED_WEIGHTS = [16.954330814946953, 0.6537987321930112, 0.641167594927603, 1.5499286338795906,
                     0.4445213636185654, 1.4479747239779968, 7.063402747746514, 2.805531684851287]


def test_served_checkpoint_is_bound_to_this_source(monkeypatch):
    monkeypatch.delenv('CATAN_PARAMETER_MODEL', raising=False)
    try:
        weights, _, path = load_parameters()
    except ValueError as exc:
        pytest.fail(f'{exc}: source changed; run scripts/bind_parameter_checkpoint.py --write')
    assert path == SERVED_CHECKPOINT.resolve()
    assert weights == EVALUATED_WEIGHTS


def test_served_checkpoint_replays_the_evaluated_snapshot_exactly(monkeypatch):
    monkeypatch.delenv('CATAN_PARAMETER_MODEL', raising=False)
    data = json.loads(SERVED_CHECKPOINT.read_text())
    assert compatibility_digest(data['compatibility']['plan']) == data['compatibility']['trace_sha256']
