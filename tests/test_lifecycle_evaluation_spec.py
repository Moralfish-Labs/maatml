"""The runner's evaluation slice must accept every key the config does.

`EvaluationSpec` forbids extras so a typo is caught before training spends the
compute. That only works if it carries every *known* key: when it fell behind
`config._EVALUATION_KNOWN_KEYS`, a folder declaring `slices` or
`cache_predictions` passed `validate`, passed `evaluate`, and was refused by
`run` — every stage individually, nothing end to end.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from maatml.config import _EVALUATION_KNOWN_KEYS
from maatml.lifecycle import EvaluationSpec


def test_spec_covers_every_known_evaluation_key():
    missing = _EVALUATION_KNOWN_KEYS - set(EvaluationSpec.model_fields)
    assert not missing, (
        f"EvaluationSpec is missing {sorted(missing)}; `run` would refuse a "
        "section that config.py, validate and evaluate all accept"
    )


def test_spec_still_rejects_an_unknown_key():
    """The point of extra="forbid": a misspelt key fails before training runs."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvaluationSpec(validator="x", slcies=["family"])


def test_spec_accepts_a_full_evidence_layer_section():
    spec = EvaluationSpec(
        predictor="p",
        validator="v",
        metrics="m",
        gates={"a_rate": 0.5},
        gates_benchmark="deadbeef",
        slices=["dataset", {"field": "spectrum", "values": ["rgb"]}],
        cache_predictions=True,
        batch_size=32,
        operating_point={"threshold_key": "score_thresh", "objective": "recall"},
    )
    assert spec.cache_predictions is True
    assert spec.slices[0] == "dataset"
