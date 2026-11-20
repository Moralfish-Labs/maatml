"""Predictor-owned observations survive validation, metrics and caching."""

import json

import pytest

from maatml.evaluation.harness import run_evaluation
from maatml.evaluation.predictions import predictions_path, read_predictions
from maatml.validation.base import ValidationResult


class ContextPredictor:
    def __init__(self):
        self.state = {"observed": []}
        self.calls = 0

    def predict(self, row):
        self.calls += 1
        self.state["observed"][:] = [row["request"]]
        return row["request"]

    def prediction_context(self, row):
        return self.state


def evaluate(tmp_path, predictor, **kwargs):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "".join(
            json.dumps(
                {
                    "request": value,
                    "prediction_context": {"observed": ["forged"]},
                }
            )
            + "\n"
            for value in ("one", "two")
        )
    )
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    return run_evaluation(
        checkpoint_dir=tmp_path / "checkpoint",
        dataset_dir=tmp_path,
        rows_path=rows,
        out_path=tmp_path / "report.json",
        predictor=predictor,
        schema_path=schema,
        device="cpu",
        cache_predictions=True,
        **kwargs,
    )


def test_context_is_bound_per_row_and_validator_cannot_rewrite_it(tmp_path):
    observed = []

    def validate(text, *, prediction_context, **kwargs):
        assert prediction_context == {"observed": [text]}
        prediction_context["observed"].append("validator mutation")
        result = ValidationResult(raw_output=text, n_layers=1)
        result.passed_layers.add(1)
        return result

    def metrics(items):
        observed.extend(item.prediction_context for item in items)
        return {
            "context_ok": float(
                all(item.prediction_context == {"observed": [item.gen_text]} for item in items)
            )
        }

    predictor = ContextPredictor()
    report = evaluate(tmp_path, predictor, validator=validate, metrics_fn=metrics)
    assert report.metrics["context_ok"] == 1.0
    assert observed == [{"observed": ["one"]}, {"observed": ["two"]}]
    predictor.state["observed"].append("later mutation")
    _, cached = read_predictions(predictions_path(tmp_path / "report.json"))
    assert [item["prediction_context"] for item in cached] == observed
    assert cached[0]["row"]["prediction_context"] == {"observed": ["forged"]}
    assert cached[0]["prediction_context"] != cached[0]["row"]["prediction_context"]


def test_old_validator_signature_and_plain_predictor_remain_compatible(tmp_path):
    def validate(text, *, user_prompt, schema_path):
        assert text == user_prompt
        result = ValidationResult(raw_output=text, n_layers=1)
        result.passed_layers.add(1)
        return result

    evaluate(tmp_path, lambda row: row["request"], validator=validate)
    _, cached = read_predictions(predictions_path(tmp_path / "report.json"))
    assert all("prediction_context" not in item for item in cached)


def test_kwargs_alone_does_not_opt_validator_into_context(tmp_path):
    def validate(text, **kwargs):
        assert "prediction_context" not in kwargs
        return ValidationResult(raw_output=text, n_layers=1)

    evaluate(tmp_path, ContextPredictor(), validator=validate)


@pytest.mark.parametrize("context", [None, [], {"value": float("nan")}, {"value": object()}])
def test_invalid_context_is_refused_instead_of_silently_dropped(tmp_path, context):
    class InvalidContext(ContextPredictor):
        def prediction_context(self, row):
            return context

    with pytest.raises(ValueError, match="prediction_context"):
        evaluate(tmp_path, InvalidContext())
    assert not (tmp_path / "report.json").exists()


def test_batch_context_requires_an_unambiguous_per_row_contract(tmp_path):
    class BatchContext(ContextPredictor):
        def predict_batch(self, rows):
            pytest.fail("must reject before prediction")

    predictor = BatchContext()
    with pytest.raises(ValueError, match="batch_size=1"):
        evaluate(tmp_path, predictor, batch_size=2)
    assert predictor.calls == 0
