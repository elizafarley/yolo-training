import json

import pytest

from model.training import extra_train_args, model_metrics


def test_extra_train_args_types_values():
    assert extra_train_args(["--flipud=0.5", "--degrees", "180", "--cache=true", "--optimizer=AdamW"]) == {
        "flipud": 0.5, "degrees": 180, "cache": True, "optimizer": "AdamW",
    }
    with pytest.raises(ValueError):
        extra_train_args(["--batch"])


def test_model_metrics_is_lists_of_finite_floats(tmp_path):
    csv = tmp_path / "results.csv"
    # older ultralytics pads headers and values with spaces
    csv.write_text(
        "  epoch,  time,train/box_loss,metrics/mAP50(B),metrics/mAP50-95(B),lr/pg0\n"
        "      1,  10.0,  1.5,  0.1,  nan,  0.01\n"
        "      2,  20.0,  1.2,  0.3,  0.2,  0.01\n"
    )
    m = model_metrics(csv)
    assert m == {"train_box_loss": [1.5, 1.2], "mAP50": [0.1, 0.3], "mAP50-95": [0.2]}
    assert "NaN" not in json.dumps(m)
