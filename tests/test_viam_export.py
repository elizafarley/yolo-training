import numpy as np
import onnxruntime
import pytest

from model.viam_export import MAX_DET, export_viam_onnx


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "yolov8n.onnx"
    export_viam_onnx("yolov8n.pt", str(path))
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_input(session):
    (image,) = session.get_inputs()
    assert image.name == "image"
    assert image.type == "tensor(uint8)"
    assert image.shape == [1, 3, 640, 640]


def test_output_names_and_static_shapes(session):
    outputs = {o.name: (o.type, o.shape) for o in session.get_outputs()}
    assert outputs == {
        "location": ("tensor(float)", [1, MAX_DET, 4]),
        "category": ("tensor(float)", [1, MAX_DET]),
        "score": ("tensor(float)", [1, MAX_DET]),
    }


@pytest.mark.parametrize("fill", ["blank", "noise"])
def test_outputs_are_full_and_sorted(session, fill):
    rng = np.random.default_rng(0)
    image = (
        np.zeros((1, 3, 640, 640), np.uint8)
        if fill == "blank"
        else rng.integers(0, 256, (1, 3, 640, 640), dtype=np.uint8)
    )
    location, category, score = session.run(None, {"image": image})
    assert location.shape == (1, MAX_DET, 4)
    assert category.shape == score.shape == (1, MAX_DET)
    assert np.all(np.diff(score[0]) <= 0)
    assert np.all((location >= 0) & (location <= 1))
    assert np.all(np.mod(category, 1) == 0)


def test_end_to_end_model_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="end-to-end"):
        export_viam_onnx("yolov10n.pt", str(tmp_path / "yolov10n.onnx"))
