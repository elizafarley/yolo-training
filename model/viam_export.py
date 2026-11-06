"""Export a trained YOLO detector to the ONNX shape that Viam's ML model tooling reads.

A plain Ultralytics ONNX export takes float input scaled to 0-1 and returns one
[1, 4 + num_classes, num_candidates] tensor with no non-maximum suppression. Viam's
cloud inference (used by model evaluation and auto-predictions) and the mlmodel vision
service with viam:onnx-cpu expect something else:

- Input: one NCHW image tensor with a fixed height and width. Cloud inference sends raw
  0-255 pixel values. The vision service sends raw values for uint8 input, but rescales
  float32 input to -1..1, so the exported model takes uint8 and scales it itself.
- Outputs: three tensors named location [1, N, 4], category [1, N] and score [1, N].
  Boxes are normalized to 0-1 in ymin, xmin, ymax, xmax order, the default box order of
  both the vision service and cloud inference.
- Cloud inference tells category from score by checking whether every value is a whole
  number, and tells it apart badly when N is 0. The model always returns MAX_DET rows,
  the best detections first, so the outputs are never empty. Rows past the real
  detections are low-confidence candidates, which a confidence threshold removes.
"""

import inspect

import onnx
import torch
import torchvision
from torch import nn

MAX_DET = 100
IOU_THRESHOLD = 0.45
# NMS over every candidate is slow on CPU, so keep only the best ones first
PRE_NMS_TOPK = 1000

OUTPUT_SHAPES = {
    "location": [1, MAX_DET, 4],
    "category": [1, MAX_DET],
    "score": [1, MAX_DET],
}


class ViamDetector(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        height, width = image.shape[2], image.shape[3]
        out = self.model(image.float() / 255.0)
        if isinstance(out, (list, tuple)):
            out = out[0]

        # [1, 4 + num_classes, num_candidates] -> [num_candidates, 4 + num_classes]
        pred = out[0].transpose(0, 1)
        scores, classes = pred[:, 4:].max(dim=1)

        k = min(PRE_NMS_TOPK, pred.shape[0])
        scores, idx = scores.topk(k)
        classes = classes[idx]
        cx, cy, w, h = pred[idx, :4].unbind(dim=1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        keep = torchvision.ops.batched_nms(boxes, scores, classes, IOU_THRESHOLD)
        keep = keep[:MAX_DET]

        x1, y1, x2, y2 = boxes[keep].unbind(dim=1)
        location = torch.stack(
            [y1 / height, x1 / width, y2 / height, x2 / width], dim=1
        ).clamp(0.0, 1.0)
        score = scores[keep]
        category = classes[keep].float()

        # NMS on 1000 candidates leaves far more than MAX_DET boxes in practice, but pad
        # anyway so the output length never depends on the image
        location = torch.cat([location, location.new_zeros(MAX_DET, 4)])[:MAX_DET]
        score = torch.cat([score, score.new_zeros(MAX_DET)])[:MAX_DET]
        category = torch.cat([category, category.new_zeros(MAX_DET)])[:MAX_DET]

        return location.unsqueeze(0), category.unsqueeze(0), score.unsqueeze(0)


def check_raw_detection_output(model, imgsz):
    """Raise ValueError unless model returns [1, 4 + num_classes, N] raw candidates.

    End-to-end models such as YOLOv10 and YOLO26 run NMS inside the model and return
    [1, 300, 6] instead, which ViamDetector would misread as boxes without an error.
    """
    if getattr(model.model[-1], "end2end", False):
        raise ValueError(
            "end-to-end YOLO models (such as YOLOv10 and YOLO26) are not supported by "
            "export_format=viam; use a YOLOv8 or YOLO11 base model, or export_format=ultralytics"
        )
    with torch.no_grad():
        out = model(torch.zeros(1, 3, imgsz, imgsz))
    if isinstance(out, (list, tuple)):
        out = out[0]
    expected = 4 + len(model.names)
    if not isinstance(out, torch.Tensor) or out.dim() != 3 or out.shape[1] != expected:
        got = list(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__
        raise ValueError(
            f"expected raw detection output [1, {expected}, N] from the model, "
            f"got {got}; export_format=viam supports YOLOv8 and YOLO11 detection models"
        )


def export_viam_onnx(weights_path, output_path, imgsz=640):
    """Wrap trained YOLO weights in ViamDetector and write the ONNX file to output_path."""
    from ultralytics import YOLO

    model = YOLO(weights_path).model.float()
    detector = ViamDetector(model).eval()
    check_raw_detection_output(model, imgsz)

    # torch 2.9 made the dynamo exporter the default; the TorchScript exporter handles
    # torchvision's NMS without extra setup
    extra = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        extra["dynamo"] = False

    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.uint8)
    with torch.no_grad():
        torch.onnx.export(
            detector,
            dummy,
            output_path,
            input_names=["image"],
            output_names=list(OUTPUT_SHAPES),
            opset_version=17,
            **extra,
        )

    # The exporter can't prove the NMS output length, so it names that dimension
    # symbolically. The padding makes it always MAX_DET, so record that in the model.
    onnx_model = onnx.load(output_path)
    for output in onnx_model.graph.output:
        dims = output.type.tensor_type.shape.dim
        for dim, size in zip(dims, OUTPUT_SHAPES[output.name]):
            dim.ClearField("dim_param")
            dim.dim_value = size
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path)
    return output_path
