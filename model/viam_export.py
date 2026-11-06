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
  number, and tells it apart badly when N is 0. The model always returns max_det rows,
  the best detections first, so the outputs are never empty. Rows past the real
  detections are low-confidence candidates, which a confidence threshold removes.
"""

import inspect

import torch
import torchvision
from torch import nn


class ViamDetector(nn.Module):
    def __init__(self, model, max_det=100, iou_threshold=0.45, pre_nms_topk=1000):
        super().__init__()
        self.model = model
        self.max_det = max_det
        self.iou_threshold = iou_threshold
        self.pre_nms_topk = pre_nms_topk

    def forward(self, image):
        height, width = image.shape[2], image.shape[3]
        out = self.model(image.float() / 255.0)
        if isinstance(out, (list, tuple)):
            out = out[0]

        # [1, 4 + num_classes, num_candidates] -> [num_candidates, 4 + num_classes]
        pred = out[0].transpose(0, 1)
        scores, classes = pred[:, 4:].max(dim=1)

        # NMS over every candidate is slow on CPU, so keep only the best ones first
        k = min(self.pre_nms_topk, pred.shape[0])
        scores, idx = scores.topk(k)
        classes = classes[idx]
        cx, cy, w, h = pred[idx, :4].unbind(dim=1)
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        keep = torchvision.ops.batched_nms(boxes, scores, classes, self.iou_threshold)
        keep = keep[: self.max_det]

        x1, y1, x2, y2 = boxes[keep].unbind(dim=1)
        location = torch.stack(
            [y1 / height, x1 / width, y2 / height, x2 / width], dim=1
        ).clamp(0.0, 1.0)
        score = scores[keep]
        category = classes[keep].float()

        # NMS on 1000 candidates leaves far more than max_det boxes in practice, but pad
        # anyway so the output length never depends on the image
        location = torch.cat([location, location.new_zeros(self.max_det, 4)])[: self.max_det]
        score = torch.cat([score, score.new_zeros(self.max_det)])[: self.max_det]
        category = torch.cat([category, category.new_zeros(self.max_det)])[: self.max_det]

        return location.unsqueeze(0), category.unsqueeze(0), score.unsqueeze(0)


def export_viam_onnx(weights_path, output_path, imgsz=640, max_det=100, iou_threshold=0.45):
    """Wrap trained YOLO weights in ViamDetector and write the ONNX file to output_path."""
    from ultralytics import YOLO

    model = YOLO(weights_path).model.float().eval()
    for p in model.parameters():
        p.requires_grad = False
    detector = ViamDetector(model, max_det=max_det, iou_threshold=iou_threshold).eval()

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
            output_names=["location", "category", "score"],
            opset_version=17,
            **extra,
        )
    return output_path
