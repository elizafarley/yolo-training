# YOLO object detection training

A Python project for building, testing, and deploying a [YOLOv8](https://docs.ultralytics.com/models/yolov8/) object detection model exported as [ONNX](https://onnx.ai/) to the Viam registry.
From the registry, the script can be used in the Viam custom training scripts flow for training ML models in the Viam cloud.

The script has been published to the Viam Registry as [`yolo-onnx-training`](https://app.viam.com/ml-training/devrel/yolo-onnx-training).

If you'd like to learn how train and run your own YOLO model, check out [this codelab](https://codelabs.viam.com/guide/yolo-training/index.html?index=..%2F..index#0)!

## Usage

**In the app:**

Follow the steps listed in the [Viam docs](https://docs.viam.com/data-ai/ai/train/#submit-a-training-job).
Make sure to select the latest version of the `yolo-onnx-training` script.

**From the command line:**

In order to submit this script with custom arguments, you must use the Viam CLI. One such example is included below:
```
viam train submit custom from-registry \
--dataset-id=<DATASET-ID> \ 
--org-id=<ORG-ID> \
--model-name=yolo-detection \
--model-type=object_detection \
--script-name=yolo-onnx-training \
--args=num_epochs=100,labels="'green_square blue_star'"
```
Be sure to note that labels is surrounded with single quotes then enclosed with double quotes to ensure it is submitted properly. If you are running the script from a previous version or from the website, you will not be able to use custom arguments.

The script accepts these arguments:

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `num_epochs` | `200` | Number of training epochs. |
| `labels` | All labels in the dataset | Space-separated labels to train on, in single quotes. |
| `base_model` | `yolov8n.pt` | Ultralytics weights to fine-tune. With `export_format=viam`, a YOLOv8 or YOLO11 detection model. |
| `imgsz` | `640` | Training and model input size, in pixels. Must be a multiple of 32. |
| `export_format` | `viam` | `viam` or `ultralytics`. See below. |

### Export formats

With `export_format=viam` (the default), the ONNX model works with Viam's model evaluation and auto-predictions, and on a machine with the `viam:mlmodel:onnx-cpu` ML model service and the `mlmodel` vision service, with no tensor remapping:

- Input: `uint8` RGB image, `[1, 3, imgsz, imgsz]`, raw 0-255 pixel values.
- Outputs: `location` `[1, 100, 4]` (normalized ymin, xmin, ymax, xmax), `category` `[1, 100]` and `score` `[1, 100]`, after non-maximum suppression.

The model always returns its 100 best detections, most confident first, so set a confidence threshold, such as `default_minimum_confidence` on the vision service.

With `export_format=ultralytics`, the script writes the plain Ultralytics ONNX export, for use with the [`hipsterbrown:mlmodel:yolo-onnx`](https://github.com/HipsterBrown/viam-yolo-onnx) module, as in the codelab above.

`export_format=viam` supports YOLOv8 and YOLO11 base models. End-to-end models such as YOLOv10 and YOLO26 run non-maximum suppression inside the model, and the script stops with an error before exporting them; use `export_format=ultralytics` for those.

You can find your organization ID when viewing your organization settings and invites.
The dataset ID can be copied from the `...` menu on your dataset overview page.

## Development

This project is managed using [`uv`](https://docs.astral.sh/uv/) and `make`.

After [installing uv](https://docs.astral.sh/uv/#installation), sync the project dependencies:

```
make install
```

Run the tests, which export `yolov8n.pt` and check the ONNX model's inputs and outputs:

```
make test
```

Build and publish a new version of the script:

1. Update the `pyproject.toml` with the new version number.
1. Run the following command with the new version number in place of `<version>`:
   ```
   version=<version> make publish
   ```
   You must be authenticated with the [Viam CLI](https://docs.viam.com/dev/tools/cli/) and have access to the Viam `devrel` organization.
