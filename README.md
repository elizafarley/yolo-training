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

You can find your organization ID when viewing your organization settings and invites.
The dataset ID can be copied from the `...` menu on your dataset overview page.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `num_epochs` | `200` | Maximum epochs. Training stops early after `patience` epochs without improvement. |
| `patience` | `50` | Early-stopping patience, in epochs. |
| `labels` | all labels in the dataset | Space-separated labels to train on, in single quotes. |
| `base_model` | `yolo11s.pt` | Any Ultralytics detection checkpoint, such as `yolo11n.pt` (faster) or `yolo11m.pt` (more accurate). |
| `imgsz` | `640` | Train and export resolution. Raise it (`960`, `1280`) when objects are small in the frame. Inference cost grows with the square of this value. |

Any other argument is passed to Ultralytics [`train()`](https://docs.ultralytics.com/modes/train/#train-settings)
as is, for example `batch=32`, `cache=true` or augmentation settings. Top-down views (tables, bins, conveyors) usually
benefit from `flipud=0.5,degrees=180`:

```
--args=imgsz=1280,labels="'fastener'",flipud=0.5,degrees=180
```

### Metrics

The script writes `model_metrics.json` to the model output directory. It contains per-epoch losses, precision, recall,
mAP50 and mAP50-95, and the training job's page in the app charts these values. The job logs also print the validation
metrics of the best checkpoint, which is the one that is exported.

## Development

This project is managed using [`uv`](https://docs.astral.sh/uv/) and `make`.

After [installing uv](https://docs.astral.sh/uv/#installation), sync the project dependencies:

```
make install
```

Build and publish a new version of the script:

1. Update the `pyproject.toml` with the new version number.
1. Run the following command with the new version number in place of `<version>`:
   ```
   version=<version> make publish
   ```
   You must be authenticated with the [Viam CLI](https://docs.viam.com/dev/tools/cli/) and have access to the Viam `devrel` organization.
