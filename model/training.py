import argparse
import json
import platform
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

# ultralytics depends on the GUI build of opencv-python, whose wheels need X11/GL shared
# libraries that the Viam training image does not have, so `import cv2` fails. Overwrite the
# cv2 package with the headless wheel (same API, no GUI deps) before ultralytics imports it.
# --no-deps and --force-reinstall keep the opencv-python dist metadata that ultralytics requires.
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--force-reinstall",
     "opencv-python-headless>=4.7,<4.13"]
)

from sklearn.model_selection import train_test_split
import torch
from ultralytics import YOLO
import yaml

from model.viam_export import export_viam_onnx


def parse_args(args):
    """Returns dataset file, model output directory, and num_epochs if present. These must be parsed as command line
    arguments and then used as the model input and output, respectively. The number of epochs can be used to optionally override the default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", dest="data_json", type=str)
    parser.add_argument("--model_output_directory", dest="model_dir", type=str)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=200)
    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        required=False,
        help="Space-separated list of labels, MUST be enclosed in single quotes",
        # ex: 'green_square blue_triangle'
    )
    parser.add_argument(
        "--base_model", dest="base_model", type=str, default="yolov8n.pt"
    )
    parser.add_argument("--imgsz", dest="imgsz", type=int, default=640)
    parser.add_argument(
        "--export_format",
        dest="export_format",
        choices=["viam", "ultralytics"],
        default="viam",
        help="viam: an ONNX model that Viam model evaluation and viam:onnx-cpu can run. "
        "ultralytics: the plain Ultralytics export, for viam-labs:vision:yolov8.",
    )
    parsed_args = parser.parse_args(args)
    return (
        parsed_args.data_json,
        parsed_args.model_dir,
        parsed_args.num_epochs,
        parsed_args.labels,
        parsed_args.base_model,
        parsed_args.imgsz,
        parsed_args.export_format,
    )


def create_train_val_split(dataset, val_size=0.2, random_state=42):
    """
    Create training and validation splits from a YOLO dataset collection.

    Args:
        dataset: A zip object of tuples (image_path, class_names, bounding_boxes)
        val_size: Proportion of the dataset to include in the validation split
        random_state: Random seed for reproducibility

    Returns:
        train_data: Training dataset
        val_data: Validation dataset
    """
    # Convert zip object to list for easier handling
    dataset_list = list(dataset)

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(
        dataset_list, test_size=val_size, random_state=random_state
    )

    return train_data, val_data


def create_yolo_yaml_config(output_path, class_map=None):
    """
    Create a YOLO configuration YAML file with the train and validation splits.

    Args:
        train_data: Training data from create_train_val_split
        val_data: Validation data from create_train_val_split
        output_path: Path to save the YAML file
        class_map: Dictionary mapping class names to indices (optional)
    """
    # If class_map is not provided, create one from the dataset
    if class_map is None:
        # Collect all unique class names
        all_classes = set()
        for _, class_names, _ in train_data + val_data:
            all_classes.update(class_names)

        # Create class name mapping
        class_map = {i: name for i, name in enumerate(sorted(all_classes))}

    # Create inverse mapping for label files
    class_to_idx = {name: idx for idx, name in class_map.items()}

    # Create YAML configuration
    config = {
        "path": os.getcwd(),  # Current directory as root
        "train": "images/train",  # List of training image paths
        "val": "images/val",  # List of validation image paths
        "names": class_map,  # Class name mapping
    }

    # Save YAML file
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"YAML configuration saved to {output_path}")

    return class_to_idx


def setup_dataset_directories(output_dir):
    """
    Create the YOLO dataset directory structure.

    Args:
        output_dir: Base directory for the dataset
    """
    # Create main directories
    train_images_dir = os.path.join(output_dir, "images", "train")
    val_images_dir = os.path.join(output_dir, "images", "val")
    train_labels_dir = os.path.join(output_dir, "labels", "train")
    val_labels_dir = os.path.join(output_dir, "labels", "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    return {
        "train_images": train_images_dir,
        "val_images": val_images_dir,
        "train_labels": train_labels_dir,
        "val_labels": val_labels_dir,
    }


def image_extension(path):
    """Sniff the image format from the file header and return its extension.

    Viam data-capture files are stored under bare IDs with no extension, and
    ultralytics only loads files whose extension is a known image type.
    """
    with open(path, "rb") as f:
        head = f.read(12)
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    return ""


def process_dataset(dataset, subset_type, dirs, class_to_idx):
    """
    Process dataset by copying images and creating label files.

    Args:
        dataset: List of (image_path, class_names, bounding_boxes)
        subset_type: 'train' or 'val'
        dirs: Dictionary with directory paths
        class_to_idx: Dictionary mapping class names to indices
    """
    images_dir = dirs[f"{subset_type}_images"]
    labels_dir = dirs[f"{subset_type}_labels"]

    for img_path, class_names, bboxes in dataset:
        # Get source image filename, adding an extension if the source has none
        img_filename = os.path.basename(img_path)
        if os.path.exists(img_path) and not os.path.splitext(img_filename)[1]:
            img_filename += image_extension(img_path)

        # Destination paths
        dest_img_path = os.path.join(images_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        # Copy image to destination
        if os.path.exists(img_path):
            shutil.copy2(img_path, dest_img_path)
            print(f"Copied {img_path} to {dest_img_path}")
        else:
            print(f"Warning: Source image {img_path} not found")
            continue

        # Write bounding boxes to label file
        with open(label_path, "w") as f:
            for class_name, bbox in zip(class_names, bboxes):
                # Get class index
                class_idx = class_to_idx[class_name]

                # YOLO format: class_idx x_center y_center width height
                bbox_str = f"{class_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                f.write(bbox_str)


def parse_filenames_and_bboxes_from_json(
    filename: str,
    all_labels: List[str] | None,
) -> Tuple[List[str], List[List[str]], List[List[List[float]]]]:
    """Load and parse JSON file to return image filenames and corresponding labels with bboxes.
        The JSON file contains lines, where each line has the key "image_path" and "bounding_box_annotations".
    Args:
        filename: JSONLines file containing filenames and bboxes
        all_labels: list of all N_LABELS
    """
    image_filenames: List[str] = []
    bbox_labels: List[List[str]] = []
    bbox_coords: List[List[List[float]]] = []

    with open(filename, "rb") as f:
        for line in f:
            json_line = json.loads(line)
            image_filenames.append(json_line["image_path"])
            annotations = json_line["bounding_box_annotations"]
            labels: List[str] = []
            coords: List[List[float]] = []
            use_label = all_labels is None
            for annotation in annotations:
                if all_labels:
                    use_label = annotation["annotation_label"] in all_labels

                if use_label:
                    labels.append(annotation["annotation_label"])
                    # Convert from [y_min, x_min, y_max, x_max] to [x_center, y_center, width, height]
                    x_min = annotation["x_min_normalized"]
                    y_min = annotation["y_min_normalized"]
                    x_max = annotation["x_max_normalized"]
                    y_max = annotation["y_max_normalized"]

                    width = x_max - x_min
                    height = y_max - y_min
                    x_center = x_min + width / 2
                    y_center = y_min + height / 2

                    coords.append([
                        x_center,
                        y_center,
                        width,
                        height,
                    ])
            bbox_labels.append(labels)
            bbox_coords.append(coords)
    return image_filenames, bbox_labels, bbox_coords


if __name__ == "__main__":
    device: Optional[str] = "cpu"
    platform_str = platform.platform().lower()

    (
        DATA_JSON,
        MODEL_DIR,
        num_epochs,
        labels,
        BASE_MODEL,
        imgsz,
        export_format,
    ) = parse_args(sys.argv[1:])
    patience = int(num_epochs / 2)

    if "macos" in platform_str and "arm64" in platform_str:
        device = "mps"
        patience = 30  # local testing cutoff

    if torch.cuda.is_available():
        device = str(torch.cuda.current_device())

    LABELS = [label for label in labels.strip("'").split()] if labels else None

    (
        image_filenames,
        bbox_labels,
        bbox_coords,
    ) = parse_filenames_and_bboxes_from_json(
        filename=DATA_JSON,
        all_labels=LABELS,
    )

    if LABELS is None:
        # Create a list of unique labels from bbox_labels
        unique_labels = set()
        for label_list in bbox_labels:
            unique_labels.update(label_list)
        LABELS = sorted(list(unique_labels))

    dataset = zip(image_filenames, bbox_labels, bbox_coords)

    train_data, val_data = create_train_val_split(dataset)
    class_map = {index: value for index, value in enumerate(LABELS)}

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    output_path = "dataset.yaml"

    class_to_idx = create_yolo_yaml_config(
        output_path,
        class_map,
    )

    dirs = setup_dataset_directories(os.getcwd())

    process_dataset(train_data, "train", dirs, class_to_idx)
    process_dataset(val_data, "val", dirs, class_to_idx)

    model = YOLO(BASE_MODEL)
    results = model.train(
        task="detect",
        data=output_path,
        epochs=num_epochs,
        imgsz=imgsz,
        device=device,
        patience=patience,
    )

    if export_format == "viam":
        export_path = export_viam_onnx(
            model.trainer.best,
            os.path.splitext(str(model.trainer.best))[0] + ".onnx",
            imgsz=imgsz,
        )
    else:
        export_path = model.export(format="onnx", device=device)

    # # Move the exported model to the specified output directory
    if os.path.exists(export_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        destination = os.path.join(MODEL_DIR, os.path.basename(export_path))
        shutil.copy2(export_path, destination)
        print(f"Model exported and copied to {destination}")

        # Create labels.txt file in the MODEL_DIR
        labels_file_path = os.path.join(MODEL_DIR, "labels.txt")
        with open(labels_file_path, "w") as f:
            for label in LABELS:
                f.write(f"{label}\n")
        print(f"Labels file created at {labels_file_path}")
