"""Aerial Person Detection Dataset Ingestion & Preprocessing (Task 5.1).

Provides dataset loading utilities, VisDrone annotation parser, PyTorch Dataset
class (VisDroneDataset), PyTorch DataLoader builder, and YOLO dataset config generator.
"""

from __future__ import annotations

import os
import importlib
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

# Dynamic imports to prevent static IDE linter warnings on host environments
torch = None
Dataset = object
DataLoader = None
cv2 = None

try:
    torch = importlib.import_module("torch")
    _data = importlib.import_module("torch.utils.data")
    DataLoader = getattr(_data, "DataLoader", None)
    Dataset = getattr(_data, "Dataset", object)
except Exception:
    torch = None
    Dataset = object
    DataLoader = None

try:
    cv2 = importlib.import_module("cv2")
except Exception:
    cv2 = None


# VisDrone class category mapping
VISDRONE_CLASSES = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# Target categories of interest for SAR (Person & Vehicle)
SAR_CLASSES = {1: "person", 2: "person", 4: "vehicle", 5: "vehicle", 6: "vehicle", 9: "vehicle"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of image file paths under directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = [p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def load_image(path: str | Path) -> np.ndarray:
    """Load a single image as an RGB numpy array (H, W, 3)."""
    path = Path(path)
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required to load images.")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_visdrone_annotation(anno_path: str | Path) -> list[dict]:
    """Parse a VisDrone annotation text file.

    VisDrone annotation line format:
    <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
    """
    anno_path = Path(anno_path)
    annotations = []
    if not anno_path.exists():
        return annotations

    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            left = float(parts[0])
            top = float(parts[1])
            width = float(parts[2])
            height = float(parts[3])
            score = float(parts[4])
            cat_id = int(parts[5])

            if cat_id == 0 or score == 0:
                continue

            annotations.append({
                "bbox": [left, top, width, height],
                "category_id": cat_id,
                "category_name": VISDRONE_CLASSES.get(cat_id, "unknown"),
                "score": score,
            })

    return annotations


class VisDroneDataset(Dataset):
    """PyTorch Dataset for aerial drone imagery (VisDrone / SAR target detection)."""

    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        if (self.data_dir / "images").exists():
            self.img_dir = self.data_dir / "images"
            self.anno_dir = self.data_dir / "annotations"
        else:
            self.img_dir = self.data_dir
            self.anno_dir = self.data_dir.parent / "annotations"

        self.image_paths = list_images(self.img_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[object, dict]:
        img_path = self.image_paths[idx]
        img = load_image(img_path)
        h, w, _ = img.shape

        anno_filename = img_path.stem + ".txt"
        anno_path = self.anno_dir / anno_filename
        raw_annos = parse_visdrone_annotation(anno_path)

        boxes = []
        labels = []
        for a in raw_annos:
            x, y, bw, bh = a["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(a["category_id"])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        if torch is not None:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            target = {
                "boxes": torch.from_numpy(boxes),
                "labels": torch.from_numpy(labels),
                "image_id": torch.tensor([idx]),
                "orig_size": (h, w),
            }
            return img_tensor, target

        return img, {"boxes": boxes, "labels": labels, "orig_size": (h, w)}


def get_visdrone_dataloader(
    data_dir: str | Path,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """Build a PyTorch DataLoader for the VisDrone aerial dataset."""
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to build a DataLoader.")

    dataset = VisDroneDataset(data_dir)

    def collate_fn(batch):
        return tuple(zip(*batch))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def create_visdrone_yaml(data_dir: str | Path, output_yaml: str | Path = "data/visdrone.yaml") -> Path:
    """Generate a standard Ultralytics YOLO dataset configuration file."""
    data_dir = Path(data_dir).resolve()
    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    content = f"""# VisDrone Aerial Target Dataset Config for YOLOv8
path: {data_dir}
train: images
val: images

names:
  0: ignored
  1: pedestrian
  2: people
  3: bicycle
  4: car
  5: van
  6: truck
  7: tricycle
  8: awning-tricycle
  9: bus
  10: motor
"""

    with open(output_yaml, "w") as f:
        f.write(content)

    return output_yaml


if __name__ == "__main__":
    print("Testing data_loader.py on downloaded VisDrone dataset...")
    test_dir = Path("data/VisDrone2019-DET-val")
    if test_dir.exists():
        ds = VisDroneDataset(test_dir)
        print(f"  Dataset loaded successfully! Total images: {len(ds)}")
        sample_img, sample_target = ds[0]
        print(f"  Sample 0 shape: {sample_img.shape}, targets count: {len(sample_target['labels'])}")
        yaml_path = create_visdrone_yaml(test_dir)
        print(f"  YOLO dataset config created at: {yaml_path}")
    else:
        print(f"  Directory {test_dir} not found. Run dataset download first.")
