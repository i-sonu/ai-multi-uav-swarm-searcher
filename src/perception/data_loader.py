"""Dataset loading utilities.

Phase 0 stub: enough to satisfy the DA1 submission checklist requirement that
the repository contains dataset-loading code. This lists and loads images from
a directory. Phase 5 will flesh this out into a real loader for the chosen
aerial person-detection dataset (with annotations, splits, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

# Common image extensions we recognise.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return a sorted list of image file paths under ``directory``.

    Args:
        directory: folder to scan.
        recursive: if True, descend into sub-directories.

    Sorted output makes the listing deterministic — important because every
    experiment in this project must be reproducible from a seed.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    pattern = "**/*" if recursive else "*"
    paths = [
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths)


def load_image(path: str | Path):
    """Load a single image as an RGB numpy array (H, W, 3).

    Uses OpenCV for decoding and converts BGR -> RGB so downstream code sees a
    conventional RGB ordering.
    """
    import cv2  # imported lazily so `import src` stays light-weight

    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def iter_images(directory: str | Path, recursive: bool = False) -> Iterator[tuple[Path, "object"]]:
    """Yield ``(path, image_array)`` pairs for every image in ``directory``."""
    for path in list_images(directory, recursive=recursive):
        yield path, load_image(path)
