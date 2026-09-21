"""Script to download and extract VisDrone aerial dataset into data/ folder."""

import os
import importlib
from pathlib import Path

# Dynamic import to satisfy host IDE linter when running inside venv
try:
    _utils = importlib.import_module("ultralytics.data.utils")
    download = getattr(_utils, "download", None)
except Exception:
    download = None


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
    print(f"Downloading VisDrone aerial dataset from {url} into data/...")

    if download is not None:
        download([url], dir=data_dir)
        print("VisDrone dataset download and extraction complete!")
    else:
        print("Ultralytics package not loaded. Run inside the project virtual environment (venv).")


if __name__ == "__main__":
    main()
