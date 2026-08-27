# YOLOv8 Training & Perception Guide

This guide explains how to train/fine-tune the YOLOv8 object detector on the VisDrone aerial dataset using Google Colab, upload the trained weights to GitHub Releases, and set them up locally for your swarm simulation.

---

## 1. Fine-Tuning YOLOv8 on Google Colab

Because training a deep neural network is computationally intensive, you should run this process on a GPU-enabled Google Colab notebook.

### Step-by-Step Setup:
1. Open **[colab.google.com](https://colab.google.com)** and create a **New Notebook**.
2. Change the runtime to use a GPU:
   * Go to **Runtime** > **Change runtime type**.
   * Under Hardware accelerator, select **T4 GPU** (free tier) and click **Save**.
3. Copy the entire code block below, paste it into a single code cell, and click the **Play** button:

```python
# 1. Install required packages
!pip install ultralytics pillow tqdm

import glob
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

# 2. Download the VisDrone validation set (548 images)
print("Downloading VisDrone dataset...")
if not os.path.exists("VisDrone2019-DET-val.zip"):
    !wget https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
!unzip -q -o VisDrone2019-DET-val.zip -d data/

# 3. Convert VisDrone annotations to YOLO normalized format
print("Converting VisDrone annotations to YOLO format...")
labels_dir = Path("data/VisDrone2019-DET-val/labels")
labels_dir.mkdir(exist_ok=True, parents=True)

img_paths = sorted(glob.glob("data/VisDrone2019-DET-val/images/*.jpg"))
for img_path in tqdm(img_paths):
    # Get image dimensions for coordinate normalization
    with Image.open(img_path) as img:
        w, h = img.size
    
    stem = Path(img_path).stem
    anno_path = Path("data/VisDrone2019-DET-val/annotations") / f"{stem}.txt"
    out_anno_path = labels_dir / f"{stem}.txt"
    
    yolo_lines = []
    if anno_path.exists():
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                
                left = float(parts[0])
                top = float(parts[1])
                width = float(parts[2])
                height = float(parts[3])
                score = float(parts[4])
                cat_id = int(parts[5])
                
                # Filter out background or ignored regions
                if score == 0 or cat_id == 0:
                    continue
                
                # Convert pixel bounding box to normalized YOLO format
                x_center = (left + width / 2.0) / w
                y_center = (top + height / 2.0) / h
                w_norm = width / w
                h_norm = height / h
                
                # Clip coordinates to [0.0, 1.0] range
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                yolo_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                
    with open(out_anno_path, "w") as f:
        f.write("\n".join(yolo_lines))

print(f"Successfully converted annotations for {len(img_paths)} images.")

# 4. Write dataset config YAML file
visdrone_yaml = """
path: /content/data/VisDrone2019-DET-val
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
with open("visdrone.yaml", "w") as f:
    f.write(visdrone_yaml)

# 5. Load pre-trained weights and run fine-tuning (15 epochs)
print("Starting YOLOv8 training on GPU...")
model = YOLO("yolov8n.pt")

results = model.train(
    data="visdrone.yaml", 
    epochs=15, 
    imgsz=640, 
    device=0,           # CUDA GPU
    project="visdrone_project",
    name="yolov8n_visdrone"
)

# 6. Automatically trigger download of best.pt weights
print("Training complete! Sourcing weights file...")
from google.colab import files
import glob
weights = sorted(glob.glob("runs/detect/visdrone_project/*/weights/best.pt"))
if weights:
    print(f"Downloading weights file: {weights[-1]}")
    files.download(weights[-1])
else:
    print("Error: Could not locate best.pt weights.")
```

4. Once the training completes (approx. 5 minutes), your web browser will prompt you to download the **`best.pt`** file. Save it locally.

---

## 2. Setting Up Pre-trained Weights via GitHub Releases

To share the weights with your team members (e.g. friends) and allow them to run the target searcher simulation:

1. **Upload to GitHub Releases:**
   * Go to your repository on GitHub.
   * On the right-hand panel, click **Releases** ➔ **Draft a new release**.
   * Choose a tag version (e.g., `v1.0.0`) and enter a title.
   * Drag and drop the **`best.pt`** weights file into the attachment box.
   * Click **Publish release**.

2. **Fetching Weights locally:**
   * Once uploaded, anyone collaborating on the project can download the weights directly.
   * Make sure the file is placed at the following relative path inside your repository directory:
     📂 **`data/weights/best.pt`**

---

## 3. Running Demos & Verification

After placing the weights at `data/weights/best.pt`, you can run the target registration visualizer demo:

```bash
# Sourcing the virtual environment
source venv/bin/activate

# Running the live multi-agent visual target search demo
python -m scripts.demo_phase5 --map office --n-agents 2 --n-targets 10
```
