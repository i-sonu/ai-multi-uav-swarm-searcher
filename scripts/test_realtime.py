"""Real-Time Inference Tester for Fine-Tuned YOLOv8s SAR Aerial Detector (22.5 MB best.pt).

Usage:
    # 1. Test on 10 random aerial drone images from VisDrone:
    python -m scripts.test_realtime --source aerial --num 10 --random

    # 2. Test on a specific image or custom folder:
    python -m scripts.test_realtime --source path/to/image.jpg
    python -m scripts.test_realtime --source path/to/folder/

    # 3. Test on live laptop webcam:
    python -m scripts.test_realtime --source webcam
"""

import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Real-time tester for 22.5MB YOLOv8s SAR model")
    parser.add_argument(
        "--source",
        default="aerial",
        help="'aerial' for VisDrone images, 'webcam' (or '0') for live camera, or a path to an image/folder",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of aerial images to test (default: 5)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick random images from the dataset instead of sequential",
    )
    parser.add_argument(
        "--weights",
        default="data/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default 0.40)")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference resolution (default 800)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' not found!")
        sys.exit(1)

    size_mb = weights_path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"Loading YOLOv8 model: {weights_path} ({size_mb:.1f} MB)")
    model = YOLO(str(weights_path))
    print(f"Target classes: {model.names}")
    print(f"Confidence threshold: {args.conf} | Resolution: {args.imgsz}x{args.imgsz}")
    print("=" * 70)

    # Mode 1: Live Webcam
    if args.source in ("webcam", "0"):
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)

        print("\n--> Starting LIVE WEBCAM feed. Press 'q' in the camera window to stop.")
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            annotated_frame = results[0].plot()
            cv2.putText(
                annotated_frame,
                f"YOLOv8s SAR (22MB) | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Real-Time YOLOv8s Aerial Target Detector (22MB best.pt)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Live webcam test ended.")

    # Mode 2: Sample Aerial VisDrone Images
    elif args.source == "aerial":
        img_pattern = "data/VisDrone2019-DET-val/images/*.jpg"
        imgs = sorted(glob.glob(img_pattern))
        if not imgs:
            print(f"No VisDrone images found matching '{img_pattern}'.")
            sys.exit(1)

        if args.random:
            selected_imgs = random.sample(imgs, min(args.num, len(imgs)))
        else:
            selected_imgs = imgs[: args.num]

        out_name = f"aerial_test_{int(time.time())}"
        print(f"\n--> Running inference on {len(selected_imgs)} aerial drone images...")
        results = model.predict(
            source=selected_imgs,
            conf=args.conf,
            imgsz=args.imgsz,
            save=True,
            project="runs/detect",
            name=out_name,
        )

        print("\n" + "-" * 70)
        print("DETECTION BREAKDOWN PER FRAME:")
        print("-" * 70)
        for i, res in enumerate(results):
            boxes = res.boxes
            counts = {}
            for cls_idx in boxes.cls.cpu().numpy():
                cname = model.names.get(int(cls_idx), str(cls_idx))
                counts[cname] = counts.get(cname, 0) + 1

            summary = ", ".join(f"{v} {k}(s)" for k, v in counts.items()) if counts else "No targets detected"
            print(f"  Frame {i+1} [{Path(res.path).name}]: {len(boxes)} targets -> {summary}")

        out_dir = Path("runs/detect") / out_name
        print("-" * 70)
        print(f"--> All annotated images saved to:")
        print(f"    {out_dir.resolve()}")

    # Mode 3: Custom Image, Video, or Folder
    else:
        src_path = Path(args.source)
        if not src_path.exists():
            print(f"Error: Source '{args.source}' not found!")
            sys.exit(1)

        out_name = f"custom_test_{int(time.time())}"
        print(f"\n--> Running inference on custom source: {src_path}")
        results = model.predict(
            source=str(src_path),
            conf=args.conf,
            imgsz=args.imgsz,
            save=True,
            project="runs/detect",
            name=out_name,
        )
        out_dir = Path("runs/detect") / out_name
        print(f"\n--> Annotated output saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
