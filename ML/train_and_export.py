#!/usr/bin/env python3
"""
train_and_export.py
===================
Train a YOLOv11s Mahjong tile detection model and export it to CoreML
(.mlpackage) for use in our iOS app with Vision framework (VNCoreMLModel).

Based on: https://github.com/smilee3998/mahjong_detection

Prerequisites
-------------
    pip install ultralytics roboflow python-dotenv coremltools

Usage
-----
    # 1. Set your Roboflow API key
    echo "API_KEY=your_key_here" > .env

    # 2. Train (downloads data automatically on first run)
    python train_and_export.py --train

    # 3. Export best weights to CoreML
    python train_and_export.py --export

    # 4. Do both
    python train_and_export.py --train --export

The exported .mlpackage will be at:
    MahjongTileDetector.mlpackage

Copy it into the Xcode project's "Mahjong Helper" group.
"""

import argparse
from pathlib import Path

# ── Roboflow dataset version used by smilee3998 ──
ROBOFLOW_VERSION = 18
DATA_ROOT = Path(f"Mahjong_detect-{ROBOFLOW_VERSION}")

# ── YOLO training settings ──
BASE_MODEL = "yolo11s.pt"    # YOLOv11 small — good speed/accuracy balance
EPOCHS = 30
IMGSZ = 640                  # 640 is standard for mobile; 1024 for max accuracy
RUNS_DIR = Path("runs/detect")

# ── CoreML export ──
COREML_OUTPUT_NAME = "MahjongTileDetector"


def download_data():
    """Download the Mahjong detection dataset from Roboflow."""
    from roboflow import Roboflow
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError(
            "Set your Roboflow API key:\n"
            "  echo 'API_KEY=your_key_here' > .env"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("mahjongdetect-6yabv").project("mahjong_detect")
    project.version(ROBOFLOW_VERSION).download("yolov11")
    print(f"✅  Dataset downloaded to {DATA_ROOT}")


def train():
    """Train YOLOv11s on the Mahjong dataset."""
    from ultralytics import YOLO

    if not DATA_ROOT.exists():
        print("📦  Dataset not found, downloading from Roboflow...")
        download_data()

    data_yaml = DATA_ROOT / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}")

    model = YOLO(BASE_MODEL)
    print(f"🚀  Training {BASE_MODEL} for {EPOCHS} epochs at imgsz={IMGSZ}...")

    model.train(
        data=str(data_yaml.absolute()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=-1,       # auto batch size (60% GPU memory)
        device="mps",   # Apple Silicon GPU; change to 0 for NVIDIA
        project=str(RUNS_DIR.parent),
        name=str(RUNS_DIR.name),
    )
    print("✅  Training complete! Weights at", get_best_weights())


def get_best_weights() -> Path:
    """Find the best.pt weights from the latest training run."""
    # Try best.pt first, then last.pt
    candidates = sorted(RUNS_DIR.parent.glob("detect*/weights/best.pt"))
    if candidates:
        return candidates[-1]
    candidates = sorted(RUNS_DIR.parent.glob("detect*/weights/last.pt"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        "No trained weights found. Run with --train first."
    )


def export_to_coreml(weights_path: Path = None):
    """
    Export YOLO .pt weights → CoreML .mlpackage

    Ultralytics' built-in export uses coremltools under the hood and
    produces an .mlpackage that works with Vision framework (VNCoreMLModel).
    """
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = get_best_weights()

    print(f"📱  Exporting {weights_path} → CoreML .mlpackage ...")

    model = YOLO(str(weights_path))

    # Export to CoreML
    # - nms=True  → bakes NMS into the model (important for VNCoreMLRequest)
    # - imgsz=640 → matches training size
    model.export(
        format="coreml",
        nms=True,
        imgsz=IMGSZ,
    )

    # Ultralytics saves the .mlpackage next to the .pt file
    exported = weights_path.with_suffix(".mlpackage")
    if exported.exists():
        # Move to project root with our desired name
        target = Path(f"{COREML_OUTPUT_NAME}.mlpackage")
        if target.exists():
            import shutil
            shutil.rmtree(target)
        exported.rename(target)
        print(f"✅  CoreML model saved: {target}")
        print()
        print("Next steps:")
        print(f"  1. Drag '{target}' into Xcode → 'Mahjong Helper' group")
        print("  2. Build & run — TileDetector.swift will pick it up automatically")
    else:
        print(f"⚠️  Expected export at {exported} but not found.")
        print("   Check the ultralytics output above for the actual path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & export Mahjong tile detection model for iOS"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--export", action="store_true", help="Export to CoreML")
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to .pt weights (for --export without --train)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    args = parser.parse_args()

    if args.epochs != EPOCHS:
        EPOCHS = args.epochs

    if not args.train and not args.export:
        parser.print_help()
        print("\nExample:")
        print("  python train_and_export.py --train --export")
        exit(0)

    if args.train:
        train()

    if args.export:
        w = Path(args.weights) if args.weights else None
        export_to_coreml(w)
