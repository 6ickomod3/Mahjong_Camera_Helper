# Mahjong Helper

A real-time Mahjong tile detection iOS app built with SwiftUI, AVFoundation, and a YOLOv11 CoreML model. Point your iPhone camera at Mahjong tiles and see them identified instantly with Chinese names, confidence scores, and bounding boxes overlaid on the live preview.

## Features

- **Real-time detection** — identifies Mahjong tiles at ~6 FPS using on-device ML inference
- **38 tile classes** — all standard tiles: 1–9 萬/索/筒, 4 winds, 3 dragons, and red fives (0m/0p/0s)
- **Multiple same-tile detection** — correctly detects 4 copies of the same tile (e.g. four 一萬)
- **Laid-down tile support** — dual-pass inference (0° + 90° rotation) detects sideways tiles
- **Bounding box overlay** — draws labeled boxes on detected tiles with Chinese name + confidence %
- **Pinch-to-zoom** — 0.5×–10× zoom using the full multi-lens virtual device range
- **Dynamic rotation** — camera preview and detection boxes adapt to portrait/landscape orientations
- **Earthy UI theme** — warm terracotta accent, cream/charcoal backgrounds, translucent HUD elements

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  ContentView.swift                                      │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │ CameraPreviewView│  │ Detection Box Overlay        │  │
│  │ (AVCaptureVideo  │  │ (GeometryReader + aspectFill │  │
│  │  PreviewLayer)   │  │  coordinate mapping)         │  │
│  └────────┬─────────┘  └──────────────┬───────────────┘  │
│           │                           │                  │
│  ┌────────▼─────────┐       ┌─────────▼──────────┐      │
│  │ CameraManager    │──────▶│ TileDetector        │      │
│  │ (AVCaptureSession │ frame │ (CoreML + custom    │      │
│  │  + orientation)   │──────▶│  NMS + smoothing)   │      │
│  └──────────────────┘       └────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## The ML Model: From GitHub to iOS

This section explains the full pipeline of how we found, converted, and integrated a pre-trained YOLO model into the iOS app.

### Step 1: Finding a Pre-Trained Model

We evaluated several GitHub repos for Mahjong tile detection and selected **[nikmomo/Mahjong-YOLO](https://github.com/nikmomo/Mahjong-YOLO)** because it:
- Provides pre-trained YOLOv11 weights (`.pt` files) ready to use
- Covers all 38 standard Mahjong tile classes
- Reports high accuracy: 0.943 precision, 0.927 recall, 0.956 mAP@50
- Uses MIT license
- Offers both nano (5.2 MB) and medium (38 MB) model sizes

We downloaded `yolo11n_best.pt` (the nano model — best for real-time mobile inference).

### Step 2: Converting PyTorch → CoreML

iOS cannot run PyTorch `.pt` files directly. Apple's **CoreML** framework is the standard way to run ML models on iOS. The conversion chain is:

```
PyTorch (.pt) ──ultralytics──▶ CoreML (.mlpackage) ──Xcode──▶ Compiled (.mlmodelc)
```

#### Setting up the Python environment

```bash
cd ML/
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install ultralytics coremltools
```

#### The export command

```python
from ultralytics import YOLO

model = YOLO("yolo11n_best.pt")
model.export(format="coreml", nms=False, imgsz=640)
```

**Critical detail: `nms=False`**

We export **without** the built-in NMS (Non-Maximum Suppression) pipeline. Why?

- Ultralytics' built-in CoreML NMS uses `pickTop.perClass = True`, meaning it keeps **at most 1 detection per class**
- In Mahjong, you can have **4 copies of the same tile** on the table
- By exporting without NMS, we get the raw tensor output and apply our own NMS in Swift that allows multiple detections of the same class

#### What the export produces

The exported `.mlpackage` has:
- **Input:** `"image"` — a 640×640 RGB image (BGRA pixel buffer)
- **Output:** `"var_1224"` — a `MultiArray` with shape `(1, 42, 8400)`
  - 42 = 4 (bounding box: cx, cy, w, h) + 38 (class confidence scores)
  - 8400 = number of anchor predictions across all scales

### Step 3: Adding the Model to Xcode

1. Renamed the exported file to `MahjongTileDetector.mlpackage`
2. Placed it inside `/Mahjong Helper/MahjongTileDetector.mlpackage/`
3. Xcode 16 uses **File System Synchronized Groups** — any file placed in the app target folder is automatically discovered and included in the build
4. At build time, Xcode compiles `.mlpackage` → `.mlmodelc` (an optimized binary format) and bundles it in the app

No manual PBXFileReference entries needed in `project.pbxproj`.

### Step 4: Loading the Model in Swift

In `TileDetector.swift`, we load the model using the raw `MLModel` API (not `VNCoreMLModel`):

```swift
guard let modelURL = Bundle.main.url(forResource: "MahjongTileDetector",
                                      withExtension: "mlmodelc") else {
    // Fall back to OCR
    return
}
let config = MLModelConfiguration()
config.computeUnits = .all  // Use CPU + GPU + Neural Engine
mlModel = try MLModel(contentsOf: modelURL, configuration: config)
```

We use `MLModel` directly instead of wrapping it in `VNCoreMLModel` + `VNCoreMLRequest` because:
- We need access to the raw output tensor to apply custom NMS
- Vision's `VNCoreMLRequest` applies its own post-processing that doesn't suit our needs

### Step 5: Running Inference

Each camera frame goes through this pipeline:

```
Camera Frame (CVPixelBuffer, e.g. 2232×1674, BGRA)
    │
    ▼
Resize to 640×640 (CIImage stretch via CIContext)
    │
    ▼
MLModel.prediction(from: MLDictionaryFeatureProvider)
    │
    ▼
Raw Output Tensor: shape (1, 42, 8400)
    │
    ▼
Parse: for each of 8400 anchors, find best class score
    │  ─ filter by confidence threshold (0.20)
    │  ─ convert cx,cy,w,h from pixels (0..640) to normalized (0..1)
    ▼
Custom NMS (IoU threshold 0.45, class-agnostic)
    │  ─ allows multiple detections of same tile type
    ▼
Temporal Smoothing (5-frame window, position-based slot keys)
    │  ─ bounding box EMA smoothing (α=0.4)
    │  ─ requires 2+ appearances in 5 frames to display
    ▼
Published to SwiftUI: [MahjongTile] with bounding boxes
```

#### Why we resize manually

`MLDictionaryFeatureProvider` (the raw MLModel API) does **not** auto-resize images, unlike `VNCoreMLRequest`. The model expects exactly 640×640 pixels, so we must resize the camera frame ourselves:

```swift
private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, to targetSize: CGSize) -> CVPixelBuffer? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let scaleX = targetSize.width / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let scaleY = targetSize.height / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
    let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    // Render to a new 640×640 BGRA pixel buffer
    var output: CVPixelBuffer?
    CVPixelBufferCreate(nil, 640, 640, kCVPixelFormatType_32BGRA, nil, &output)
    ciContext.render(scaled, to: output!)
    return output
}
```

#### Why we force BGRA pixel format

The camera defaults to YCbCr 420f (efficient for video) but CoreML image inputs only accept 32BGRA or 32ARGB. We force the format on the camera output:

```swift
videoOutput.videoSettings = [
    String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA
]
```

### Step 6: Parsing the Raw YOLO Output

The output tensor has shape `(1, 42, 8400)`. Each of the 8400 columns is one anchor prediction:

```
Row 0: cx  (center x, in pixels 0..640)
Row 1: cy  (center y, in pixels 0..640)
Row 2: w   (width in pixels)
Row 3: h   (height in pixels)
Row 4..41: confidence score for each of the 38 classes
```

For each anchor, we:
1. Find the class with the highest score
2. Check if it exceeds `confidenceThreshold` (0.20)
3. Convert pixel coordinates to normalized 0..1 by dividing by 640
4. Create a `MahjongTile` with the bounding box

### Step 7: Custom NMS (Non-Maximum Suppression)

Standard NMS removes overlapping detections. Our version is **class-agnostic** — it suppresses based purely on spatial overlap (IoU > 0.45), regardless of class. This means:

- Two overlapping boxes for the same tile → the lower-confidence one is removed ✓
- Two boxes for the same tile type at different positions → both are kept ✓ (e.g. four 一萬)

### Step 8: Mapping YOLO Labels to Display Names

The model outputs class indices 0–37. Each maps to a label string like `"1m"`, `"5p"`, `"7z"`:

| Suffix | Suit | Chinese | Example |
|--------|------|---------|---------|
| `m` | 萬 (Character/Man) | 一萬…九萬 | `5m` → 五萬 |
| `p` | 筒 (Dot/Pin) | 一筒…九筒 | `3p` → 三筒 |
| `s` | 索 (Bamboo/Sou) | 一索…九索 | `7s` → 七索 |
| `z` | Honor tiles | 東南西北中發白 | `1z` → 東 (East) |
| `0m/0p/0s` | Red fives | 赤五萬/筒/索 | `0m` → 赤五萬 |

The mapping is in `MahjongTile.fromYOLOLabel()`.

## File Structure

```
Mahjong Helper/
├── README.md                   ← You are here
├── style.md                    ← Design guide (colors, fonts, layout)
├── ML/                         ← Model training & export
│   ├── train_and_export.py     ← Training script (Roboflow dataset + YOLO)
│   ├── yolo11n_best.pt         ← Pre-trained nano model (5.2 MB)
│   ├── yolo11m_best.pt         ← Pre-trained medium model (38 MB)
│   └── yolo11n_best.mlpackage/ ← Exported CoreML model (no NMS)
├── Mahjong Helper/             ← iOS app source
│   ├── Mahjong_HelperApp.swift ← App entry point
│   ├── ContentView.swift       ← Main view: camera + overlay + HUD
│   ├── CameraManager.swift     ← AVCaptureSession, zoom, orientation
│   ├── CameraPreviewView.swift ← UIViewRepresentable for preview layer
│   ├── TileDetector.swift      ← CoreML inference, NMS, smoothing
│   ├── MahjongTile.swift       ← Tile data model + label mapping
│   ├── Theme.swift             ← App-wide color theme
│   ├── MahjongTileDetector.mlpackage/ ← CoreML model (bundled in app)
│   └── Assets.xcassets/        ← App icons, accent color
├── Mahjong Helper.xcodeproj/   ← Xcode project
├── Mahjong HelperTests/        ← Unit tests
└── Mahjong HelperUITests/      ← UI tests
```

## Key Technical Decisions

| Decision | Why |
|----------|-----|
| **YOLOv11 nano** over medium | 5.2 MB vs 38 MB, fast enough for real-time on iPhone |
| **Export without NMS** | Built-in NMS limits to 1 detection per class; Mahjong needs 4 |
| **MLModel directly** instead of VNCoreMLRequest | Need raw tensor access for custom NMS |
| **Manual 640×640 resize** | MLDictionaryFeatureProvider doesn't auto-resize (unlike VNCoreMLRequest) |
| **Force BGRA pixel format** | Camera defaults to YCbCr 420f; CoreML needs 32BGRA |
| **Dual-pass inference** (0° + 90°) | Detects tiles that are laid down sideways |
| **Temporal smoothing** (5-frame window) | Reduces flickering; requires 2+ appearances to show |
| **Position-based slot keys** (10×10 grid) | Tracks multiple instances of same tile at different positions |
| **Class-agnostic NMS** | Suppresses by spatial overlap only, not by class |
| **Bounding box EMA** (α=0.4) | Smooth box movement without lag |

## Requirements

- iOS 18.5+
- Xcode 16.4+
- iPhone with camera (tested on iPhone Pro with triple camera)
- Python 3.9+ (only for model export — not needed to build the app)

## Getting Started

### Build & Run the App

1. Open `Mahjong Helper.xcodeproj` in Xcode
2. Select your iPhone as the run destination
3. Build and run (⌘R)
4. Grant camera permission when prompted
5. Point the camera at Mahjong tiles

The `MahjongTileDetector.mlpackage` is already included — no model conversion needed.

### Re-export the Model (Optional)

If you want to use a different model or re-export:

```bash
cd ML/
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install ultralytics coremltools

python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11n_best.pt')
model.export(format='coreml', nms=False, imgsz=640)
"

# Copy to the app target
cp -r yolo11n_best.mlpackage ../Mahjong\ Helper/MahjongTileDetector.mlpackage
```

## Credits

- **ML Model:** [nikmomo/Mahjong-YOLO](https://github.com/nikmomo/Mahjong-YOLO) — YOLOv11 Mahjong tile detection (MIT License)
- **Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO training & CoreML export
- **Dataset:** Mahjong tile detection dataset via Roboflow
