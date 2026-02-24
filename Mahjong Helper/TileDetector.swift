//
//  TileDetector.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import Vision
import AVFoundation
import UIKit
import CoreML
import Accelerate

/// Processes camera frames to detect Mahjong tiles.
///
/// **Primary method:** Uses a YOLOv11 CoreML model (`MahjongTileDetector.mlpackage`)
/// exported **without** NMS. We parse the raw (1, 42, 8400) tensor and apply our
/// own NMS, which correctly handles multiple tiles of the same class (e.g. four 一萬).
///
/// **Fallback:** If no CoreML model is bundled, falls back to Vision text
/// recognition (OCR) for tiles with Chinese characters.
final class TileDetector: NSObject, ObservableObject {

    // MARK: - Published

    /// All tiles detected in the current frame.
    @Published var detectedTiles: [MahjongTile] = []

    /// The single best-confidence detected tile (convenience).
    @Published var detectedTile: MahjongTile?

    /// Raw recognized text when using OCR fallback.
    @Published var rawText: String = ""

    /// Which detection backend is active.
    @Published var detectionMode: DetectionMode = .initializing

    enum DetectionMode: String {
        case initializing = "Initializing…"
        case coreML       = "CoreML"
        case ocr          = "OCR"
    }

    // MARK: - Detection Hyperparameters

    /// Minimum confidence to accept a detection.
    private let confidenceThreshold: Float = 0.20

    /// IoU threshold for Non-Maximum Suppression.
    /// Lower = more aggressive suppression (fewer overlapping boxes).
    private let iouThreshold: Float = 0.45

    /// Maximum number of detections to return per frame.
    private let maxDetections = 50

    /// Seconds between inference runs.
    private let analysisInterval: TimeInterval = 0.15

    /// Number of recent frames for temporal smoothing.
    private let smoothingWindowSize = 5

    /// Also run inference on the 90°-rotated image to detect laid-down tiles.
    private let detectRotatedTiles = true

    /// The 38 YOLO class labels in index order.
    private let classLabels: [String] = [
        "1m","1p","1s","1z","2m","2p","2s","2z",
        "3m","3p","3s","3z","4m","4p","4s","4z",
        "5m","5p","5s","5z","6m","6p","6s","6z",
        "7m","7p","7s","7z","8m","8p","8s",
        "9m","9p","9s","UNKNOWN","0m","0p","0s"
    ]
    private var numClasses: Int { classLabels.count }  // 38

    // MARK: - Private

    private var lastAnalysisTime: TimeInterval = 0
    /// Each frame stores tiles with a stable positional ID for smoothing
    private var recentFrameResults: [[DetectedInstance]] = []

    /// How quickly the bounding box tracks the new position
    private let boxSmoothingAlpha: CGFloat = 0.4
    /// Smoothed bounding boxes keyed by positional slot ID
    private var smoothedBoxes: [String: CGRect] = [:]

    /// The raw CoreML model (no VNCoreMLModel wrapper needed for raw output)
    private var mlModel: MLModel?

    /// Shared CIContext for pixel buffer operations (rotation)
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Internal detection struct with position-based identity

    private struct DetectedInstance {
        let tile: MahjongTile
        /// Unique-ish key combining tile type + grid position for smoothing
        let slotKey: String
    }

    // MARK: - Init

    override init() {
        super.init()
        loadCoreMLModel()
    }

    // MARK: - CoreML Model Loading

    private func loadCoreMLModel() {
        guard let modelURL = Bundle.main.url(forResource: "MahjongTileDetector",
                                              withExtension: "mlmodelc")
               ?? Bundle.main.url(forResource: "MahjongTileDetector",
                                  withExtension: "mlpackage")
        else {
            print("⚠️ MahjongTileDetector model not found in bundle — using OCR fallback")
            DispatchQueue.main.async { self.detectionMode = .ocr }
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            DispatchQueue.main.async { self.detectionMode = .coreML }
            print("✅ MahjongTileDetector CoreML model loaded (raw output, custom NMS)")
        } catch {
            print("⚠️ Failed to load CoreML model: \(error) — using OCR fallback")
            DispatchQueue.main.async { self.detectionMode = .ocr }
        }
    }

    // MARK: - Public API

    private var frameCount = 0

    func processFrame(_ sampleBuffer: CMSampleBuffer) {
        let now = CACurrentMediaTime()
        guard now - lastAnalysisTime >= analysisInterval else { return }
        lastAnalysisTime = now

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("⚠️ [TileDetector] No pixel buffer in sample buffer")
            return
        }

        frameCount += 1
        if frameCount % 20 == 1 {
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            let fmt = CVPixelBufferGetPixelFormatType(pixelBuffer)
            print("🔍 [TileDetector] Frame #\(frameCount): \(w)×\(h), format=\(fmt), model=\(mlModel != nil ? "loaded" : "nil")")
        }

        if mlModel != nil {
            detectWithCoreML(in: pixelBuffer)
        } else {
            recognizeText(in: pixelBuffer)
        }
    }

    // MARK: - CoreML Detection (Raw Tensor + Custom NMS)

    private func detectWithCoreML(in pixelBuffer: CVPixelBuffer) {
        guard let model = mlModel else { return }

        // --- Pass 1: normal orientation ---
        var frameTiles = runRawInference(model: model, pixelBuffer: pixelBuffer, rotated: false)

        // --- Pass 2: 90° rotation for laid-down tiles ---
        if detectRotatedTiles {
            frameTiles.append(contentsOf: runRawInference(model: model, pixelBuffer: pixelBuffer, rotated: true))
        }

        // Cross-pass NMS: suppress overlapping boxes regardless of which pass found them
        frameTiles = applyNMS(frameTiles, iouThreshold: iouThreshold)

        // Build instances with positional slot keys
        let instances = frameTiles.map { tile -> DetectedInstance in
            let gridKey = slotKey(for: tile)
            return DetectedInstance(tile: tile, slotKey: gridKey)
        }

        // Temporal smoothing
        let smoothed = applyTemporalSmoothing(instances)

        if frameCount % 20 == 1 {
            print("🔍 [TileDetector] raw=\(frameTiles.count) afterNMS=\(instances.count) smoothed=\(smoothed.count)")
        }

        DispatchQueue.main.async {
            self.detectedTiles = smoothed
            self.detectedTile = smoothed.first
            self.rawText = smoothed.map(\.chineseName).joined(separator: " ")
        }
    }

    /// Run a single raw-tensor inference pass, apply confidence filter + NMS.
    private func runRawInference(model: MLModel, pixelBuffer: CVPixelBuffer, rotated: Bool) -> [MahjongTile] {

        // For rotation, create a rotated pixel buffer via CIImage
        let orientedBuffer: CVPixelBuffer
        if rotated {
            guard let rotBuf = createRotatedBuffer(pixelBuffer) else { return [] }
            orientedBuffer = rotBuf
        } else {
            orientedBuffer = pixelBuffer
        }

        // Resize to exactly 640×640 (the model's expected input size).
        // MLDictionaryFeatureProvider does NOT auto-resize unlike VNCoreMLRequest.
        guard let resizedBuffer = resizePixelBuffer(orientedBuffer, to: CGSize(width: 640, height: 640)) else {
            print("⚠️ [TileDetector] Failed to resize pixel buffer to 640×640")
            return []
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: MLDictionaryFeatureProvider(
                dictionary: ["image": MLFeatureValue(pixelBuffer: resizedBuffer)]
            ))
        } catch {
            print("❌ [TileDetector] CoreML prediction failed: \(error)")
            return []
        }

        // Find the multi-array output
        guard let outputArray = findOutputArray(in: prediction) else {
            print("⚠️ Could not find output multi-array in model prediction")
            return []
        }

        // Parse raw YOLO output: shape (1, 42, 8400)
        let tiles = parseYOLOOutput(
            outputArray,
            confidenceThreshold: confidenceThreshold,
            rotated: rotated
        )

        return applyNMS(tiles, iouThreshold: iouThreshold)
    }

    /// Create a 90°-rotated copy of a pixel buffer for detecting laid-down tiles.
    private func createRotatedBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // Rotate 90° CCW and translate so origin stays in positive quadrant
        let rotated = ciImage
            .transformed(by: CGAffineTransform(translationX: 0, y: w)
                .rotated(by: -.pi / 2))

        var output: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(h), Int(w),
                            CVPixelBufferGetPixelFormatType(pixelBuffer), nil, &output)
        guard let outBuf = output else { return nil }
        ciContext.render(rotated, to: outBuf)
        return outBuf
    }

    /// Resize a pixel buffer to the target size using CIImage + CIContext.
    /// The output is a stretch (non-aspect-preserving) resize to exactly `targetSize`.
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, to targetSize: CGSize) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let srcW = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcH = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let scaleX = targetSize.width / srcW
        let scaleY = targetSize.height / srcH
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var output: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        CVPixelBufferCreate(nil, Int(targetSize.width), Int(targetSize.height),
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &output)
        guard let outBuf = output else { return nil }
        ciContext.render(scaled, to: outBuf)
        return outBuf
    }

    /// Find the first multi-array output from the model prediction
    private func findOutputArray(in prediction: MLFeatureProvider) -> MLMultiArray? {
        for name in prediction.featureNames {
            if let value = prediction.featureValue(for: name),
               value.type == .multiArray,
               let array = value.multiArrayValue {
                return array
            }
        }
        return nil
    }

    /// Parse the raw YOLO tensor (1, 4+C, 8400) into MahjongTile detections.
    /// Bounding boxes are normalized 0..1 with origin at top-left (SwiftUI convention).
    private func parseYOLOOutput(
        _ output: MLMultiArray,
        confidenceThreshold: Float,
        rotated: Bool
    ) -> [MahjongTile] {

        let shape = output.shape.map(\.intValue)
        // Expected shapes: [1, 42, 8400] or [42, 8400]
        let rows: Int       // 42 (4 bbox + 38 classes)
        let numAnchors: Int // 8400

        if shape.count == 3 {
            rows = shape[1]
            numAnchors = shape[2]
        } else if shape.count == 2 {
            rows = shape[0]
            numAnchors = shape[1]
        } else {
            print("⚠️ Unexpected output shape: \(shape)")
            return []
        }

        guard rows == 4 + numClasses else {
            print("⚠️ Output rows \(rows) ≠ 4+\(numClasses)=\(4+numClasses)")
            return []
        }

        let ptr = output.dataPointer.assumingMemoryBound(to: Float.self)
        let stride0 = output.strides.count == 3 ? output.strides[1].intValue : output.strides[0].intValue
        let stride1 = output.strides.count == 3 ? output.strides[2].intValue : output.strides[1].intValue

        var tiles: [MahjongTile] = []

        for a in 0..<numAnchors {
            // Find best class score for this anchor
            var bestClassIdx = 0
            var bestScore: Float = -1

            for c in 0..<numClasses {
                let score = ptr[(4 + c) * stride0 + a * stride1]
                if score > bestScore {
                    bestScore = score
                    bestClassIdx = c
                }
            }

            guard bestScore > confidenceThreshold else { continue }

            let label = classLabels[bestClassIdx]
            guard label != "UNKNOWN" else { continue }

            // bbox: cx, cy, w, h in pixel coords (0..640)
            // CoreML stretched the image to 640×640, so dividing by 640
            // gives us normalized 0..1 coordinates in the original image space.
            let cx = CGFloat(ptr[0 * stride0 + a * stride1]) / 640.0
            let cy = CGFloat(ptr[1 * stride0 + a * stride1]) / 640.0
            let w  = CGFloat(ptr[2 * stride0 + a * stride1]) / 640.0
            let h  = CGFloat(ptr[3 * stride0 + a * stride1]) / 640.0

            // SwiftUI convention: origin at top-left, normalized 0..1
            var box = CGRect(
                x: cx - w / 2,
                y: cy - h / 2,
                width: w,
                height: h
            )

            if rotated {
                // Transform bbox from 90°-rotated image back to original orientation.
                // Rotated image coords (rx, ry) → original (ry, 1-rx-rw)
                let origBox = box
                box = CGRect(
                    x: origBox.minY,
                    y: 1.0 - origBox.maxX,
                    width: origBox.height,
                    height: origBox.width
                )
            }

            if let tile = MahjongTile.fromYOLOLabel(
                label,
                confidence: Double(bestScore),
                boundingBox: box
            ) {
                tiles.append(tile)
            }
        }

        return tiles
    }

    // MARK: - NMS

    /// Standard Non-Maximum Suppression — suppresses overlapping boxes regardless of class.
    /// This allows multiple detections of the SAME class if they are spatially separated.
    private func applyNMS(_ tiles: [MahjongTile], iouThreshold: Float) -> [MahjongTile] {
        guard !tiles.isEmpty else { return [] }

        // Sort by confidence descending
        let sorted = tiles.sorted { $0.confidence > $1.confidence }
        var keep: [MahjongTile] = []
        var suppressed = Set<Int>()

        for i in 0..<sorted.count {
            guard !suppressed.contains(i) else { continue }
            keep.append(sorted[i])
            guard keep.count < maxDetections else { break }

            guard let boxA = sorted[i].boundingBox else { continue }

            for j in (i + 1)..<sorted.count {
                guard !suppressed.contains(j),
                      let boxB = sorted[j].boundingBox else { continue }

                if computeIoU(boxA, boxB) > CGFloat(iouThreshold) {
                    suppressed.insert(j)
                }
            }
        }

        return keep
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    // MARK: - Positional Slot Key

    /// Create a key that combines tile type + coarse grid position.
    /// This allows tracking multiple instances of the same tile at different positions.
    private func slotKey(for tile: MahjongTile) -> String {
        guard let box = tile.boundingBox else {
            return "\(tile.suit.rawValue)-\(tile.rank.rawValue)-nobox"
        }
        // Quantize position to a 10×10 grid to create stable slot IDs
        let gridX = Int(box.midX * 10)
        let gridY = Int(box.midY * 10)
        return "\(tile.suit.rawValue)-\(tile.rank.rawValue)-\(gridX)-\(gridY)"
    }

    // MARK: - Temporal Smoothing

    /// Smooths detections across recent frames using positional slot keys.
    /// Multiple tiles of the same type at different positions are tracked independently.
    private func applyTemporalSmoothing(_ currentFrame: [DetectedInstance]) -> [MahjongTile] {
        recentFrameResults.append(currentFrame)
        if recentFrameResults.count > smoothingWindowSize {
            recentFrameResults.removeFirst()
        }

        // Count appearances of each slot key
        var slotFrequency: [String: (count: Int, bestConfidence: Double, tile: MahjongTile)] = [:]

        for frame in recentFrameResults {
            var seenInFrame: Set<String> = []
            for instance in frame {
                let key = instance.slotKey
                guard !seenInFrame.contains(key) else { continue }
                seenInFrame.insert(key)

                if let existing = slotFrequency[key] {
                    slotFrequency[key] = (
                        count: existing.count + 1,
                        bestConfidence: max(existing.bestConfidence, instance.tile.confidence),
                        tile: instance.tile.confidence > existing.bestConfidence ? instance.tile : existing.tile
                    )
                } else {
                    slotFrequency[key] = (1, instance.tile.confidence, instance.tile)
                }
            }
        }

        let minAppearances = 2

        // Clean up smoothed boxes for disappeared slots
        let activeKeys = Set(slotFrequency.keys.filter { slotFrequency[$0]!.count >= minAppearances })
        smoothedBoxes = smoothedBoxes.filter { activeKeys.contains($0.key) }

        var smoothedTiles: [MahjongTile] = []
        for (key, entry) in slotFrequency where entry.count >= minAppearances {
            let tile = entry.tile

            // Smooth the bounding box
            var finalBox = tile.boundingBox
            if let newBox = tile.boundingBox {
                if let prevBox = smoothedBoxes[key] {
                    let a = boxSmoothingAlpha
                    finalBox = CGRect(
                        x: prevBox.minX + a * (newBox.minX - prevBox.minX),
                        y: prevBox.minY + a * (newBox.minY - prevBox.minY),
                        width: prevBox.width + a * (newBox.width - prevBox.width),
                        height: prevBox.height + a * (newBox.height - prevBox.height)
                    )
                }
                smoothedBoxes[key] = finalBox!
            }

            smoothedTiles.append(MahjongTile(
                suit: tile.suit,
                rank: tile.rank,
                confidence: entry.bestConfidence,
                boundingBox: finalBox
            ))
        }

        smoothedTiles.sort { $0.confidence > $1.confidence }
        return smoothedTiles
    }

    // MARK: - OCR Fallback

    private func recognizeText(in pixelBuffer: CVPixelBuffer) {
        let request = VNRecognizeTextRequest { [weak self] request, error in
            guard error == nil,
                  let observations = request.results as? [VNRecognizedTextObservation]
            else {
                DispatchQueue.main.async {
                    self?.detectedTile = nil
                    self?.detectedTiles = []
                    self?.rawText = ""
                }
                return
            }

            var allText: [String] = []
            var bestTile: MahjongTile?
            var bestConfidence: Double = 0

            for observation in observations {
                guard let candidate = observation.topCandidates(1).first else { continue }
                let text = candidate.string
                allText.append(text)

                let confidence = Double(candidate.confidence)
                if let tile = MahjongTile.fromRecognizedText(text, confidence: confidence),
                   tile.confidence > bestConfidence {
                    bestTile = tile
                    bestConfidence = tile.confidence
                }
            }

            let joined = allText.joined(separator: " ")
            if bestTile == nil && !joined.isEmpty {
                bestTile = MahjongTile.fromRecognizedText(joined, confidence: 0.5)
            }

            DispatchQueue.main.async {
                self?.rawText = joined
                self?.detectedTile = bestTile
                self?.detectedTiles = bestTile.map { [$0] } ?? []
            }
        }

        request.recognitionLanguages = ["zh-Hans", "zh-Hant"]
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([request])
    }
}
