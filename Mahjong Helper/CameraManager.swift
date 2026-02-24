//
//  CameraManager.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import AVFoundation
import UIKit

/// Manages the AVCaptureSession, camera input, and video output.
final class CameraManager: NSObject, ObservableObject {

    // MARK: - Published state

    @Published var permissionGranted = false
    @Published var isSessionRunning  = false
    @Published var zoomFactor: CGFloat = 1.0
    @Published var minZoom: CGFloat = 1.0
    @Published var maxZoom: CGFloat = 1.0

    /// The camera frame dimensions (in pixels, rotated to match current orientation).
    /// Used by the bounding box overlay to correct for aspectFill cropping.
    @Published var frameSize: CGSize = .zero

    /// The current video rotation angle matching the device orientation.
    /// Published so that CameraPreviewView can keep its preview layer in sync.
    @Published var videoAngle: CGFloat = 90

    // MARK: - Capture objects

    let session = AVCaptureSession()

    /// Called on every video frame; set by TileDetector.
    var onFrame: ((CMSampleBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoOutput  = AVCaptureVideoDataOutput()
    private var currentCamera: AVCaptureDevice?
    private var orientationObserver: NSObjectProtocol?

    /// Tracks frame size on the capture queue to avoid reading @Published from a background thread.
    private var lastKnownFrameSize: CGSize = .zero

    // MARK: - Lifecycle

    func requestPermissionAndSetup() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.permissionGranted = true }
            sessionQueue.async { self.configureSession() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { self.permissionGranted = granted }
                if granted {
                    self.sessionQueue.async { self.configureSession() }
                }
            }
        default:
            DispatchQueue.main.async { self.permissionGranted = false }
        }
    }

    func startSession() {
        sessionQueue.async {
            if !self.session.isRunning {
                self.session.startRunning()
                DispatchQueue.main.async { self.isSessionRunning = true }
            }
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
                DispatchQueue.main.async { self.isSessionRunning = false }
            }
        }
    }

    deinit {
        if let observer = orientationObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        UIDevice.current.endGeneratingDeviceOrientationNotifications()
    }

    // MARK: - Orientation

    /// Maps UIDeviceOrientation → AVCaptureConnection videoRotationAngle.
    /// The rear camera sensor is mounted in landscape-left orientation, so:
    ///   portrait → 90°,  landscapeLeft → 0°,  landscapeRight → 180°,  upsideDown → 270°
    private static func rotationAngle(for orientation: UIDeviceOrientation) -> CGFloat {
        switch orientation {
        case .portrait:            return 90
        case .landscapeLeft:       return 0    // home button right
        case .landscapeRight:      return 180  // home button left
        case .portraitUpsideDown:  return 270
        default:                   return 90   // faceUp / faceDown / unknown → keep portrait
        }
    }

    private func startOrientationObserving() {
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        orientationObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.orientationDidChangeNotification,
            object: nil,
            queue: nil
        ) { [weak self] _ in
            self?.updateVideoRotation()
        }
    }

    /// Update both the video data output and the published angle when orientation changes.
    private func updateVideoRotation() {
        let orientation = UIDevice.current.orientation
        // Ignore flat orientations — keep the last known angle
        guard orientation != .faceUp, orientation != .faceDown, orientation != .unknown else { return }

        let angle = Self.rotationAngle(for: orientation)

        sessionQueue.async { [weak self] in
            guard let self else { return }
            // Update the video data output connection so pixel buffers match the new orientation
            if let connection = self.videoOutput.connection(with: .video),
               connection.isVideoRotationAngleSupported(angle) {
                connection.videoRotationAngle = angle
            }
            // Reset size tracker so captureOutput picks up the new dimensions
            self.lastKnownFrameSize = .zero
            DispatchQueue.main.async {
                self.videoAngle = angle
            }
        }
    }

    // MARK: - Private

    /// Set the zoom factor on the active camera device.
    func setZoom(_ factor: CGFloat) {
        guard let device = currentCamera else { return }
        let clamped = min(max(factor, minZoom), maxZoom)
        do {
            try device.lockForConfiguration()
            device.videoZoomFactor = clamped
            device.unlockForConfiguration()
            DispatchQueue.main.async { self.zoomFactor = clamped }
        } catch {}
    }

    private func configureSession() {
        session.beginConfiguration()
        // .photo preserves the virtual device's full zoom range (0.5× – 5×+)
        // whereas the default (.high) restricts minZoom to 1.0.
        session.sessionPreset = .photo

        // Prefer a multi-lens virtual device so we get the full zoom range
        // (0.5× ultra-wide through telephoto), just like the system Camera app.
        let camera: AVCaptureDevice? = {
            // Triple camera (ultra-wide + wide + tele) — iPhone Pro models
            if let triple = AVCaptureDevice.default(.builtInTripleCamera,
                                                     for: .video,
                                                     position: .back) {
                return triple
            }
            // Dual wide (ultra-wide + wide) — e.g. iPhone 13/14/15 base models
            if let dualWide = AVCaptureDevice.default(.builtInDualWideCamera,
                                                       for: .video,
                                                       position: .back) {
                return dualWide
            }
            // Dual camera (wide + tele)
            if let dual = AVCaptureDevice.default(.builtInDualCamera,
                                                    for: .video,
                                                    position: .back) {
                return dual
            }
            // Fallback: single wide-angle lens
            return AVCaptureDevice.default(.builtInWideAngleCamera,
                                           for: .video,
                                           position: .back)
        }()

        guard let camera,
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)
        currentCamera = camera

        // Video data output (for frame-by-frame analysis)
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.frameProcessing"))
        videoOutput.alwaysDiscardsLateVideoFrames = true
        // CoreML requires 32BGRA; the camera defaults to YCbCr 420f which is unsupported.
        videoOutput.videoSettings = [
            String(kCVPixelBufferPixelFormatTypeKey): kCVPixelFormatType_32BGRA
        ]
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)

            // Set initial rotation to match current device orientation
            let angle = Self.rotationAngle(for: UIDevice.current.orientation)
            if let connection = videoOutput.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(angle) {
                    connection.videoRotationAngle = angle
                }
            }
            DispatchQueue.main.async { self.videoAngle = angle }
        }

        session.commitConfiguration()

        // Set zoom to the widest the device supports and publish limits.
        // On multi-lens devices this is typically 0.5× (the ultra-wide lens).
        let minFactor = camera.minAvailableVideoZoomFactor
        let maxFactor = min(camera.maxAvailableVideoZoomFactor, 10.0) // cap at 10×
        DispatchQueue.main.async {
            self.minZoom = minFactor
            self.maxZoom = maxFactor
            self.zoomFactor = minFactor
        }
        do {
            try camera.lockForConfiguration()
            camera.videoZoomFactor = minFactor
            camera.unlockForConfiguration()
        } catch {}

        startSession()
        startOrientationObserving()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Update frame dimensions whenever they change (e.g. orientation rotation)
        if let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            let w = CGFloat(CVPixelBufferGetWidth(pb))
            let h = CGFloat(CVPixelBufferGetHeight(pb))
            let newSize = CGSize(width: w, height: h)
            if newSize != lastKnownFrameSize {
                lastKnownFrameSize = newSize
                DispatchQueue.main.async { self.frameSize = newSize }
            }
        }
        // DEBUG: confirm frames are being forwarded
        if onFrame == nil {
            print("⚠️ [CameraManager] onFrame callback is nil — frames not forwarded!")
        }
        onFrame?(sampleBuffer)
    }
}
