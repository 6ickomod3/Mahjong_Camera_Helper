//
//  CameraPreviewView.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import SwiftUI
import AVFoundation

/// A SwiftUI wrapper around `AVCaptureVideoPreviewLayer`.
struct CameraPreviewView: UIViewRepresentable {

    let session: AVCaptureSession
    /// The rotation angle to apply to the preview layer connection.
    /// Must match the video data output's angle so detection boxes align.
    let videoAngle: CGFloat

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.currentAngle = videoAngle
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        // SwiftUI calls this when videoAngle changes — push the new angle
        uiView.currentAngle = videoAngle
        uiView.applyRotation()
    }
}

/// Thin UIView subclass that hosts an `AVCaptureVideoPreviewLayer`.
final class PreviewUIView: UIView {

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        layer as! AVCaptureVideoPreviewLayer
    }

    /// The rotation angle to apply. Updated from the SwiftUI side.
    var currentAngle: CGFloat = 90

    override func layoutSubviews() {
        super.layoutSubviews()
        applyRotation()
    }

    /// Apply the current rotation angle to the preview connection.
    func applyRotation() {
        guard let connection = previewLayer.connection else { return }
        if connection.isVideoRotationAngleSupported(currentAngle) {
            connection.videoRotationAngle = currentAngle
        }
    }
}
