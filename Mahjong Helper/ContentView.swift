//
//  ContentView.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import SwiftUI

struct ContentView: View {

    @StateObject private var cameraManager = CameraManager()
    @StateObject private var tileDetector  = TileDetector()

    /// Tracks the zoom factor at the start of a pinch gesture.
    @State private var lastZoomFactor: CGFloat = 1.0
    /// True while a pinch gesture is active (prevents external sync).
    @State private var isPinching = false

    var body: some View {
        ZStack {
            // Full-screen camera preview
            if cameraManager.permissionGranted {
                CameraPreviewView(session: cameraManager.session,
                                  videoAngle: cameraManager.videoAngle)
                    .ignoresSafeArea()
                    .gesture(pinchToZoom)
                    .overlay { detectionBoxOverlay }
            } else {
                permissionDeniedView
            }

            // HUD overlay
            VStack {
                // Top bar: zoom + detection mode
                HStack {
                    zoomIndicator
                    Spacer()
                    detectionModeBadge
                }
                .padding(.horizontal)
                .padding(.top, 12)

                Spacer()

                // Empty state hint when nothing detected
                if tileDetector.detectedTiles.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: "viewfinder")
                            .font(.title2)
                            .foregroundStyle(AppTheme.accent)
                        Text("对准麻将牌")
                            .font(.subheadline)
                            .foregroundStyle(AppTheme.textSecondary)
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .padding(.horizontal)
                    .padding(.bottom, 48)
                }
            }
        }
        .onAppear {
            cameraManager.requestPermissionAndSetup()
            // Wire camera frames → detector
            cameraManager.onFrame = { [weak tileDetector] buffer in
                tileDetector?.processFrame(buffer)
            }
        }
        .onDisappear {
            cameraManager.stopSession()
        }
        // Keep lastZoomFactor in sync when zoom changes outside of a pinch
        .onChange(of: cameraManager.zoomFactor) { _, newValue in
            if !isPinching {
                lastZoomFactor = newValue
            }
        }
    }

    // MARK: - Sub-views

    /// Bounding box overlay drawn on top of the camera preview.
    /// Accounts for the preview layer's `.resizeAspectFill` cropping.
    @ViewBuilder
    private var detectionBoxOverlay: some View {
        GeometryReader { geo in
            let size = geo.size
            let cam = cameraManager.frameSize

            // Calculate aspectFill scaling: the preview fills the view,
            // cropping one axis. We need the same mapping for bboxes.
            let imageAspect = cam.width > 0 ? cam.width / cam.height : size.width / size.height
            let viewAspect = size.width / size.height

            // Scale & offset to convert 0..1 normalized coords to screen pts.
            // When image is wider → horizontal crop; when taller → vertical crop.
            let isWider = imageAspect > viewAspect
            let scaleX = isWider ? size.height * imageAspect : size.width
            let scaleY = isWider ? size.height : size.width / imageAspect
            let offsetX = isWider ? (scaleX - size.width) / 2 : 0.0
            let offsetY = isWider ? 0.0 : (scaleY - size.height) / 2

            ForEach(tileDetector.detectedTiles) { tile in
                if let box = tile.boundingBox {
                    // box is normalized 0..1 with origin top-left (SwiftUI convention)
                    let rect = CGRect(
                        x: box.minX * scaleX - offsetX,
                        y: box.minY * scaleY - offsetY,
                        width: box.width * scaleX,
                        height: box.height * scaleY
                    )

                    // Bounding box rectangle
                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                        .stroke(AppTheme.accent, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)

                    // Label pill above the box
                    Text("\(tile.chineseName) \(Int(tile.confidence * 100))%")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(AppTheme.accent.opacity(0.85))
                        .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                        .position(
                            x: rect.midX,
                            y: max(rect.minY - 12, 12)
                        )
                }
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
        .animation(.easeInOut(duration: 0.15), value: tileDetector.detectedTiles.map(\.id))
    }

    /// Pinch-to-zoom gesture
    private var pinchToZoom: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                isPinching = true
                let newZoom = lastZoomFactor * value.magnification
                cameraManager.setZoom(newZoom)
            }
            .onEnded { _ in
                lastZoomFactor = cameraManager.zoomFactor
                isPinching = false
            }
    }

    /// Small pill showing the current zoom level
    @ViewBuilder
    private var zoomIndicator: some View {
        if cameraManager.permissionGranted {
            Text(String(format: "%.1f×", cameraManager.zoomFactor))
                .font(.caption.weight(.semibold).monospacedDigit())
                .foregroundStyle(AppTheme.textPrimary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial)
                .clipShape(Capsule())
        }
    }

    /// Small pill showing CoreML vs OCR detection mode
    @ViewBuilder
    private var detectionModeBadge: some View {
        if cameraManager.permissionGranted {
            HStack(spacing: 4) {
                Circle()
                    .fill(tileDetector.detectionMode == .coreML
                          ? Color.green : AppTheme.accent)
                    .frame(width: 6, height: 6)
                Text(tileDetector.detectionMode.rawValue)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(AppTheme.textSecondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
        }
    }

    /// Shown when camera access was denied
    private var permissionDeniedView: some View {
        VStack(spacing: 20) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundStyle(AppTheme.accent)

            Text("Camera Access Required")
                .font(.title3.weight(.semibold))
                .foregroundStyle(AppTheme.textPrimary)

            Text("Open Settings and allow camera access\nso Mahjong Helper can identify tiles.")
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundStyle(AppTheme.textSecondary)

            Button {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            } label: {
                Text("Open Settings")
                    .fontWeight(.medium)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 10)
                    .background(AppTheme.accent)
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(AppTheme.background)
    }
}

#Preview {
    ContentView()
}
