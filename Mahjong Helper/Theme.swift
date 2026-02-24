//
//  Theme.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import SwiftUI

/// Earthy color palette used across the app.
/// See style.md §2 for rationale.
enum AppTheme {

    // MARK: - Accent

    /// Primary accent — warm terracotta / clay
    static let accent = Color(light: .init(hex: 0xC2703E),
                              dark:  .init(hex: 0xD4895A))

    // MARK: - Backgrounds

    /// Main background
    static let background = Color(light: .init(hex: 0xF5F0E8),
                                  dark:  .init(hex: 0x2C2420))

    /// Surface / card background (slight offset from main background)
    static let surface = Color(light: .init(hex: 0xEDE7DB),
                               dark:  .init(hex: 0x3A322D))

    // MARK: - Text

    /// Primary text color
    static let textPrimary = Color(light: .init(hex: 0x3B2F2F),
                                   dark:  .init(hex: 0xF0E6D8))

    /// Secondary / caption text color
    static let textSecondary = Color(light: .init(hex: 0x8C7B75),
                                     dark:  .init(hex: 0xA89B94))

    // MARK: - Overlay

    /// Semi-transparent overlay for camera HUD
    static let overlayBackground = Color.black.opacity(0.55)
}

// MARK: - Helpers

private extension Color {
    /// Create a dynamic color that adapts to light / dark mode.
    init(light: UIColor, dark: UIColor) {
        self.init(uiColor: UIColor { traits in
            traits.userInterfaceStyle == .dark ? dark : light
        })
    }
}

private extension UIColor {
    /// Convenience init from a hex integer (e.g. `0xC2703E`).
    convenience init(hex: UInt32, alpha: CGFloat = 1.0) {
        self.init(
            red:   CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >>  8) & 0xFF) / 255,
            blue:  CGFloat( hex        & 0xFF) / 255,
            alpha: alpha
        )
    }
}
