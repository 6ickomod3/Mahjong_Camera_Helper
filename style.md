# Mahjong Helper – Style Guide

## 1. Design Language
- Follow the latest iOS design conventions (iOS 18+): use native SwiftUI components, SF Symbols, and system materials/blur effects.
- Embrace translucency, depth, and smooth animations consistent with Apple's Human Interface Guidelines.
- Use Dynamic Type for all text to support accessibility.
- Prefer rounded corners, generous spacing, and clean visual hierarchy.

## 2. Color Palette
- Use an **earthy, warm** color palette that feels coherent across every screen.
- Keep colors consistent — accent color, backgrounds, and text tints should all draw from the same earthy family.
- **Be restrained with color.** Avoid introducing new colors unless absolutely necessary. The app should feel calm and unified, never too colorful or distracting.
- Suggested base tones (adjust as needed):
  - **Primary accent:** warm terracotta / clay (`#C2703E` or similar)
  - **Secondary accent:** muted sage green (`#8A9A5B`) — use sparingly and only when a second color is truly needed
  - **Background:** soft warm cream (`#F5F0E8`) in light mode, deep charcoal-brown (`#2C2420`) in dark mode
  - **Surface / Card:** slightly lighter or darker shade of the background
  - **Text primary:** deep espresso brown (`#3B2F2F`) in light mode, warm off-white (`#F0E6D8`) in dark mode
  - **Text secondary:** muted stone gray (`#8C7B75`)
- Support both Light and Dark modes with matching earthy tones.
- **Text readability:** Ensure sufficient contrast between text and its background at all times. Text colors should feel natural against earthy backgrounds — never harsh or neon. Use the primary/secondary text colors consistently; avoid tinting body text with accent colors.

## 3. Core Features
1. **Camera access** — the app opens the device camera and shows a live preview.
2. **Mahjong tile identification** — using Vision / CoreML, the app detects and classifies Mahjong tiles visible in the camera frame in real time.
3. **Display detected tile info** — the recognized tile name (and any relevant details) is shown as an overlay on the camera view, styled consistently with the earthy theme.
