//
//  MahjongTile.swift
//  Mahjong Helper
//
//  Created by Ji Dai on 2/24/26.
//

import Foundation
import CoreGraphics

/// Represents a single Mahjong tile detected in the camera frame.
struct MahjongTile: Identifiable, Equatable {
    let id = UUID()
    let suit: Suit
    let rank: Rank
    let confidence: Double              // 0 … 1
    let boundingBox: CGRect?            // normalized Vision bounding box (optional)

    init(suit: Suit, rank: Rank, confidence: Double, boundingBox: CGRect? = nil) {
        self.suit = suit
        self.rank = rank
        self.confidence = confidence
        self.boundingBox = boundingBox
    }

    static func == (lhs: MahjongTile, rhs: MahjongTile) -> Bool {
        lhs.suit == rhs.suit && lhs.rank == rhs.rank
    }

    /// Chinese display name, e.g. "五万" or "红中"
    var chineseName: String {
        switch suit {
        case .character: return "\(rank.chineseNumber)萬"
        case .bamboo:    return "\(rank.chineseNumber)索"
        case .dot:       return "\(rank.chineseNumber)筒"
        case .wind:      return rank.windChinese
        case .dragon:    return rank.dragonChinese
        case .flower:    return rank.flowerChinese
        case .season:    return rank.seasonChinese
        }
    }

    /// English display name, e.g. "5 Bamboo" or "Red Dragon"
    var englishName: String {
        switch suit {
        case .bamboo:    return "\(rank.rawValue) Bamboo"
        case .character: return "\(rank.rawValue) Character"
        case .dot:       return "\(rank.rawValue) Dot"
        case .wind:      return "\(rank.windEnglish) Wind"
        case .dragon:    return "\(rank.dragonEnglish) Dragon"
        case .flower:    return "Flower \(rank.rawValue)"
        case .season:    return "Season \(rank.rawValue)"
        }
    }

    /// Emoji or symbol for quick visual hint
    var symbol: String {
        switch suit {
        case .bamboo:    return "🎋"
        case .character: return "🀄"
        case .dot:       return "🔴"
        case .wind:      return "🌬️"
        case .dragon:    return "🐉"
        case .flower:    return "🌸"
        case .season:    return "🍂"
        }
    }
}

// MARK: - Suit & Rank

extension MahjongTile {

    enum Suit: String, CaseIterable, Codable {
        case bamboo, character, dot
        case wind, dragon
        case flower, season

        var chineseName: String {
            switch self {
            case .bamboo:    return "索"
            case .character: return "萬"
            case .dot:       return "筒"
            case .wind:      return "風"
            case .dragon:    return "箭"
            case .flower:    return "花"
            case .season:    return "季"
            }
        }
    }

    enum Rank: Int, CaseIterable, Codable {
        case one = 1, two, three, four, five, six, seven, eight, nine

        var chineseNumber: String {
            switch self {
            case .one:   return "一"
            case .two:   return "二"
            case .three: return "三"
            case .four:  return "四"
            case .five:  return "五"
            case .six:   return "六"
            case .seven: return "七"
            case .eight: return "八"
            case .nine:  return "九"
            }
        }

        var windChinese: String {
            switch self {
            case .one:   return "東"
            case .two:   return "南"
            case .three: return "西"
            case .four:  return "北"
            default:     return chineseNumber
            }
        }

        var windEnglish: String {
            switch self {
            case .one:   return "East"
            case .two:   return "South"
            case .three: return "West"
            case .four:  return "North"
            default:     return "\(rawValue)"
            }
        }

        var dragonChinese: String {
            switch self {
            case .one:   return "中"
            case .two:   return "發"
            case .three: return "白"
            default:     return chineseNumber
            }
        }

        var dragonEnglish: String {
            switch self {
            case .one:   return "Red"
            case .two:   return "Green"
            case .three: return "White"
            default:     return "\(rawValue)"
            }
        }

        var flowerChinese: String {
            switch self {
            case .one:   return "春"
            case .two:   return "夏"
            case .three: return "秋"
            case .four:  return "冬"
            case .five:  return "梅"
            case .six:   return "蘭"
            case .seven: return "菊"
            case .eight: return "竹"
            default:     return "花\(rawValue)"
            }
        }

        var seasonChinese: String {
            flowerChinese  // flowers/seasons share the same Chinese names
        }
    }
}

// MARK: - YOLO Label → Tile Mapping
//
// The YOLOv11 model from nikmomo/Mahjong-YOLO uses 38 class labels:
//
//  Index  Label   Chinese      Index  Label   Chinese
//  ─────  ─────   ───────      ─────  ─────   ───────
//   0     1m      一萬          1     1p      一筒
//   2     1s      一索          3     1z      東
//   4     2m      二萬          5     2p      二筒
//   6     2s      二索          7     2z      南
//   8     3m      三萬          9     3p      三筒
//  10     3s      三索         11     3z      西
//  12     4m      四萬         13     4p      四筒
//  14     4s      四索         15     4z      北
//  16     5m      五萬         17     5p      五筒
//  18     5s      五索         19     5z      中
//  20     6m      六萬         21     6p      六筒
//  22     6s      六索         23     6z      發
//  24     7m      七萬         25     7p      七筒
//  26     7s      七索         27     7z      白
//  28     8m      八萬         29     8p      八筒
//  30     8s      八索         31     9m      九萬
//  32     9p      九筒         33     9s      九索
//  34     UNKNOWN              35     0m      赤五萬
//  36     0p      赤五筒       37     0s      赤五索

extension MahjongTile {

    /// Maps a YOLO class label (e.g. "5m", "1z", "0p") to a MahjongTile.
    static func fromYOLOLabel(_ label: String,
                              confidence: Double,
                              boundingBox: CGRect? = nil) -> MahjongTile? {
        let key = label.lowercased()
        guard key != "unknown",
              let (suit, rank) = yoloLabelMap[key] else { return nil }
        return MahjongTile(suit: suit, rank: rank,
                           confidence: confidence, boundingBox: boundingBox)
    }

    /// Complete mapping of YOLO label → (Suit, Rank)
    private static let yoloLabelMap: [String: (Suit, Rank)] = [
        // 萬子 (Characters / Man)
        "1m": (.character, .one),   "2m": (.character, .two),   "3m": (.character, .three),
        "4m": (.character, .four),  "5m": (.character, .five),  "6m": (.character, .six),
        "7m": (.character, .seven), "8m": (.character, .eight), "9m": (.character, .nine),
        "0m": (.character, .five),  // 赤五萬 (red five)

        // 筒子 (Dots / Pin)
        "1p": (.dot, .one),   "2p": (.dot, .two),   "3p": (.dot, .three),
        "4p": (.dot, .four),  "5p": (.dot, .five),  "6p": (.dot, .six),
        "7p": (.dot, .seven), "8p": (.dot, .eight), "9p": (.dot, .nine),
        "0p": (.dot, .five),  // 赤五筒 (red five)

        // 索子 (Bamboo / Sou)
        "1s": (.bamboo, .one),   "2s": (.bamboo, .two),   "3s": (.bamboo, .three),
        "4s": (.bamboo, .four),  "5s": (.bamboo, .five),  "6s": (.bamboo, .six),
        "7s": (.bamboo, .seven), "8s": (.bamboo, .eight), "9s": (.bamboo, .nine),
        "0s": (.bamboo, .five),  // 赤五索 (red five)

        // 風牌 (Winds)
        "1z": (.wind, .one),    // 東
        "2z": (.wind, .two),    // 南
        "3z": (.wind, .three),  // 西
        "4z": (.wind, .four),   // 北

        // 箭牌 (Dragons)
        // nikmomo/Mahjong-YOLO convention: 5z=White, 6z=Green, 7z=Red
        "5z": (.dragon, .three),  // 白 (White Dragon)
        "6z": (.dragon, .two),    // 發 (Green Dragon)
        "7z": (.dragon, .one),    // 中 (Red Dragon)
    ]
}

// MARK: - OCR Text → Tile Mapping

extension MahjongTile {

    /// Map a recognized Chinese character (or small group) to a MahjongTile.
    /// Returns nil if the text doesn't match any known tile character.
    static func fromRecognizedText(_ text: String, confidence: Double) -> MahjongTile? {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let numberMap: [String: Rank] = [
            "一": .one, "二": .two, "三": .three, "四": .four, "五": .five,
            "六": .six, "七": .seven, "八": .eight, "九": .nine,
            "1": .one, "2": .two, "3": .three, "4": .four, "5": .five,
            "6": .six, "7": .seven, "8": .eight, "9": .nine,
        ]

        // Full tile names (e.g. "一万", "五条")
        for (numStr, rank) in numberMap {
            if t.contains(numStr) {
                if t.contains("万") || t.contains("萬") {
                    return MahjongTile(suit: .character, rank: rank, confidence: confidence)
                }
                if t.contains("条") || t.contains("索") {
                    return MahjongTile(suit: .bamboo, rank: rank, confidence: confidence)
                }
                if t.contains("筒") || t.contains("饼") {
                    return MahjongTile(suit: .dot, rank: rank, confidence: confidence)
                }
            }
        }

        // Standalone suit character
        if t.contains("万") || t.contains("萬") {
            return MahjongTile(suit: .character, rank: .one, confidence: confidence * 0.5)
        }

        // Winds
        if t.contains("东") || t.contains("東") { return MahjongTile(suit: .wind, rank: .one, confidence: confidence) }
        if t.contains("南") { return MahjongTile(suit: .wind, rank: .two, confidence: confidence) }
        if t.contains("西") { return MahjongTile(suit: .wind, rank: .three, confidence: confidence) }
        if t.contains("北") { return MahjongTile(suit: .wind, rank: .four, confidence: confidence) }

        // Dragons
        if t.contains("中") { return MahjongTile(suit: .dragon, rank: .one, confidence: confidence) }
        if t.contains("發") || t.contains("发") { return MahjongTile(suit: .dragon, rank: .two, confidence: confidence) }

        return nil
    }
}
