// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ToolResultLimitTests: XCTestCase {
    func testShortResultPassesThrough() {
        XCTAssertEqual(ToolResultLimit.apply("hello", limit: 10), "hello")
    }

    func testZeroSendsItWhole() {
        let text = String(repeating: "a", count: 50_000)
        XCTAssertEqual(ToolResultLimit.apply(text, limit: 0), text)
    }

    func testKeepsHeadAndTailWithANote() {
        let text = String(repeating: "h", count: 600) + String(repeating: "m", count: 800) + String(repeating: "t", count: 600)
        let capped = ToolResultLimit.apply(text, limit: 1_000)

        XCTAssertTrue(capped.hasPrefix(String(repeating: "h", count: 600)))
        XCTAssertTrue(capped.hasSuffix(String(repeating: "t", count: 250)))
        XCTAssertTrue(capped.contains("1000 of 2000 characters omitted"))
    }
}
