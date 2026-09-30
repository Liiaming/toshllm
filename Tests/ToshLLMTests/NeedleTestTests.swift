// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

@MainActor
final class NeedleTestTests: XCTestCase {
    func testTheCodeSitsAtTheRequestedDepthAndTheTextHasTheRequestedSize() {
        for depth in NeedleTest.depths {
            let text = NeedleTest.prompt(tokens: 8192, depth: depth, code: "AMBER-1234")
            let position = text.range(of: "AMBER-1234")!.lowerBound
            let fraction = Double(text.distance(from: text.startIndex, to: position)) / Double(text.count)
            XCTAssertEqual(fraction, Double(depth) / 100, accuracy: 0.03)
            XCTAssertEqual(Double(text.count) / NeedleTest.charsPerToken, 8192, accuracy: 8192 * 0.1)
        }
    }

    func testOnlyLengthsThatFitTheContextAreOffered() {
        let test = NeedleTest()
        XCTAssertEqual(test.lengths(fitting: 16384), [8192])
        XCTAssertEqual(test.lengths(fitting: 262144), [8192, 32768, 131072])
        XCTAssertEqual(test.lengths(fitting: nil), [])
    }
}
