// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class InlineMathDetectionTests: XCTestCase {
    func testProseWithShellVariablesIsNotMath() {
        for prose in [
            "Set $HOME and then export $PATH before running the script.",
            "Run `$ pip install openai` and then `$ python demo.py` to try it.",
            "La variable $HOME identifica el directorio raiz, mientras que $PATH garantiza el resto.",
            "Use awk '{print $1}' on the log, then check $? for errors.",
            "Regex: match a literal paren with \\( and close it with \\) in POSIX.",
            "Exporta $PORT=$PORT2 antes de arrancar.",
        ] {
            XCTAssertFalse(RichText.containsInlineMath(prose), prose)
        }
    }

    func testPricesAreNotMath() {
        XCTAssertFalse(RichText.containsInlineMath("El coste pasa de $10 a $25 por millon de tokens."))
        XCTAssertFalse(RichText.containsInlineMath("Cuesta entre $10-$25 al mes."))
    }

    func testFormulasAreStillDetected() {
        for formula in [
            "The area is $A = \\pi r^2$ for a circle.",
            "Con $x_1$ y $x_2$ se resuelve la ecuacion.",
            "La complejidad es $O(n^2)$ en el peor caso.",
            "Escrito como \\(e^{i\\pi} + 1 = 0\\), la identidad de Euler.",
            "Sea $n$ el numero de capas.",
        ] {
            XCTAssertTrue(RichText.containsInlineMath(formula), formula)
        }
    }

    func testUnpairedDelimiterIsNotMath() {
        XCTAssertFalse(RichText.containsInlineMath("Revisa $LOG_FILE antes de continuar."))
    }

    func testSymbolOnlyFormulasBecomeText() {
        XCTAssertEqual(RichText.symbolizingMath("a $\\neq$ b"), "a ≠ b")
        XCTAssertEqual(RichText.symbolizingMath("**Paso 1 $\\rightarrow$ Paso 2**"), "**Paso 1 → Paso 2**")
        XCTAssertEqual(RichText.symbolizingMath("si $x \\leq y$ y $\\alpha$"), "si x ≤ y y α")
        XCTAssertFalse(RichText.containsInlineMath(RichText.symbolizingMath("Entonces $a \\neq b$.")))
    }

    func testRealFormulasAndCodeAreLeftAlone() {
        for text in [
            "The area is $A = \\pi r^2$ for a circle.",
            "Suma $\\sum_{i=1}^n i$ en total.",
            "Usa $\\frac{a}{b}$ aqui.",
            "Escribe `$\\neq$` literal.",
        ] {
            XCTAssertEqual(RichText.symbolizingMath(text), text, text)
        }
    }

    func testSpanDoesNotCrossLines() {
        XCTAssertFalse(RichText.containsInlineMath("Primero $HOME\ny luego $PATH del sistema."))
    }
}
