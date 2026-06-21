"""
test_tools.py

Unit tests for the deterministic tools (ratio calculator and compliance
screener) that need no network or API key, so they can run anywhere,
including in CI. search_documents is excluded here since it requires a
live OpenAI key and a populated vector store.

Run with: python -m pytest tests/  (or: python tests/test_tools.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools import calculate_financial_ratio, flag_compliance_risk


def test_calculate_financial_ratio_basic():
    result = calculate_financial_ratio("current_ratio", 200, 100)
    assert "2.00" in result


def test_calculate_financial_ratio_zero_denominator():
    result = calculate_financial_ratio("current_ratio", 100, 0)
    assert "Error" in result


def test_flag_compliance_risk_detects_structuring():
    text = "The account showed signs of structuring with repeated small transfers."
    result = flag_compliance_risk(text)
    assert "STRUCTURING" in result


def test_flag_compliance_risk_detects_multiple_flags():
    text = "This shell company moved funds in rapid movement to a high-risk jurisdiction."
    result = flag_compliance_risk(text)
    assert "SHELL COMPANY" in result
    assert "RAPID MOVEMENT" in result
    assert "HIGH-RISK JURISDICTION" in result


def test_flag_compliance_risk_clean_text():
    text = "This is a routine, well-documented business transaction."
    result = flag_compliance_risk(text)
    assert "No predefined" in result


if __name__ == "__main__":
    test_calculate_financial_ratio_basic()
    test_calculate_financial_ratio_zero_denominator()
    test_flag_compliance_risk_detects_structuring()
    test_flag_compliance_risk_detects_multiple_flags()
    test_flag_compliance_risk_clean_text()
    print("All tests passed.")
