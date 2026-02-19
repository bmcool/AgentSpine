"""Tests for context_estimate (token estimation)."""

from __future__ import annotations

import unittest

from src.context_estimate import estimate_tokens


class EstimateTokensTests(unittest.TestCase):
    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)

    def test_short_string_returns_at_least_one(self) -> None:
        self.assertGreaterEqual(estimate_tokens("a"), 1)

    def test_roughly_four_chars_per_token(self) -> None:
        # Heuristic: len // 4
        self.assertEqual(estimate_tokens("a" * 4), 1)
        self.assertEqual(estimate_tokens("a" * 8), 2)
        self.assertEqual(estimate_tokens("a" * 40), 10)


if __name__ == "__main__":
    unittest.main()
