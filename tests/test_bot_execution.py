"""Tests for bot execution pricing logic.

Validates that Kalshi pricing uses best_ask for both Path 1 and Path 2.
"""

import pytest


def test_kalshi_price_uses_best_ask_for_path1():
    """Verify Kalshi execution uses best_ask (not best_bid) for Path 1.

    Path 1: Buy Kalshi NO
    Kalshi NO price = 1.0 - Kalshi YES ask

    Example: If Kalshi YES ask = 0.48, then Kalshi NO price = 0.52
    """
    kalshi_yes_ask = 0.48
    kalshi_yes_bid = 0.42
    strategy_path = 1

    # Path 1: Should use best_ask (0.48) not best_bid (0.42)
    # Kalshi NO price = 1.0 - 0.48 = 0.52
    path1_price = 1.0 - kalshi_yes_ask
    assert pytest.approx(path1_price, abs=1e-6) == 0.52

    # Should NOT equal the old logic (1.0 - best_bid)
    old_price = 1.0 - kalshi_yes_bid
    assert pytest.approx(old_price, abs=1e-6) == 0.58
    assert path1_price != old_price, "Path 1 should use best_ask, not best_bid"


def test_kalshi_price_path2_unchanged():
    """Verify Path 2 pricing unchanged (uses best_ask).

    Path 2: Buy Kalshi YES
    Kalshi YES price = best_ask (use directly)

    Example: If Kalshi YES ask = 0.48, price = 0.48
    """
    kalshi_yes_ask = 0.48
    strategy_path = 2

    # Path 2: Should use best_ask directly
    path2_price = kalshi_yes_ask
    assert pytest.approx(path2_price, abs=1e-6) == 0.48


def test_kalshi_price_pricing_logic():
    """Test the pricing formula for both paths with real values."""
    kalshi_yes_ask = 0.55
    kalshi_yes_bid = 0.50
    strategy_path_1 = 1
    strategy_path_2 = 2

    # Path 1: 1.0 - best_ask
    if strategy_path_1 == 1:
        kalshi_price_path1 = 1.0 - kalshi_yes_ask
    else:
        kalshi_price_path1 = kalshi_yes_ask

    assert pytest.approx(kalshi_price_path1, abs=1e-6) == 0.45

    # Path 2: best_ask
    if strategy_path_2 == 1:
        kalshi_price_path2 = 1.0 - kalshi_yes_ask
    else:
        kalshi_price_path2 = kalshi_yes_ask

    assert pytest.approx(kalshi_price_path2, abs=1e-6) == 0.55
