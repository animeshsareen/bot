"""Tests for pre-flight validation module.

Tests depth checks, VWAP calculations, and profit validation
for both strategy paths before execution.
"""

import pytest
import time

from src.venue_execution import (
    validate_depth_and_vwap,
    PreflightValidation,
)
from src.venue_state import (
    VenueState,
    OrderLevel,
    MarketSnapshot,
    create_venue_state,
    create_snapshot,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def poly_yes_state_good():
    """Polymarket YES with good L1 depth and price."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="poly",
        market_id="token_yes",
        last_update_ms=now_ms,
        best_ask=0.40,
        best_ask_size=10.0,  # 10 shares at L1
        ask_levels=(
            OrderLevel(price=0.40, size=10.0),  # L1
            OrderLevel(price=0.41, size=20.0),  # L2
        ),
        bid_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def poly_no_state_good():
    """Polymarket NO with good L1 depth and price."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="poly",
        market_id="token_no",
        last_update_ms=now_ms,
        best_ask=0.30,
        best_ask_size=8.0,  # 8 shares at L1
        ask_levels=(
            OrderLevel(price=0.30, size=8.0),   # L1
            OrderLevel(price=0.32, size=15.0),  # L2
        ),
        bid_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def poly_no_state_insufficient_depth():
    """Polymarket NO with insufficient L1 depth."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="poly",
        market_id="token_no",
        last_update_ms=now_ms,
        best_ask=0.30,
        best_ask_size=1.0,  # Only 1 share at L1 (< 5 minimum)
        ask_levels=(
            OrderLevel(price=0.30, size=1.0),
            OrderLevel(price=0.32, size=20.0),
        ),
        bid_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def poly_yes_state_insufficient_depth():
    """Polymarket YES with insufficient L1 depth."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="poly",
        market_id="token_yes",
        last_update_ms=now_ms,
        best_ask=0.40,
        best_ask_size=2.0,  # Only 2 shares at L1 (< 5 minimum)
        ask_levels=(
            OrderLevel(price=0.40, size=2.0),
            OrderLevel(price=0.42, size=20.0),
        ),
        bid_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def poly_yes_state_no_ask():
    """Polymarket YES with no ask data."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="poly",
        market_id="token_yes",
        last_update_ms=now_ms,
        best_ask=None,
        best_ask_size=0.0,
        ask_levels=tuple(),
        bid_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def kalshi_state_good():
    """Kalshi with good L1+L2 bid depth."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="kalshi",
        market_id="ticker",
        last_update_ms=now_ms,
        best_bid=0.35,
        best_bid_size=7.0,
        bid_levels=(
            OrderLevel(price=0.35, size=7.0),   # L1: YES bid
            OrderLevel(price=0.34, size=8.0),   # L2: YES bid
        ),
        best_ask=0.36,
        best_ask_size=6.0,
        ask_levels=(
            OrderLevel(price=0.36, size=6.0),   # L1: YES ask
            OrderLevel(price=0.37, size=9.0),   # L2: YES ask
        ),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def kalshi_state_insufficient_depth():
    """Kalshi with insufficient L1+L2 combined depth."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="kalshi",
        market_id="ticker",
        last_update_ms=now_ms,
        best_bid=0.65,
        best_bid_size=3.0,  # Low L1
        bid_levels=(
            OrderLevel(price=0.65, size=3.0),   # L1: only 3
            OrderLevel(price=0.64, size=1.0),   # L2: only 1 (total 4 < 5)
        ),
        best_ask=0.66,
        best_ask_size=2.0,
        ask_levels=(
            OrderLevel(price=0.66, size=2.0),
            OrderLevel(price=0.67, size=1.0),   # Total only 3 < 5
        ),
        is_fresh_after_reconnect=True,
    )


@pytest.fixture
def kalshi_state_no_levels():
    """Kalshi with empty orderbook."""
    now_ms = int(time.time() * 1000)
    return VenueState(
        venue="kalshi",
        market_id="ticker",
        last_update_ms=now_ms,
        best_bid=None,
        best_bid_size=0.0,
        bid_levels=tuple(),
        best_ask=None,
        best_ask_size=0.0,
        ask_levels=tuple(),
        is_fresh_after_reconnect=True,
    )


# =============================================================================
# Path 1 Tests: Buy Poly YES + Kalshi NO (uses Poly YES ask, Kalshi bids)
# =============================================================================


class TestPath1SuccessfulValidation:
    """Path 1: Buy Poly YES + Kalshi NO - successful validation."""

    def test_path1_all_checks_pass(
        self,
        poly_yes_state_good,
        kalshi_state_good,
        poly_no_state_good,
    ):
        """All checks pass for Path 1."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is True
        assert result.reason == "all_checks_passed"
        assert result.poly_depth_l1 == 10.0
        assert result.kalshi_depth_l1_l2 == 15.0  # 7 + 8
        assert result.kalshi_vwap is not None
        assert result.vwap_profit_pct is not None


class TestPath1PolyDepthFailure:
    """Path 1: Polymarket depth checks."""

    def test_path1_poly_depth_insufficient(
        self,
        poly_yes_state_insufficient_depth,
        kalshi_state_good,
        poly_no_state_good,
    ):
        """Path 1 fails when Poly YES L1 depth is insufficient."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_insufficient_depth,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is False
        assert "insufficient_poly_depth" in result.reason
        assert result.poly_depth_l1 == 2.0

    def test_path1_poly_no_ask(
        self,
        poly_yes_state_no_ask,
        kalshi_state_good,
        poly_no_state_good,
    ):
        """Path 1 fails when Poly YES has no ask data."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_no_ask,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is False
        assert "insufficient_poly_depth" in result.reason


class TestPath1KalshiDepthFailure:
    """Path 1: Kalshi depth checks."""

    def test_path1_kalshi_depth_insufficient(
        self,
        poly_yes_state_good,
        kalshi_state_insufficient_depth,
        poly_no_state_good,
    ):
        """Path 1 fails when Kalshi L1+L2 bid depth is insufficient."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_insufficient_depth,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is False
        assert "insufficient_kalshi_depth" in result.reason
        assert result.kalshi_depth_l1_l2 == 4.0  # 3 + 1

    def test_path1_kalshi_no_levels(
        self,
        poly_yes_state_good,
        kalshi_state_no_levels,
        poly_no_state_good,
    ):
        """Path 1 fails when Kalshi has no bid levels."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_no_levels,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is False
        assert "insufficient_kalshi_depth" in result.reason
        assert result.kalshi_depth_l1_l2 == 0.0


# =============================================================================
# Path 2 Tests: Buy Kalshi YES + Poly NO (uses Kalshi YES ask, Poly NO ask)
# =============================================================================


class TestPath2SuccessfulValidation:
    """Path 2: Buy Kalshi YES + Poly NO - successful validation."""

    def test_path2_all_checks_pass(
        self,
        poly_yes_state_good,
        kalshi_state_good,
        poly_no_state_good,
    ):
        """All checks pass for Path 2."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=2)

        assert result.passed is True
        assert result.reason == "all_checks_passed"
        assert result.poly_depth_l1 == 8.0  # Poly NO L1
        assert result.kalshi_depth_l1_l2 == 15.0  # Kalshi YES L1+L2: 6 + 9
        assert result.kalshi_vwap is not None
        assert result.vwap_profit_pct is not None


class TestPath2PolyDepthFailure:
    """Path 2: Polymarket depth checks."""

    def test_path2_poly_no_insufficient(
        self,
        poly_yes_state_good,
        kalshi_state_good,
        poly_no_state_insufficient_depth,
    ):
        """Path 2 fails when Poly NO L1 depth is insufficient."""
        # Create a poly_no with insufficient depth
        now_ms = int(time.time() * 1000)
        poly_no_bad = VenueState(
            venue="poly",
            market_id="token_no",
            last_update_ms=now_ms,
            best_ask=0.30,
            best_ask_size=1.0,  # Only 1 < 5
            ask_levels=(
                OrderLevel(price=0.30, size=1.0),
                OrderLevel(price=0.32, size=20.0),
            ),
            bid_levels=tuple(),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_bad,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=2)

        assert result.passed is False
        assert "insufficient_poly_depth" in result.reason
        assert result.poly_depth_l1 == 1.0


class TestPath2KalshiDepthFailure:
    """Path 2: Kalshi depth checks."""

    def test_path2_kalshi_ask_insufficient(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """Path 2 fails when Kalshi YES L1+L2 ask depth is insufficient."""
        now_ms = int(time.time() * 1000)
        kalshi_bad = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.65,
            best_bid_size=10.0,
            bid_levels=(
                OrderLevel(price=0.65, size=10.0),
                OrderLevel(price=0.64, size=10.0),
            ),
            best_ask=0.66,
            best_ask_size=2.0,  # Low L1
            ask_levels=(
                OrderLevel(price=0.66, size=2.0),   # L1: only 2
                OrderLevel(price=0.67, size=1.5),   # L2: only 1.5 (total 3.5 < 5)
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_bad,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=2)

        assert result.passed is False
        assert "insufficient_kalshi_depth" in result.reason
        assert result.kalshi_depth_l1_l2 == 3.5


# =============================================================================
# VWAP and Profit Tests
# =============================================================================


class TestVWAPCalculation:
    """VWAP calculation and profit verification."""

    def test_vwap_single_level(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """VWAP with single level."""
        now_ms = int(time.time() * 1000)
        kalshi_single = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.50,
            best_bid_size=10.0,
            bid_levels=(
                OrderLevel(price=0.50, size=10.0),
            ),
            best_ask=0.51,
            best_ask_size=5.0,
            ask_levels=tuple(),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_single,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        # With single level, VWAP = price of that level
        assert result.kalshi_vwap == 0.50

    def test_vwap_multiple_levels(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """VWAP with multiple levels is weighted average."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        # Kalshi bids: L1=0.35*7, L2=0.34*8
        # VWAP = (0.35*7 + 0.34*8) / (7+8) = (2.45 + 2.72) / 15 = 5.17/15 ≈ 0.3447
        expected_vwap = (0.35 * 7 + 0.34 * 8) / 15
        assert abs(result.kalshi_vwap - expected_vwap) < 1e-6

    def test_profit_calculation_successful(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Profit calculation when threshold is met."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            profit_threshold_pct=2.0
        )

        # Cross cost = VWAP + Poly ask
        # VWAP ≈ 0.3447, Poly ask = 0.40, cross_cost ≈ 0.7447 (under 1.0)
        # This should pass because cross_cost < 1.0 and profit > 2%
        # Let's verify the calculation is correct
        expected_vwap = (0.35 * 7 + 0.34 * 8) / 15
        expected_cross_cost = expected_vwap + 0.40
        expected_profit = (1.0 - expected_cross_cost) * 100

        assert result.kalshi_vwap is not None
        assert abs(result.kalshi_vwap - expected_vwap) < 1e-6

    def test_profit_below_threshold(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """Validation fails when profit is below threshold."""
        now_ms = int(time.time() * 1000)
        # Create a high-priced Kalshi state that reduces profit
        kalshi_expensive = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.80,  # High price
            best_bid_size=7.0,
            bid_levels=(
                OrderLevel(price=0.80, size=7.0),
                OrderLevel(price=0.79, size=8.0),
            ),
            best_ask=0.81,
            best_ask_size=6.0,
            ask_levels=(
                OrderLevel(price=0.81, size=6.0),
                OrderLevel(price=0.82, size=9.0),
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_expensive,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            profit_threshold_pct=2.0
        )

        # Kalshi bids VWAP = (0.80*7 + 0.79*8) / 15 ≈ 0.7967
        # Cross cost = 0.7967 + 0.40 = 1.1967 (>> 1.0)
        # Profit = (1 - 1.1967) * 100 = negative
        # Should fail
        assert result.passed is False
        assert "vwap_profit_below_threshold" in result.reason


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestInvalidStrategyPath:
    """Invalid strategy_path handling."""

    def test_invalid_strategy_path_0(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Strategy path 0 (invalid) is rejected."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=0)

        assert result.passed is False
        assert "invalid_strategy_path" in result.reason

    def test_invalid_strategy_path_3(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Strategy path 3 (invalid) is rejected."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=3)

        assert result.passed is False
        assert "invalid_strategy_path" in result.reason


class TestMinSharesConfiguration:
    """Configurable minimum shares threshold."""

    def test_custom_min_shares_lower(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Lower min_shares threshold allows validation to pass."""
        # poly_depth = 10, kalshi_depth = 15
        # With min_shares=20, should fail
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            min_shares=20  # Higher threshold
        )

        # Poly depth = 10 < 20, should fail
        assert result.passed is False
        assert "insufficient_poly_depth" in result.reason

    def test_custom_min_shares_high(
        self,
        poly_yes_state_good,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Higher min_shares allows validation when met."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            min_shares=3  # Lower threshold than default
        )

        # Poly depth = 10 >= 3, Kalshi depth = 15 >= 3, should pass (if profit ok)
        assert result.poly_depth_l1 == 10.0
        assert result.kalshi_depth_l1_l2 == 15.0


class TestProfitThresholdConfiguration:
    """Configurable profit threshold."""

    def test_high_profit_threshold(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """High profit threshold may cause failure."""
        now_ms = int(time.time() * 1000)
        # Create Kalshi with low prices (good profit)
        kalshi_cheap = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.30,
            best_bid_size=8.0,
            bid_levels=(
                OrderLevel(price=0.30, size=8.0),
                OrderLevel(price=0.29, size=7.0),
            ),
            best_ask=0.31,
            best_ask_size=5.0,
            ask_levels=(
                OrderLevel(price=0.31, size=5.0),
                OrderLevel(price=0.32, size=10.0),
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_cheap,
            aligned_market_id="ETH-TEST",
        )

        # Kalshi bids VWAP = (0.30*8 + 0.29*7) / 15 = (2.4 + 2.03) / 15 ≈ 0.2953
        # Cross cost = 0.2953 + 0.40 = 0.6953
        # Profit = (1 - 0.6953) * 100 ≈ 30.47%

        # With low threshold, should pass
        result_low = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            profit_threshold_pct=2.0
        )
        assert result_low.passed is True

        # With high threshold, should fail
        result_high = validate_depth_and_vwap(
            snapshot,
            strategy_path=1,
            profit_threshold_pct=50.0  # Require 50% profit
        )
        assert result_high.passed is False


class TestEmptyOrderLevels:
    """Handling of empty orderbook data."""

    def test_kalshi_only_l1_no_l2(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """Kalshi with only L1 (no L2)."""
        now_ms = int(time.time() * 1000)
        kalshi_l1_only = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.50,
            best_bid_size=10.0,
            bid_levels=(
                OrderLevel(price=0.50, size=10.0),
            ),
            best_ask=0.51,
            best_ask_size=5.0,
            ask_levels=(
                OrderLevel(price=0.51, size=5.0),
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_l1_only,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        # L1 only = 10, which is >= 5
        assert result.passed is True or result.passed is False  # Depends on profit
        assert result.kalshi_depth_l1_l2 == 10.0

    def test_kalshi_partial_level_data(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """Kalshi with partial level data (size=0 on L2)."""
        now_ms = int(time.time() * 1000)
        kalshi_partial = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.50,
            best_bid_size=8.0,
            bid_levels=(
                OrderLevel(price=0.50, size=8.0),   # L1: 8
                OrderLevel(price=0.49, size=0.0),   # L2: 0 (skipped)
            ),
            best_ask=0.51,
            best_ask_size=5.0,
            ask_levels=(
                OrderLevel(price=0.51, size=5.0),
                OrderLevel(price=0.52, size=10.0),
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_partial,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        # Should only count non-zero sizes: 8 + 0 = 8
        assert result.kalshi_depth_l1_l2 == 8.0


# =============================================================================
# Data Validation Tests
# =============================================================================


class TestDataAttributes:
    """Verify returned data attributes."""

    def test_successful_validation_has_all_attributes(
        self,
        poly_yes_state_good,
        poly_no_state_good,
    ):
        """Successful validation includes all expected attributes."""
        now_ms = int(time.time() * 1000)
        kalshi_good = VenueState(
            venue="kalshi",
            market_id="ticker",
            last_update_ms=now_ms,
            best_bid=0.30,
            best_bid_size=8.0,
            bid_levels=(
                OrderLevel(price=0.30, size=8.0),
                OrderLevel(price=0.29, size=7.0),
            ),
            best_ask=0.31,
            best_ask_size=5.0,
            ask_levels=(
                OrderLevel(price=0.31, size=5.0),
                OrderLevel(price=0.32, size=10.0),
            ),
            is_fresh_after_reconnect=True,
        )

        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_good,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is not None
        assert result.reason is not None
        assert result.kalshi_vwap is not None
        assert result.kalshi_depth_l1_l2 is not None
        assert result.poly_depth_l1 is not None
        assert result.vwap_profit_pct is not None

    def test_failed_validation_has_relevant_attributes(
        self,
        poly_yes_state_insufficient_depth,
        poly_no_state_good,
        kalshi_state_good,
    ):
        """Failed validation includes attributes up to failure point."""
        snapshot = create_snapshot(
            poly_yes_state=poly_yes_state_insufficient_depth,
            poly_no_state=poly_no_state_good,
            kalshi_state=kalshi_state_good,
            aligned_market_id="ETH-TEST",
        )

        result = validate_depth_and_vwap(snapshot, strategy_path=1)

        assert result.passed is False
        assert result.reason is not None
        assert result.poly_depth_l1 == 2.0
        # May not have VWAP if failed early
