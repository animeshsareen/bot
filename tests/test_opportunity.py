"""Tests for naive opportunity detection module."""

import pytest
from src.opportunity import (
    OpportunityTrigger,
    OpportunityDetector,
    CooldownTracker,
    compute_cross_cost_yes_no,
    compute_raw_profit_pct,
    DEFAULT_COOLDOWN_MS,
    DEFAULT_COOLDOWN_IMPROVEMENT_THRESHOLD,
    DEFAULT_RAW_PROFIT_TRIGGER_PCT,
    DEFAULT_MAX_COMBINED_COST_CAP,
)
from src.venue_state import VenueState, MarketSnapshot, OrderLevel


# =============================================================================
# Test Fixtures
# =============================================================================


def make_venue_state(
    venue: str,
    market_id: str,
    best_ask: float | None,
    best_bid: float | None = None,
    update_ms: int = 1000,
    ask_size: float = 100.0,
    bid_size: float = 100.0,
) -> VenueState:
    """Create a VenueState for testing."""
    bid_levels = []
    ask_levels = []
    if best_bid is not None:
        bid_levels.append(OrderLevel(price=best_bid, size=bid_size))
    if best_ask is not None:
        ask_levels.append(OrderLevel(price=best_ask, size=ask_size))

    return VenueState(
        venue=venue,
        market_id=market_id,
        last_update_ms=update_ms,
        best_bid=best_bid,
        best_ask=best_ask,
        best_bid_size=bid_size if best_bid else 0.0,
        best_ask_size=ask_size if best_ask else 0.0,
        bid_levels=tuple(bid_levels),
        ask_levels=tuple(ask_levels),
        is_fresh_after_reconnect=True,
    )


def make_snapshot(
    poly_ask: float | None,
    kalshi_ask: float | None,
    poly_no_ask: float | None = None,
    aligned_market_id: str = "test_market",
    snapshot_ms: int = 1000,
) -> MarketSnapshot:
    """Create a MarketSnapshot for testing.

    Creates a 3-leg snapshot with separate YES/NO legs for Polymarket.
    """
    poly_yes_state = make_venue_state("poly", "poly_yes_token", poly_ask, 0.40)
    # If poly_no_ask not specified, default to inverse of poly_yes_ask for testing
    if poly_no_ask is None:
        poly_no_ask = 1.0 - (poly_ask or 0.50)
    poly_no_state = make_venue_state("poly", "poly_no_token", poly_no_ask, 0.40)
    kalshi_state = make_venue_state("kalshi", "kalshi_ticker", kalshi_ask, 0.40)
    return MarketSnapshot(
        poly_yes_state=poly_yes_state,
        poly_no_state=poly_no_state,
        kalshi_state=kalshi_state,
        snapshot_ms=snapshot_ms,
        aligned_market_id=aligned_market_id,
    )


# =============================================================================
# Tests for compute_cross_cost
# =============================================================================


class TestComputeCrossCost:
    """Tests for cross_cost computation."""

    def test_basic_cross_cost(self):
        """Cross cost is sum of best asks."""
        assert compute_cross_cost_yes_no(0.45, 0.50) == 0.95

    def test_low_cross_cost(self):
        """Low asks result in low cross cost."""
        assert compute_cross_cost_yes_no(0.30, 0.30) == 0.60

    def test_high_cross_cost(self):
        """High asks result in high cross cost."""
        assert compute_cross_cost_yes_no(0.55, 0.55) == 1.10

    def test_cross_cost_precision(self):
        """Cross cost maintains precision."""
        result = compute_cross_cost_yes_no(0.485, 0.495)
        assert abs(result - 0.98) < 0.0001


# =============================================================================
# Tests for compute_raw_profit_pct
# =============================================================================


class TestComputeRawProfitPct:
    """Tests for raw profit percentage computation."""

    def test_basic_raw_profit(self):
        """Raw profit is (1 - cross_cost) * 100."""
        assert abs(compute_raw_profit_pct(0.98) - 2.0) < 0.0001

    def test_zero_profit(self):
        """Cross cost of 1.0 means zero profit."""
        assert abs(compute_raw_profit_pct(1.0) - 0.0) < 0.0001

    def test_negative_profit(self):
        """Cross cost > 1.0 means negative profit."""
        assert abs(compute_raw_profit_pct(1.05) - (-5.0)) < 0.0001

    def test_high_profit(self):
        """Low cross cost means high profit."""
        assert abs(compute_raw_profit_pct(0.90) - 10.0) < 0.0001


# =============================================================================
# Tests for CooldownTracker
# =============================================================================


class TestCooldownTracker:
    """Tests for cooldown tracking."""

    def test_no_cooldown_initially(self):
        """No cooldown on first trigger."""
        tracker = CooldownTracker()
        assert not tracker.is_on_cooldown("test", 0.95, 1000)

    def test_cooldown_after_trigger(self):
        """Cooldown active after recording trigger."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("test", 0.95, trigger_ms=1000)
        # Check 100ms later - should be on cooldown
        assert tracker.is_on_cooldown("test", 0.95, 1100)

    def test_cooldown_expires(self):
        """Cooldown expires after cooldown_ms."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("test", 0.95, trigger_ms=1000)
        # Check 500ms later - cooldown should be over
        assert not tracker.is_on_cooldown("test", 0.95, 1500)

    def test_cooldown_bypassed_by_improvement(self):
        """Cooldown bypassed if cross_cost improves by threshold."""
        tracker = CooldownTracker(cooldown_ms=500, improvement_threshold=0.002)
        tracker.record_trigger("test", 0.95, trigger_ms=1000)
        # Improved by 0.003 (>= 0.002 threshold)
        assert not tracker.is_on_cooldown("test", 0.947, 1100)

    def test_cooldown_not_bypassed_below_threshold(self):
        """Cooldown not bypassed if improvement below threshold."""
        tracker = CooldownTracker(cooldown_ms=500, improvement_threshold=0.002)
        tracker.record_trigger("test", 0.95, trigger_ms=1000)
        # Improved by only 0.001 (< 0.002 threshold)
        assert tracker.is_on_cooldown("test", 0.949, 1100)

    def test_independent_strategies(self):
        """Cooldown tracked independently per strategy."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("strategy_a", 0.95, trigger_ms=1000)
        # Different strategy should not be on cooldown
        assert not tracker.is_on_cooldown("strategy_b", 0.95, 1100)
        # Original strategy should be on cooldown
        assert tracker.is_on_cooldown("strategy_a", 0.95, 1100)

    def test_clear_specific_strategy(self):
        """Can clear cooldown for specific strategy."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("strategy_a", 0.95, trigger_ms=1000)
        tracker.record_trigger("strategy_b", 0.95, trigger_ms=1000)
        tracker.clear("strategy_a")
        assert not tracker.is_on_cooldown("strategy_a", 0.95, 1100)
        assert tracker.is_on_cooldown("strategy_b", 0.95, 1100)

    def test_clear_all(self):
        """Can clear all cooldowns."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("strategy_a", 0.95, trigger_ms=1000)
        tracker.record_trigger("strategy_b", 0.95, trigger_ms=1000)
        tracker.clear()
        assert not tracker.is_on_cooldown("strategy_a", 0.95, 1100)
        assert not tracker.is_on_cooldown("strategy_b", 0.95, 1100)

    def test_default_cooldown_ms(self):
        """Default cooldown matches PRD config."""
        assert DEFAULT_COOLDOWN_MS == 500

    def test_default_improvement_threshold(self):
        """Default improvement threshold matches PRD."""
        assert DEFAULT_COOLDOWN_IMPROVEMENT_THRESHOLD == 0.002


# =============================================================================
# Tests for OpportunityDetector
# =============================================================================


class TestOpportunityDetector:
    """Tests for opportunity detection."""

    def test_trigger_on_raw_profit_threshold(self):
        """Triggers when raw_profit_pct >= raw_profit_trigger_pct."""
        detector = OpportunityDetector(raw_profit_trigger_pct=2.0)
        # cross_cost = 0.48 + 0.50 = 0.98, raw_profit = 2%
        snapshot = make_snapshot(0.48, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert result.triggered
        assert result.reason == "raw_profit_trigger"
        assert abs(result.cross_cost - 0.98) < 0.0001
        assert abs(result.raw_profit_pct - 2.0) < 0.0001

    def test_trigger_on_cost_cap(self):
        """Triggers when cross_cost <= max_combined_cost_cap."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=5.0,  # Very high, won't trigger on profit
            max_combined_cost_cap=0.99,
        )
        # cross_cost = 0.49 + 0.50 = 0.99, raw_profit = 1%
        snapshot = make_snapshot(0.49, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert result.triggered
        assert result.reason == "cost_cap_trigger"
        assert abs(result.cross_cost - 0.99) < 0.0001

    def test_no_trigger_below_threshold(self):
        """No trigger when neither condition met."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            max_combined_cost_cap=0.95,
        )
        # cross_cost = 0.50 + 0.50 = 1.00, raw_profit = 0%
        snapshot = make_snapshot(0.50, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered
        assert result.reason == "below_threshold"

    def test_no_trigger_missing_poly_ask(self):
        """No trigger when Polymarket ask missing."""
        detector = OpportunityDetector()
        snapshot = make_snapshot(None, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered
        assert result.reason == "missing_ask_or_bid"
        assert result.cross_cost is None
        assert result.poly_ask is None

    def test_no_trigger_missing_kalshi_ask(self):
        """No trigger when Kalshi ask missing."""
        detector = OpportunityDetector()
        snapshot = make_snapshot(0.50, None)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered
        assert result.reason == "missing_ask_or_bid"
        assert result.cross_cost is None
        assert result.kalshi_ask is None

    def test_cooldown_prevents_trigger(self):
        """Cooldown prevents re-trigger."""
        tracker = CooldownTracker(cooldown_ms=500)
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            cooldown_tracker=tracker,
        )
        snapshot = make_snapshot(0.48, 0.50)

        # First trigger succeeds
        result1 = detector.detect(snapshot, now_ms=1000)
        assert result1.triggered

        # Second trigger blocked by cooldown
        result2 = detector.detect(snapshot, now_ms=1100)
        assert not result2.triggered
        assert result2.reason == "cooldown"

    def test_cooldown_allows_after_expiry(self):
        """Trigger allowed after cooldown expires."""
        tracker = CooldownTracker(cooldown_ms=500)
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            cooldown_tracker=tracker,
        )
        snapshot = make_snapshot(0.48, 0.50)

        # First trigger
        result1 = detector.detect(snapshot, now_ms=1000)
        assert result1.triggered

        # After cooldown expires
        result2 = detector.detect(snapshot, now_ms=1600)
        assert result2.triggered

    def test_cooldown_allows_on_improvement(self):
        """Trigger allowed during cooldown if cross_cost improves."""
        tracker = CooldownTracker(cooldown_ms=500, improvement_threshold=0.002)
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            cooldown_tracker=tracker,
        )

        # First trigger at 0.98
        snapshot1 = make_snapshot(0.48, 0.50)
        result1 = detector.detect(snapshot1, now_ms=1000)
        assert result1.triggered

        # Better opportunity at 0.975 (improved by 0.005)
        snapshot2 = make_snapshot(0.475, 0.50)
        result2 = detector.detect(snapshot2, now_ms=1100)
        assert result2.triggered

    def test_strategy_id_from_aligned_market(self):
        """Strategy ID comes from aligned_market_id."""
        detector = OpportunityDetector()
        snapshot = make_snapshot(0.48, 0.50, aligned_market_id="eth_15m_123")
        result = detector.detect(snapshot, now_ms=1000)
        assert result.strategy_id == "eth_15m_123"

    def test_result_contains_trigger_inputs(self):
        """Result contains all trigger inputs for logging."""
        detector = OpportunityDetector()
        snapshot = make_snapshot(0.48, 0.50, snapshot_ms=5000)
        result = detector.detect(snapshot, now_ms=5000)

        assert result.poly_ask == 0.48
        assert result.kalshi_ask == 0.50
        assert result.snapshot_ms == 5000
        assert abs(result.cross_cost - 0.98) < 0.0001
        assert abs(result.raw_profit_pct - 2.0) < 0.0001

    def test_default_config_values(self):
        """Default config matches PRD values."""
        assert DEFAULT_RAW_PROFIT_TRIGGER_PCT == 2.0
        assert DEFAULT_MAX_COMBINED_COST_CAP == 0.99


# =============================================================================
# Tests for OpportunityTrigger
# =============================================================================


class TestOpportunityTrigger:
    """Tests for OpportunityTrigger dataclass."""

    def test_trigger_is_frozen(self):
        """OpportunityTrigger is immutable."""
        trigger = OpportunityTrigger(
            triggered=True,
            cross_cost=0.98,
            raw_profit_pct=2.0,
            reason="test",
            poly_ask=0.48,
            kalshi_ask=0.50,
            snapshot_ms=1000,
            strategy_id="test",
        )
        with pytest.raises(AttributeError):
            trigger.triggered = False

    def test_log_details_triggered(self, caplog):
        """Log details for triggered opportunity."""
        import logging
        caplog.set_level(logging.INFO)

        trigger = OpportunityTrigger(
            triggered=True,
            cross_cost=0.98,
            raw_profit_pct=2.0,
            reason="raw_profit_trigger",
            poly_ask=0.48,
            kalshi_ask=0.50,
            snapshot_ms=1000,
            strategy_id="test_market",
        )
        trigger.log_details()

        assert "TRIGGER" in caplog.text
        assert "cross_cost=0.9800" in caplog.text
        assert "raw_profit_pct=2.00%" in caplog.text
        assert "poly_ask=0.4800" in caplog.text
        assert "kalshi_ask=0.5000" in caplog.text
        assert "raw_profit_trigger" in caplog.text

    def test_log_details_skipped(self, caplog):
        """Log details for skipped opportunity."""
        import logging
        caplog.set_level(logging.INFO)

        trigger = OpportunityTrigger(
            triggered=False,
            cross_cost=1.02,
            raw_profit_pct=-2.0,
            reason="below_threshold",
            poly_ask=0.51,
            kalshi_ask=0.51,
            snapshot_ms=1000,
            strategy_id="test_market",
        )
        trigger.log_details()

        assert "SKIP" in caplog.text
        assert "below_threshold" in caplog.text

    def test_log_details_missing_asks(self, caplog):
        """Log details when asks are missing."""
        import logging
        caplog.set_level(logging.INFO)

        trigger = OpportunityTrigger(
            triggered=False,
            cross_cost=None,
            raw_profit_pct=None,
            reason="missing_ask_or_bid",
            poly_ask=None,
            kalshi_ask=None,
            snapshot_ms=1000,
            strategy_id="test_market",
        )
        trigger.log_details()

        assert "SKIP" in caplog.text
        assert "missing_ask_or_bid" in caplog.text
        assert "N/A" in caplog.text


# =============================================================================
# Tests for edge cases and boundaries
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_at_profit_threshold(self):
        """Triggers at exactly the profit threshold."""
        detector = OpportunityDetector(raw_profit_trigger_pct=2.0)
        # cross_cost = 0.98 exactly, raw_profit = 2.0% exactly
        snapshot = make_snapshot(0.48, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert result.triggered

    def test_just_below_profit_threshold(self):
        """Does not trigger just below profit threshold."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            max_combined_cost_cap=0.95,  # Won't trigger on cost cap
        )
        # cross_cost = 0.981, raw_profit = 1.9%
        snapshot = make_snapshot(0.481, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered

    def test_exactly_at_cost_cap(self):
        """Triggers at exactly the cost cap."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=5.0,  # High, won't trigger
            max_combined_cost_cap=0.99,
        )
        # cross_cost = 0.99 exactly
        snapshot = make_snapshot(0.49, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert result.triggered
        assert result.reason == "cost_cap_trigger"

    def test_just_above_cost_cap(self):
        """Does not trigger just above cost cap."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=5.0,
            max_combined_cost_cap=0.99,
        )
        # cross_cost = 0.991
        snapshot = make_snapshot(0.491, 0.50)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered

    def test_profit_trigger_takes_precedence(self):
        """Raw profit trigger reason used when both conditions met."""
        detector = OpportunityDetector(
            raw_profit_trigger_pct=2.0,
            max_combined_cost_cap=0.99,
        )
        # Both conditions met
        snapshot = make_snapshot(0.48, 0.50)  # cross_cost=0.98, profit=2%
        result = detector.detect(snapshot, now_ms=1000)
        assert result.triggered
        assert result.reason == "raw_profit_trigger"

    def test_negative_raw_profit(self):
        """Handles negative raw profit correctly."""
        detector = OpportunityDetector()
        # cross_cost = 1.10, raw_profit = -10%
        snapshot = make_snapshot(0.55, 0.55)
        result = detector.detect(snapshot, now_ms=1000)
        assert not result.triggered
        assert abs(result.raw_profit_pct - (-10.0)) < 0.0001

    def test_cooldown_exact_boundary(self):
        """Cooldown exactly at boundary."""
        tracker = CooldownTracker(cooldown_ms=500)
        tracker.record_trigger("test", 0.95, trigger_ms=1000)
        # Exactly at cooldown boundary (500ms later)
        assert not tracker.is_on_cooldown("test", 0.95, 1500)
        # Just before boundary (499ms later)
        assert tracker.is_on_cooldown("test", 0.95, 1499)

    def test_improvement_exact_threshold(self):
        """Improvement exactly at threshold bypasses cooldown."""
        tracker = CooldownTracker(cooldown_ms=500, improvement_threshold=0.002)
        tracker.record_trigger("test", 0.950, trigger_ms=1000)
        # Improvement of exactly 0.002
        assert not tracker.is_on_cooldown("test", 0.948, 1100)
        # Improvement of 0.00199 (just under)
        assert tracker.is_on_cooldown("test", 0.94801, 1100)
