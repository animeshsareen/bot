"""Tests for venue_state module.

Tests cover:
- VenueState creation and validation
- OrderLevel creation and validation
- MarketSnapshot creation and validity checking
- Staleness detection
- Orderbook parsing for Kalshi and Polymarket
"""

import pytest
import time

from src.venue_state import (
    OrderLevel,
    VenueState,
    MarketSnapshot,
    create_venue_state,
    create_snapshot,
    parse_kalshi_orderbook,
    parse_poly_orderbook,
    DEFAULT_STALE_MS,
)


class TestOrderLevel:
    """Tests for OrderLevel dataclass."""

    def test_create_valid_order_level(self):
        level = OrderLevel(price=0.55, size=100.0)
        assert level.price == 0.55
        assert level.size == 100.0

    def test_order_level_is_frozen(self):
        level = OrderLevel(price=0.55, size=100.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            level.price = 0.60

    def test_negative_price_raises(self):
        with pytest.raises(ValueError, match="price must be non-negative"):
            OrderLevel(price=-0.1, size=100.0)

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="size must be non-negative"):
            OrderLevel(price=0.55, size=-10.0)

    def test_zero_price_allowed(self):
        level = OrderLevel(price=0.0, size=50.0)
        assert level.price == 0.0

    def test_zero_size_allowed(self):
        level = OrderLevel(price=0.55, size=0.0)
        assert level.size == 0.0


class TestVenueState:
    """Tests for VenueState dataclass."""

    def test_create_poly_state(self):
        state = VenueState(venue="poly", market_id="token123")
        assert state.venue == "poly"
        assert state.market_id == "token123"
        assert state.last_update_ms == 0
        assert state.best_bid is None
        assert state.best_ask is None

    def test_create_kalshi_state(self):
        state = VenueState(venue="kalshi", market_id="TICKER-001")
        assert state.venue == "kalshi"
        assert state.market_id == "TICKER-001"

    def test_invalid_venue_raises(self):
        with pytest.raises(ValueError, match="venue must be 'poly' or 'kalshi'"):
            VenueState(venue="invalid", market_id="token123")

    def test_empty_market_id_raises(self):
        with pytest.raises(ValueError, match="market_id must not be empty"):
            VenueState(venue="poly", market_id="")

    def test_negative_last_update_raises(self):
        with pytest.raises(ValueError, match="last_update_ms must be non-negative"):
            VenueState(venue="poly", market_id="token123", last_update_ms=-1)

    def test_venue_state_is_frozen(self):
        state = VenueState(venue="poly", market_id="token123")
        with pytest.raises(Exception):
            state.venue = "kalshi"

    def test_with_orderbook_updates_state(self):
        state = VenueState(venue="poly", market_id="token123")
        bid_levels = [OrderLevel(price=0.50, size=100.0)]
        ask_levels = [OrderLevel(price=0.55, size=50.0)]

        new_state = state.with_orderbook(bid_levels, ask_levels, update_ms=1000)

        assert new_state.best_bid == 0.50
        assert new_state.best_ask == 0.55
        assert new_state.best_bid_size == 100.0
        assert new_state.best_ask_size == 50.0
        assert new_state.last_update_ms == 1000
        assert len(new_state.bid_levels) == 1
        assert len(new_state.ask_levels) == 1

    def test_with_orderbook_preserves_venue_and_market(self):
        state = VenueState(venue="kalshi", market_id="TICKER-001")
        bid_levels = [OrderLevel(price=0.50, size=100.0)]
        ask_levels = [OrderLevel(price=0.55, size=50.0)]

        new_state = state.with_orderbook(bid_levels, ask_levels, update_ms=1000)

        assert new_state.venue == "kalshi"
        assert new_state.market_id == "TICKER-001"

    def test_with_orderbook_marks_fresh_after_reconnect(self):
        state = VenueState(
            venue="poly", market_id="token123", is_fresh_after_reconnect=False
        )
        bid_levels = [OrderLevel(price=0.50, size=100.0)]

        new_state = state.with_orderbook(bid_levels, [], update_ms=1000)

        assert new_state.is_fresh_after_reconnect is True

    def test_with_orderbook_empty_levels(self):
        state = VenueState(venue="poly", market_id="token123")
        new_state = state.with_orderbook([], [], update_ms=1000)

        assert new_state.best_bid is None
        assert new_state.best_ask is None
        assert new_state.best_bid_size == 0.0
        assert new_state.best_ask_size == 0.0

    def test_mark_reconnecting(self):
        state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=1000,
            best_bid=0.50,
            is_fresh_after_reconnect=True,
        )

        new_state = state.mark_reconnecting()

        assert new_state.is_fresh_after_reconnect is False
        # Other fields preserved
        assert new_state.last_update_ms == 1000
        assert new_state.best_bid == 0.50

    def test_is_stale_false_when_fresh(self):
        now_ms = 10000
        state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,  # 100ms ago
            is_fresh_after_reconnect=True,
        )
        assert state.is_stale(now_ms=now_ms, stale_ms=500) is False

    def test_is_stale_true_when_old(self):
        now_ms = 10000
        state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 600,  # 600ms ago
            is_fresh_after_reconnect=True,
        )
        assert state.is_stale(now_ms=now_ms, stale_ms=500) is True

    def test_is_stale_true_when_not_fresh_after_reconnect(self):
        now_ms = 10000
        state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,  # Recent but not fresh
            is_fresh_after_reconnect=False,
        )
        assert state.is_stale(now_ms=now_ms, stale_ms=500) is True

    def test_total_bid_depth(self):
        state = VenueState(
            venue="poly",
            market_id="token123",
            bid_levels=(
                OrderLevel(price=0.50, size=100.0),
                OrderLevel(price=0.49, size=50.0),
                OrderLevel(price=0.48, size=25.0),
            ),
        )
        assert state.total_bid_depth() == 175.0
        assert state.total_bid_depth(max_levels=2) == 150.0

    def test_total_ask_depth(self):
        state = VenueState(
            venue="poly",
            market_id="token123",
            ask_levels=(
                OrderLevel(price=0.55, size=80.0),
                OrderLevel(price=0.56, size=40.0),
            ),
        )
        assert state.total_ask_depth() == 120.0
        assert state.total_ask_depth(max_levels=1) == 80.0


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_create_snapshot(self):
        poly_state = VenueState(venue="poly", market_id="token123")
        kalshi_state = VenueState(venue="kalshi", market_id="TICKER-001")

        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=1000,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.poly_state == poly_state
        assert snapshot.kalshi_state == kalshi_state
        assert snapshot.snapshot_ms == 1000
        assert snapshot.aligned_market_id == "ETH-15M-123"

    def test_snapshot_wrong_poly_venue_raises(self):
        poly_state = VenueState(venue="kalshi", market_id="WRONG")
        kalshi_state = VenueState(venue="kalshi", market_id="TICKER-001")

        with pytest.raises(ValueError, match="poly_state must have venue='poly'"):
            MarketSnapshot(
                poly_state=poly_state,
                kalshi_state=kalshi_state,
                snapshot_ms=1000,
                aligned_market_id="ETH-15M-123",
            )

    def test_snapshot_wrong_kalshi_venue_raises(self):
        poly_state = VenueState(venue="poly", market_id="token123")
        kalshi_state = VenueState(venue="poly", market_id="WRONG")

        with pytest.raises(ValueError, match="kalshi_state must have venue='kalshi'"):
            MarketSnapshot(
                poly_state=poly_state,
                kalshi_state=kalshi_state,
                snapshot_ms=1000,
                aligned_market_id="ETH-15M-123",
            )

    def test_is_valid_true_when_both_fresh(self):
        now_ms = 10000
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 200,
            is_fresh_after_reconnect=True,
        )

        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is True

    def test_is_valid_false_when_poly_stale(self):
        now_ms = 10000
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 600,  # Stale
            is_fresh_after_reconnect=True,
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )

        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is False

    def test_is_valid_false_when_kalshi_stale(self):
        now_ms = 10000
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 600,  # Stale
            is_fresh_after_reconnect=True,
        )

        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is False

    def test_is_valid_false_when_poly_not_fresh_after_reconnect(self):
        now_ms = 10000
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=False,  # Reconnecting
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )

        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is False


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_venue_state_poly(self):
        state = create_venue_state("poly", "token123")
        assert state.venue == "poly"
        assert state.market_id == "token123"
        assert state.last_update_ms == 0
        assert state.is_fresh_after_reconnect is False  # Not fresh until first update

    def test_create_venue_state_kalshi(self):
        state = create_venue_state("kalshi", "TICKER-001")
        assert state.venue == "kalshi"
        assert state.market_id == "TICKER-001"

    def test_create_snapshot_function(self):
        poly_state = VenueState(venue="poly", market_id="token123")
        kalshi_state = VenueState(venue="kalshi", market_id="TICKER-001")

        snapshot = create_snapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            aligned_market_id="ETH-15M-123",
            snapshot_ms=1000,
        )

        assert snapshot.poly_state == poly_state
        assert snapshot.kalshi_state == kalshi_state
        assert snapshot.aligned_market_id == "ETH-15M-123"
        assert snapshot.snapshot_ms == 1000


class TestParseKalshiOrderbook:
    """Tests for Kalshi orderbook parsing."""

    def test_parse_empty_orderbook(self):
        bid_levels, ask_levels = parse_kalshi_orderbook([], [])
        assert bid_levels == []
        assert ask_levels == []

    def test_parse_yes_bids(self):
        yes_bids = [("0.50", "100.0"), ("0.45", "50.0")]
        bid_levels, _ = parse_kalshi_orderbook(yes_bids, [])

        assert len(bid_levels) == 2
        # Sorted best (highest) to worst
        assert bid_levels[0].price == 0.50
        assert bid_levels[0].size == 100.0
        assert bid_levels[1].price == 0.45
        assert bid_levels[1].size == 50.0

    def test_parse_no_bids_as_yes_asks(self):
        # NO bids become YES asks with inverted price
        no_bids = [("0.40", "80.0"), ("0.45", "60.0")]
        _, ask_levels = parse_kalshi_orderbook([], no_bids)

        assert len(ask_levels) == 2
        # YES ask = 1 - NO bid price
        # Sorted best (lowest) to worst
        assert ask_levels[0].price == pytest.approx(0.55)  # 1 - 0.45
        assert ask_levels[0].size == 60.0
        assert ask_levels[1].price == pytest.approx(0.60)  # 1 - 0.40
        assert ask_levels[1].size == 80.0

    def test_parse_skips_zero_size(self):
        yes_bids = [("0.50", "100.0"), ("0.45", "0.0")]
        bid_levels, _ = parse_kalshi_orderbook(yes_bids, [])

        assert len(bid_levels) == 1
        assert bid_levels[0].price == 0.50

    def test_parse_full_orderbook(self):
        yes_bids = [("0.50", "100.0"), ("0.48", "50.0")]
        no_bids = [("0.45", "80.0"), ("0.42", "40.0")]

        bid_levels, ask_levels = parse_kalshi_orderbook(yes_bids, no_bids)

        assert len(bid_levels) == 2
        assert len(ask_levels) == 2
        assert bid_levels[0].price == 0.50
        assert ask_levels[0].price == pytest.approx(0.55)  # 1 - 0.45


class TestParsePolyOrderbook:
    """Tests for Polymarket orderbook parsing."""

    def test_parse_empty_orderbook(self):
        bid_levels, ask_levels = parse_poly_orderbook([], [])
        assert bid_levels == []
        assert ask_levels == []

    def test_parse_bids(self):
        bids = [
            {"price": "0.50", "size": "100.0"},
            {"price": "0.45", "size": "50.0"},
        ]
        bid_levels, _ = parse_poly_orderbook(bids, [])

        assert len(bid_levels) == 2
        # Sorted best (highest) to worst
        assert bid_levels[0].price == 0.50
        assert bid_levels[0].size == 100.0
        assert bid_levels[1].price == 0.45
        assert bid_levels[1].size == 50.0

    def test_parse_asks(self):
        asks = [
            {"price": "0.55", "size": "80.0"},
            {"price": "0.60", "size": "40.0"},
        ]
        _, ask_levels = parse_poly_orderbook([], asks)

        assert len(ask_levels) == 2
        # Sorted best (lowest) to worst
        assert ask_levels[0].price == 0.55
        assert ask_levels[0].size == 80.0
        assert ask_levels[1].price == 0.60
        assert ask_levels[1].size == 40.0

    def test_parse_skips_zero_size(self):
        bids = [
            {"price": "0.50", "size": "100.0"},
            {"price": "0.45", "size": "0.0"},
        ]
        bid_levels, _ = parse_poly_orderbook(bids, [])

        assert len(bid_levels) == 1
        assert bid_levels[0].price == 0.50

    def test_parse_full_orderbook(self):
        bids = [
            {"price": "0.50", "size": "100.0"},
            {"price": "0.48", "size": "50.0"},
        ]
        asks = [
            {"price": "0.55", "size": "80.0"},
            {"price": "0.58", "size": "40.0"},
        ]

        bid_levels, ask_levels = parse_poly_orderbook(bids, asks)

        assert len(bid_levels) == 2
        assert len(ask_levels) == 2
        assert bid_levels[0].price == 0.50
        assert ask_levels[0].price == 0.55


class TestDefaultStaleMs:
    """Test default staleness threshold matches PRD config."""

    def test_default_stale_ms(self):
        assert DEFAULT_STALE_MS == 500  # From PRD config stale_ms
