"""Tests for ws_manager module.

Tests cover:
- Independent WS tasks per venue
- State updates with last_update_ms
- Snapshot validity based on freshness
- WS reconnect marking state as stale
- REST fallback when WS stale
"""

import pytest
import asyncio
import time

from src.venue_state import OrderLevel, VenueState, create_venue_state
from src.ws_manager import (
    WSConfig,
    VenueWSHandler,
    MockVenueWSHandler,
    MarketDataManager,
)


class TestWSConfig:
    """Tests for WSConfig dataclass."""

    def test_default_config(self):
        config = WSConfig()
        assert config.stale_ms == 500
        assert config.reconnect_delay_ms == 1000
        assert config.max_reconnect_delay_ms == 30000
        assert config.rest_timeout_ms == 300

    def test_custom_config(self):
        config = WSConfig(stale_ms=200, rest_timeout_ms=150)
        assert config.stale_ms == 200
        assert config.rest_timeout_ms == 150


class TestMockVenueWSHandler:
    """Tests for MockVenueWSHandler."""

    def test_initial_state(self):
        handler = MockVenueWSHandler("poly", "token123")
        state = handler.state
        assert state.venue == "poly"
        assert state.market_id == "token123"
        assert state.is_fresh_after_reconnect is False

    @pytest.mark.asyncio
    async def test_update_state_directly(self):
        handler = MockVenueWSHandler("poly", "token123")
        bid_levels = [OrderLevel(price=0.50, size=100.0)]
        ask_levels = [OrderLevel(price=0.55, size=50.0)]

        await handler.update_state_directly(bid_levels, ask_levels, update_ms=1000)

        state = handler.state
        assert state.best_bid == 0.50
        assert state.best_ask == 0.55
        assert state.last_update_ms == 1000
        assert state.is_fresh_after_reconnect is True

    @pytest.mark.asyncio
    async def test_process_message_updates_state(self):
        handler = MockVenueWSHandler("poly", "token123")
        bid_levels = [OrderLevel(price=0.50, size=100.0)]
        ask_levels = [OrderLevel(price=0.55, size=50.0)]

        new_state = await handler.process_message({
            "bid_levels": bid_levels,
            "ask_levels": ask_levels,
            "update_ms": 1000,
        })

        assert new_state is not None
        assert new_state.best_bid == 0.50
        assert new_state.best_ask == 0.55
        assert new_state.is_fresh_after_reconnect is True

    @pytest.mark.asyncio
    async def test_process_message_ignores_non_orderbook(self):
        handler = MockVenueWSHandler("poly", "token123")

        new_state = await handler.process_message({"type": "heartbeat"})

        assert new_state is None

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_returns_none_by_default(self):
        handler = MockVenueWSHandler("poly", "token123")
        result = await handler.fetch_rest_orderbook()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_returns_set_response(self):
        handler = MockVenueWSHandler("poly", "token123")
        bid_levels = [OrderLevel(price=0.50, size=100.0)]
        ask_levels = [OrderLevel(price=0.55, size=50.0)]
        handler.set_rest_response(bid_levels, ask_levels)

        result = await handler.fetch_rest_orderbook()

        assert result is not None
        assert result.best_bid == 0.50
        assert result.best_ask == 0.55


class TestVenueWSHandlerReconnect:
    """Tests for WS reconnect behavior."""

    @pytest.mark.asyncio
    async def test_mark_reconnecting_sets_not_fresh(self):
        handler = MockVenueWSHandler("poly", "token123")

        # First, make it fresh
        await handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=1000
        )
        assert handler.state.is_fresh_after_reconnect is True

        # Simulate reconnect via mark_reconnecting
        new_state = handler.state.mark_reconnecting()
        # Update handler state directly for test
        await handler._update_state(new_state)

        assert handler.state.is_fresh_after_reconnect is False

    @pytest.mark.asyncio
    async def test_fresh_update_after_reconnect_marks_fresh(self):
        handler = MockVenueWSHandler("poly", "token123")

        # Mark as reconnecting
        await handler._update_state(handler.state.mark_reconnecting())
        assert handler.state.is_fresh_after_reconnect is False

        # Fresh update should mark as fresh
        await handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=2000
        )

        assert handler.state.is_fresh_after_reconnect is True


class TestMarketDataManager:
    """Tests for MarketDataManager."""

    @pytest.fixture
    def manager(self):
        poly_yes_handler = MockVenueWSHandler("poly", "token123")
        poly_no_handler = MockVenueWSHandler("poly", "token124")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")
        return MarketDataManager(
            poly_yes_handler=poly_yes_handler,
            poly_no_handler=poly_no_handler,
            kalshi_handler=kalshi_handler,
            aligned_market_id="ETH-15M-123",
            config=WSConfig(stale_ms=500),
        )

    def test_get_snapshot_returns_snapshot(self, manager):
        snapshot = manager.get_snapshot(now_ms=1000)
        assert snapshot.aligned_market_id == "ETH-15M-123"
        assert snapshot.poly_yes_state.venue == "poly"
        assert snapshot.poly_no_state.venue == "poly"
        assert snapshot.kalshi_state.venue == "kalshi"

    def test_get_snapshot_with_fresh_data_is_valid(self, manager):
        now_ms = 10000

        # Set fresh data for all legs
        asyncio.get_event_loop().run_until_complete(
            manager.poly_yes_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [OrderLevel(price=0.55, size=50.0)],
                update_ms=now_ms - 100,
            )
        )
        asyncio.get_event_loop().run_until_complete(
            manager.poly_no_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [OrderLevel(price=0.55, size=50.0)],
                update_ms=now_ms - 100,
            )
        )
        asyncio.get_event_loop().run_until_complete(
            manager.kalshi_handler.update_state_directly(
                [OrderLevel(price=0.48, size=80.0)],
                [OrderLevel(price=0.52, size=40.0)],
                update_ms=now_ms - 200,
            )
        )

        snapshot = manager.get_snapshot(now_ms=now_ms)
        assert snapshot.is_valid(stale_ms=500) is True

    def test_get_snapshot_with_stale_poly_is_invalid(self, manager):
        now_ms = 10000

        # Poly YES is stale (per-leg freshness check)
        asyncio.get_event_loop().run_until_complete(
            manager.poly_yes_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [],
                update_ms=now_ms - 600,  # Stale
            )
        )
        # Poly NO is fresh
        asyncio.get_event_loop().run_until_complete(
            manager.poly_no_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [],
                update_ms=now_ms - 100,
            )
        )
        # Kalshi is fresh
        asyncio.get_event_loop().run_until_complete(
            manager.kalshi_handler.update_state_directly(
                [OrderLevel(price=0.48, size=80.0)],
                [],
                update_ms=now_ms - 100,
            )
        )

        snapshot = manager.get_snapshot(now_ms=now_ms)
        assert snapshot.is_valid(stale_ms=500) is False

    def test_get_snapshot_with_stale_kalshi_is_invalid(self, manager):
        now_ms = 10000

        # Poly YES and NO are fresh
        asyncio.get_event_loop().run_until_complete(
            manager.poly_yes_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [],
                update_ms=now_ms - 100,
            )
        )
        asyncio.get_event_loop().run_until_complete(
            manager.poly_no_handler.update_state_directly(
                [OrderLevel(price=0.50, size=100.0)],
                [],
                update_ms=now_ms - 100,
            )
        )
        # Kalshi is stale
        asyncio.get_event_loop().run_until_complete(
            manager.kalshi_handler.update_state_directly(
                [OrderLevel(price=0.48, size=80.0)],
                [],
                update_ms=now_ms - 600,  # Stale
            )
        )

        snapshot = manager.get_snapshot(now_ms=now_ms)
        assert snapshot.is_valid(stale_ms=500) is False

    @pytest.mark.asyncio
    async def test_get_fresh_snapshot_no_rest_when_ws_fresh(self, manager):
        now_ms = 10000

        # All legs have fresh WS data
        await manager.poly_yes_handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=now_ms - 100,
        )
        await manager.poly_no_handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=now_ms - 100,
        )
        await manager.kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 200,
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # Should be valid, no REST was needed
        assert snapshot.is_valid(stale_ms=500) is True
        assert snapshot.poly_yes_state.best_bid == 0.50
        assert snapshot.kalshi_state.best_bid == 0.48

    @pytest.mark.asyncio
    async def test_get_fresh_snapshot_uses_rest_fallback_when_stale(self, manager):
        now_ms = 10000

        # Poly YES WS is stale
        await manager.poly_yes_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],  # Old data
            [],
            update_ms=now_ms - 600,  # Stale
        )
        # Poly NO and Kalshi are fresh
        await manager.poly_no_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],
            [],
            update_ms=now_ms - 100,
        )
        await manager.kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 100,
        )

        # Set REST fallback response for poly_yes
        manager.poly_yes_handler.set_rest_response(
            [OrderLevel(price=0.50, size=100.0)],  # Fresh data
            [OrderLevel(price=0.55, size=50.0)],
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # REST fallback should have been used for poly_yes
        assert snapshot.poly_yes_state.best_bid == 0.50  # From REST
        assert snapshot.kalshi_state.best_bid == 0.48  # From WS

    @pytest.mark.asyncio
    async def test_get_fresh_snapshot_handles_rest_failure(self, manager):
        now_ms = 10000

        # All stale, but no REST response set (simulates failure)
        await manager.poly_yes_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],
            [],
            update_ms=now_ms - 600,
        )
        await manager.poly_no_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],
            [],
            update_ms=now_ms - 600,
        )
        await manager.kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 600,
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # REST failed, returns stale data (snapshot will be invalid)
        assert snapshot.poly_yes_state.best_bid == 0.45
        assert snapshot.kalshi_state.best_bid == 0.48
        # Note: snapshot may be marked invalid due to age

    @pytest.mark.asyncio
    async def test_start_and_stop(self, manager):
        poly_yes_task, poly_no_task, kalshi_task = manager.start()

        assert poly_yes_task is not None
        assert poly_no_task is not None
        assert kalshi_task is not None
        assert not poly_yes_task.done()
        assert not poly_no_task.done()
        assert not kalshi_task.done()

        await manager.stop()

        # Give tasks time to cancel
        await asyncio.sleep(0.2)
        assert poly_yes_task.done()
        assert poly_no_task.done()
        assert kalshi_task.done()


class TestIndependentWSTasks:
    """Tests for independent WS tasks per venue (US-003 AC1)."""

    @pytest.mark.asyncio
    async def test_venues_update_independently(self):
        poly_handler = MockVenueWSHandler("poly", "token123")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")

        # Update poly only
        await poly_handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=1000,
        )

        assert poly_handler.state.best_bid == 0.50
        assert poly_handler.state.last_update_ms == 1000

        # Kalshi unchanged
        assert kalshi_handler.state.best_bid is None
        assert kalshi_handler.state.last_update_ms == 0

        # Now update kalshi
        await kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=2000,
        )

        assert kalshi_handler.state.best_bid == 0.48
        assert kalshi_handler.state.last_update_ms == 2000

        # Poly still unchanged from its last update
        assert poly_handler.state.last_update_ms == 1000


class TestL1L2DepthTracking:
    """Tests for L1 + L2 depth tracking (US-003 AC2)."""

    @pytest.mark.asyncio
    async def test_tracks_multiple_bid_levels(self):
        handler = MockVenueWSHandler("poly", "token123")
        bid_levels = [
            OrderLevel(price=0.50, size=100.0),  # L1
            OrderLevel(price=0.49, size=50.0),   # L2
            OrderLevel(price=0.48, size=25.0),   # L2
        ]

        await handler.update_state_directly(bid_levels, [], update_ms=1000)

        state = handler.state
        assert state.best_bid == 0.50
        assert state.best_bid_size == 100.0
        assert len(state.bid_levels) == 3
        assert state.total_bid_depth() == 175.0
        assert state.total_bid_depth(max_levels=2) == 150.0

    @pytest.mark.asyncio
    async def test_tracks_multiple_ask_levels(self):
        handler = MockVenueWSHandler("poly", "token123")
        ask_levels = [
            OrderLevel(price=0.55, size=80.0),   # L1
            OrderLevel(price=0.56, size=40.0),   # L2
        ]

        await handler.update_state_directly([], ask_levels, update_ms=1000)

        state = handler.state
        assert state.best_ask == 0.55
        assert state.best_ask_size == 80.0
        assert len(state.ask_levels) == 2
        assert state.total_ask_depth() == 120.0


class TestSnapshotValidity:
    """Tests for snapshot validity based on freshness (US-003 AC3)."""

    def test_snapshot_valid_when_both_within_stale_ms(self):
        now_ms = 10000
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 400,  # 400ms ago
            is_fresh_after_reconnect=True,
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 300,  # 300ms ago
            is_fresh_after_reconnect=True,
        )

        from src.venue_state import MarketSnapshot
        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is True

    def test_snapshot_invalid_when_one_venue_stale(self):
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

        from src.venue_state import MarketSnapshot
        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",
        )

        assert snapshot.is_valid(stale_ms=500) is False

    def test_snapshot_invalid_with_different_market_freshness(self):
        """Snapshot must validate SAME aligned market, not mismatched."""
        now_ms = 10000
        # Both fresh but for same aligned market
        poly_state = VenueState(
            venue="poly",
            market_id="token123",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )
        kalshi_state = VenueState(
            venue="kalshi",
            market_id="TICKER-001",
            last_update_ms=now_ms - 100,
            is_fresh_after_reconnect=True,
        )

        from src.venue_state import MarketSnapshot
        snapshot = MarketSnapshot(
            poly_state=poly_state,
            kalshi_state=kalshi_state,
            snapshot_ms=now_ms,
            aligned_market_id="ETH-15M-123",  # Both for same aligned market
        )

        assert snapshot.is_valid(stale_ms=500) is True


class TestWSReconnectDiscardCache:
    """Tests for discarding cache on WS reconnect (US-003 AC4)."""

    @pytest.mark.asyncio
    async def test_reconnect_marks_state_stale(self):
        handler = MockVenueWSHandler("poly", "token123")

        # First update makes it fresh
        await handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=1000,
        )
        assert handler.state.is_fresh_after_reconnect is True

        # Simulate reconnect
        new_state = handler.state.mark_reconnecting()
        await handler._update_state(new_state)

        assert handler.state.is_fresh_after_reconnect is False
        # Old data still there but marked stale
        assert handler.state.best_bid == 0.50

    @pytest.mark.asyncio
    async def test_reconnected_state_is_stale_in_snapshot(self):
        now_ms = 10000
        poly_handler = MockVenueWSHandler("poly", "token123")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")

        # Both have recent data
        await poly_handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=now_ms - 100,
        )
        await kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 100,
        )

        # Poly reconnects
        await poly_handler._update_state(poly_handler.state.mark_reconnecting())

        manager = MarketDataManager(
            poly_handler=poly_handler,
            kalshi_handler=kalshi_handler,
            aligned_market_id="ETH-15M-123",
        )

        snapshot = manager.get_snapshot(now_ms=now_ms)

        # Snapshot invalid because poly is reconnecting
        assert snapshot.is_valid(stale_ms=500) is False

    @pytest.mark.asyncio
    async def test_fresh_update_after_reconnect_restores_validity(self):
        now_ms = 10000
        handler = MockVenueWSHandler("poly", "token123")

        # Initial data
        await handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=now_ms - 200,
        )

        # Reconnect
        await handler._update_state(handler.state.mark_reconnecting())
        assert handler.state.is_stale(now_ms=now_ms, stale_ms=500) is True

        # Fresh update arrives
        await handler.update_state_directly(
            [OrderLevel(price=0.51, size=110.0)],
            [],
            update_ms=now_ms - 50,
        )

        assert handler.state.is_stale(now_ms=now_ms, stale_ms=500) is False
        assert handler.state.best_bid == 0.51


class TestRESTFallback:
    """Tests for REST fallback polling (US-003 AC5)."""

    @pytest.mark.asyncio
    async def test_rest_fallback_when_ws_stale(self):
        now_ms = 10000
        poly_handler = MockVenueWSHandler("poly", "token123")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")

        # Poly WS is stale
        await poly_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],
            [],
            update_ms=now_ms - 600,
        )
        # Set REST response
        poly_handler.set_rest_response(
            [OrderLevel(price=0.50, size=100.0)],
            [OrderLevel(price=0.55, size=50.0)],
        )

        # Kalshi WS is fresh
        await kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 100,
        )

        manager = MarketDataManager(
            poly_handler=poly_handler,
            kalshi_handler=kalshi_handler,
            aligned_market_id="ETH-15M-123",
            config=WSConfig(stale_ms=500),
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # REST fallback was used for poly
        assert snapshot.poly_state.best_bid == 0.50

    @pytest.mark.asyncio
    async def test_rest_fallback_for_both_venues_concurrently(self):
        now_ms = 10000
        poly_handler = MockVenueWSHandler("poly", "token123")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")

        # Both WS stale
        await poly_handler.update_state_directly(
            [OrderLevel(price=0.45, size=50.0)],
            [],
            update_ms=now_ms - 600,
        )
        await kalshi_handler.update_state_directly(
            [OrderLevel(price=0.40, size=40.0)],
            [],
            update_ms=now_ms - 600,
        )

        # Set REST responses
        poly_handler.set_rest_response(
            [OrderLevel(price=0.50, size=100.0)],
            [],
        )
        kalshi_handler.set_rest_response(
            [OrderLevel(price=0.48, size=80.0)],
            [],
        )

        manager = MarketDataManager(
            poly_handler=poly_handler,
            kalshi_handler=kalshi_handler,
            aligned_market_id="ETH-15M-123",
            config=WSConfig(stale_ms=500),
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # Both REST fallbacks used
        assert snapshot.poly_state.best_bid == 0.50
        assert snapshot.kalshi_state.best_bid == 0.48

    @pytest.mark.asyncio
    async def test_no_rest_when_ws_fresh(self):
        now_ms = 10000
        poly_handler = MockVenueWSHandler("poly", "token123")
        kalshi_handler = MockVenueWSHandler("kalshi", "TICKER-001")

        # Both WS fresh
        await poly_handler.update_state_directly(
            [OrderLevel(price=0.50, size=100.0)],
            [],
            update_ms=now_ms - 100,
        )
        await kalshi_handler.update_state_directly(
            [OrderLevel(price=0.48, size=80.0)],
            [],
            update_ms=now_ms - 100,
        )

        # Set different REST responses (should not be used)
        poly_handler.set_rest_response(
            [OrderLevel(price=0.99, size=999.0)],
            [],
        )

        manager = MarketDataManager(
            poly_handler=poly_handler,
            kalshi_handler=kalshi_handler,
            aligned_market_id="ETH-15M-123",
            config=WSConfig(stale_ms=500),
        )

        snapshot = await manager.get_fresh_snapshot(now_ms=now_ms)

        # WS data used, not REST
        assert snapshot.poly_state.best_bid == 0.50  # From WS, not REST (0.99)
