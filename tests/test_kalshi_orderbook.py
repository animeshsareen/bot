"""Tests for Kalshi orderbook parsing and REST fetch integration.

Tests cover:
- Parsing logic: KalshiClient.parse_orderbook_to_levels()
- REST fetch integration: KalshiWSHandler.fetch_rest_orderbook()
- Proper conversion of NO bids to YES asks with price inversion
- Filtering of zero-quantity levels
- Async REST fetch with mocked client
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.kalshi_client import KalshiClient
from src.kalshi_ws import KalshiWSHandler
from src.venue_state import OrderLevel, VenueState


class TestParseOrderbookToLevels:
    """Tests for KalshiClient.parse_orderbook_to_levels()."""

    def test_parse_orderbook_to_levels_empty(self):
        """Empty orderbook returns empty tuples."""
        raw = {"order_book": {"yes": [], "no": []}}
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)
        assert bids == ()
        assert asks == ()

    def test_parse_orderbook_to_levels_with_data(self):
        """YES bids stay as bids, NO bids invert to asks."""
        raw = {
            "order_book": {
                "yes": [[50, 100], [45, 200]],    # YES bids at 50¢ and 45¢
                "no": [[60, 150], [55, 250]]      # NO bids at 60¢ and 55¢
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # YES bids: 50¢ = $0.50, 45¢ = $0.45 (sorted descending)
        assert len(bids) == 2
        assert bids[0].price == 0.50
        assert bids[0].size == 100
        assert bids[1].price == 0.45
        assert bids[1].size == 200

        # NO bids inverted: 60¢ → 40¢, 55¢ → 45¢ (sorted ascending)
        assert len(asks) == 2
        assert asks[0].price == 0.40
        assert asks[0].size == 150
        assert asks[1].price == 0.45
        assert asks[1].size == 250

    def test_parse_orderbook_to_levels_filters_zero_qty(self):
        """Zero-quantity levels are filtered out."""
        raw = {
            "order_book": {
                "yes": [[50, 0], [45, 100]],
                "no": [[60, 150]]
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # Only non-zero quantities included
        assert len(bids) == 1
        assert bids[0].price == 0.45
        assert len(asks) == 1

    def test_parse_orderbook_to_levels_bids_sorted_descending(self):
        """YES bids sorted in descending order (best first)."""
        raw = {
            "order_book": {
                "yes": [[30, 100], [50, 200], [40, 150]],  # Unordered input
                "no": []
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # Should be sorted descending: 50, 40, 30
        assert len(bids) == 3
        assert bids[0].price == 0.50
        assert bids[1].price == 0.40
        assert bids[2].price == 0.30

    def test_parse_orderbook_to_levels_asks_sorted_ascending(self):
        """YES asks (inverted NO bids) sorted ascending (best first)."""
        raw = {
            "order_book": {
                "yes": [],
                "no": [[30, 100], [50, 200], [40, 150]]  # Unordered input
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # NO 30 → YES ask 70, NO 50 → YES ask 50, NO 40 → YES ask 60
        # Sorted ascending: 50, 60, 70
        assert len(asks) == 3
        assert asks[0].price == 0.50  # 100 - 50
        assert asks[1].price == 0.60  # 100 - 40
        assert asks[2].price == 0.70  # 100 - 30

    def test_parse_orderbook_to_levels_mixed_with_zeros(self):
        """Mixed zero and non-zero quantities."""
        raw = {
            "order_book": {
                "yes": [[50, 100], [45, 0], [40, 50]],
                "no": [[60, 0], [55, 100]]
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # YES: 50 and 40 (skip 45 with qty 0)
        assert len(bids) == 2
        assert bids[0].price == 0.50
        assert bids[1].price == 0.40

        # NO: 55 → 45 (skip 60 with qty 0)
        assert len(asks) == 1
        assert asks[0].price == 0.45

    def test_parse_orderbook_returns_tuples(self):
        """Returns tuples, not lists."""
        raw = {
            "order_book": {
                "yes": [[50, 100]],
                "no": [[60, 150]]
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        assert isinstance(bids, tuple)
        assert isinstance(asks, tuple)
        assert isinstance(bids[0], OrderLevel)
        assert isinstance(asks[0], OrderLevel)

    def test_parse_orderbook_price_conversion_cents_to_dollars(self):
        """Prices correctly converted from cents to dollars."""
        raw = {
            "order_book": {
                "yes": [[1, 10], [99, 20]],
                "no": [[2, 15], [98, 25]]
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        # YES: 1¢ = 0.01, 99¢ = 0.99
        assert bids[0].price == 0.99
        assert bids[1].price == 0.01

        # NO: 2¢ → 98¢ = 0.98, 98¢ → 2¢ = 0.02
        assert asks[0].price == 0.02
        assert asks[1].price == 0.98

    def test_parse_orderbook_size_types(self):
        """Order sizes can be integers or floats."""
        raw = {
            "order_book": {
                "yes": [[50, 100], [45, 50.5]],
                "no": [[60, 150.75]]
            }
        }
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)

        assert bids[0].size == 100
        assert bids[1].size == 50.5
        assert asks[0].size == 150.75

    def test_parse_orderbook_malformed_returns_empty(self):
        """Malformed orderbook returns empty tuples without exception."""
        # Missing 'order_book' key
        raw = {"data": {}}
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)
        assert bids == ()
        assert asks == ()

    def test_parse_orderbook_missing_sides_returns_empty(self):
        """Missing yes/no sides treated as empty."""
        raw = {"order_book": {}}
        bids, asks = KalshiClient.parse_orderbook_to_levels(raw)
        assert bids == ()
        assert asks == ()


class TestGetOrderbook:
    """Tests for KalshiClient.get_orderbook()."""

    @pytest.mark.asyncio
    async def test_get_orderbook_returns_dict(self):
        """get_orderbook() returns raw dict response."""
        mock_response = {
            "order_book": {
                "yes": [[50, 100]],
                "no": [[60, 150]]
            }
        }

        # Mock the client with its request method
        mock_client = AsyncMock(spec=KalshiClient)
        mock_client.get_orderbook = AsyncMock(return_value=mock_response)

        result = await mock_client.get_orderbook("KXBTC15M", depth=10)

        assert result == mock_response
        # Verify get_orderbook was called with correct params
        mock_client.get_orderbook.assert_called_once_with("KXBTC15M", depth=10)


class TestFetchRestOrderbook:
    """Tests for KalshiWSHandler.fetch_rest_orderbook()."""

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_returns_venue_state(self):
        """fetch_rest_orderbook() returns valid VenueState on success."""
        # Mock the REST fetch response
        mock_orderbook = {
            "order_book": {
                "yes": [[50, 100]],
                "no": [[60, 150]]
            }
        }

        # Create mocked client
        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        # Create handler with mocked logger
        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        assert isinstance(result, VenueState)
        assert result.venue == "kalshi"
        assert result.market_id == "KXBTC15M"
        assert result.best_bid == 0.50
        assert result.best_ask == 0.40
        assert result.is_fresh_after_reconnect is True

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_preserves_bid_ask_sizes(self):
        """fetch_rest_orderbook() preserves best bid/ask sizes."""
        mock_orderbook = {
            "order_book": {
                "yes": [[50, 100]],
                "no": [[60, 150]]
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        assert result.best_bid_size == 100
        assert result.best_ask_size == 150

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_handles_empty_book(self):
        """fetch_rest_orderbook() handles empty orderbook."""
        mock_orderbook = {
            "order_book": {
                "yes": [],
                "no": []
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        assert isinstance(result, VenueState)
        assert result.best_bid is None
        assert result.best_ask is None
        assert result.best_bid_size == 0.0
        assert result.best_ask_size == 0.0

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_returns_none_on_fetch_failure(self):
        """fetch_rest_orderbook() returns None if get_orderbook fails."""
        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(side_effect=Exception("API error"))

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_returns_none_on_none_response(self):
        """fetch_rest_orderbook() returns None if get_orderbook returns None."""
        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=None)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_calls_get_orderbook_with_correct_params(self):
        """fetch_rest_orderbook() calls get_orderbook with market_id and depth."""
        mock_orderbook = {
            "order_book": {
                "yes": [[50, 100]],
                "no": [[60, 150]]
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        await handler.fetch_rest_orderbook()

        # Verify get_orderbook was called with correct parameters
        mock_client.get_orderbook.assert_called_once_with(
            ticker="KXBTC15M",
            depth=10
        )

    @pytest.mark.asyncio
    async def test_fetch_rest_orderbook_full_depth(self):
        """fetch_rest_orderbook() includes multiple depth levels."""
        mock_orderbook = {
            "order_book": {
                "yes": [[50, 100], [48, 200], [46, 150]],
                "no": [[60, 100], [58, 200], [56, 150]]
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        # Check we have multiple levels
        assert len(result.bid_levels) == 3
        assert len(result.ask_levels) == 3

        # Verify bid sorting (descending)
        assert result.bid_levels[0].price == 0.50
        assert result.bid_levels[1].price == 0.48
        assert result.bid_levels[2].price == 0.46

        # Verify ask sorting (ascending) and inversion
        assert result.ask_levels[0].price == 0.40  # 100 - 60
        assert result.ask_levels[1].price == 0.42  # 100 - 58
        assert result.ask_levels[2].price == 0.44  # 100 - 56


class TestIntegrationScenarios:
    """Integration tests combining parsing and REST fetch."""

    @pytest.mark.asyncio
    async def test_full_flow_real_like_data(self):
        """Full flow with realistic Kalshi orderbook data."""
        # Realistic orderbook response from Kalshi API
        mock_orderbook = {
            "order_book": {
                "yes": [
                    [75, 500],  # Strong bid at 75¢
                    [74, 300],
                    [73, 200],
                    [72, 150],
                ],
                "no": [
                    [25, 600],  # This is ask side (100 - 25 = 75¢ YES ask)
                    [26, 400],  # 100 - 26 = 74¢
                    [27, 250],  # 100 - 27 = 73¢
                ]
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        # Verify structure
        assert result.venue == "kalshi"
        assert result.market_id == "KXBTC15M"

        # Verify spread (bid-ask)
        assert result.best_bid == 0.75
        # Best ask is lowest ask price: 27→73, 26→74, 25→75. Sorted ascending: 73 is best
        assert result.best_ask == 0.73

        # Verify all levels are populated
        assert len(result.bid_levels) == 4
        assert len(result.ask_levels) == 3

    @pytest.mark.asyncio
    async def test_zero_quantity_filtering_in_rest_flow(self):
        """Zero quantities are filtered in REST fetch flow."""
        mock_orderbook = {
            "order_book": {
                "yes": [[50, 100], [45, 0], [40, 50]],  # Skip 45¢
                "no": [[60, 0], [55, 100]]  # Skip 60¢
            }
        }

        mock_client = AsyncMock()
        mock_client.get_orderbook = AsyncMock(return_value=mock_orderbook)

        handler = KalshiWSHandler(
            market_id="KXBTC15M",
            client=mock_client,
            config=None
        )

        result = await handler.fetch_rest_orderbook()

        # Only non-zero levels included
        assert len(result.bid_levels) == 2
        assert len(result.ask_levels) == 1
        assert result.best_bid == 0.50
        assert result.best_ask == 0.45
