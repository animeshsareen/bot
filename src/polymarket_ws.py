import asyncio
import json
import logging
import websockets
from typing import Optional, Any, List
import time

from .ws_manager import VenueWSHandler, WSConfig
from .venue_state import VenueState, OrderLevel
from .polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)

class PolymarketWSHandler(VenueWSHandler):
    """
    WebSocket handler for Polymarket (CLOB).
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(self, market_id: str, client: PolymarketClient, config: WSConfig = None):
        super().__init__("poly", market_id, config or WSConfig())
        self.client = client

    async def connect_and_subscribe(self) -> Any:
        """Connect and subscribe to market channel."""
        ws = await websockets.connect(self.WS_URL, ping_interval=10, ping_timeout=10)
        
        # Subscribe to market (lowercase "market" as per legacy bot)
        msg = {
            "type": "market",
            "assets_ids": [self.market_id]
        }
        await ws.send(json.dumps(msg))
        logger.info(f"Polymarket WS subscribed to {self.market_id}")
        return ws

    async def process_message(self, message: dict) -> Optional[VenueState]:
        """Process Polymarket WS message per documentation.

        Per Polymarket docs, messages come as:
        - book: Full orderbook snapshot (on subscribe and on trades)
        - price_change: Incremental BBO updates (on order place/cancel)
        """
        if isinstance(message, list):
            # Multiple events - process in order, return latest state
            latest_state = None
            for event in message:
                state = self._process_single_event(event)
                if state:
                    latest_state = state
            return latest_state
        elif isinstance(message, dict):
            return self._process_single_event(message)
        return None

    def _process_single_event(self, event: dict) -> Optional[VenueState]:
        """Process a single Polymarket market channel message.

        Per docs, handles:
        - book: Full orderbook with bids/asks arrays
        - price_change: BBO updates in price_changes array
        """
        event_type = event.get("event_type")
        asset_id = event.get("asset_id")

        # Filter to our token
        if asset_id and asset_id != self.market_id:
            return None

        if event_type == "book":
            return self._handle_book_event(event)
        elif event_type == "price_change":
            return self._handle_price_change_event(event)

        return None

    def _handle_book_event(self, event: dict) -> Optional[VenueState]:
        """Handle book message: full orderbook snapshot.

        Per docs: emitted on first subscribe and when trades affect book.
        Format: bids/asks are OrderSummary[] with {price, size}.
        """
        bids_raw = event.get("bids", [])
        asks_raw = event.get("asks", [])

        bids = self._parse_levels(bids_raw)
        asks = self._parse_levels(asks_raw)

        # Sort: bids descending (best=highest), asks ascending (best=lowest)
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        ts = event.get("timestamp")
        update_ms = int(ts) if ts else int(time.time() * 1000)

        return self._state.with_orderbook(bids, asks, update_ms)

    def _handle_price_change_event(self, event: dict) -> Optional[VenueState]:
        """Handle price_change message: incremental BBO updates.

        Per docs: emitted when orders placed/cancelled. Includes array of
        price_changes, each with best_bid/best_ask for the asset.
        """
        changes = event.get("price_changes", [])

        # Extract timestamp (shared across all changes in message)
        ts = event.get("timestamp")
        update_ms = int(ts) if ts else int(time.time() * 1000)

        # Find the change for our asset (should be just one per token typically)
        for change in changes:
            if change.get("asset_id") != self.market_id:
                continue

            # Extract best bid/ask from the price_change
            best_bid_str = change.get("best_bid")
            best_ask_str = change.get("best_ask")

            if not best_bid_str or not best_ask_str:
                continue

            try:
                best_bid = float(best_bid_str)
                best_ask = float(best_ask_str)
            except (ValueError, TypeError):
                logger.warning(f"Failed to parse best_bid/ask: {best_bid_str}/{best_ask_str}")
                continue

            # BBO-only update: preserve existing depth levels from last book event.
            # price_change only carries L1 prices, not depth. Using with_orderbook()
            # here would destroy all L2 data and set sizes to 0, causing sizing to fail.
            return self._state.with_bbo_update(
                best_bid=best_bid,
                best_bid_size=self._state.best_bid_size,
                best_ask=best_ask,
                best_ask_size=self._state.best_ask_size,
                update_ms=update_ms,
            )

        return None

    def _parse_levels(self, raw: List[Any]) -> List[OrderLevel]:
        levels = []
        for item in raw:
            p, s = 0.0, 0.0
            if isinstance(item, dict):
                # Try verbose keys first, then compressed
                p = float(item.get("price") or item.get("p", 0))
                s = float(item.get("size") or item.get("s", 0))
            elif hasattr(item, "price") and hasattr(item, "size"):
                p = float(item.price)
                s = float(item.size)
            elif isinstance(item, list) and len(item) >= 2:
                p = float(item[0])
                s = float(item[1])
            
            if s > 0:
                levels.append(OrderLevel(price=p, size=s))
        
        return levels
        
    async def fetch_rest_orderbook(self) -> Optional[VenueState]:
        try:
            # Wrap synchronous get_orderbook in executor to avoid blocking async loop
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, self.client.get_orderbook, self.market_id)
            
            # Handle OrderBookSummary object or dict
            if hasattr(raw, "bids") and hasattr(raw, "asks"):
                bids_raw = raw.bids
                asks_raw = raw.asks
            elif isinstance(raw, dict):
                bids_raw = raw.get("bids", [])
                asks_raw = raw.get("asks", [])
            else:
                logger.error(f"Unknown orderbook format: {type(raw)}")
                return None
            
            bids = self._parse_levels(bids_raw)
            asks = self._parse_levels(asks_raw)
            
            # Ensure correct sorting:
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            
            return self._state.with_orderbook(bids, asks)
        except Exception as e:
            logger.error(f"Failed to fetch Polymarket REST orderbook: {e}")
            return None

