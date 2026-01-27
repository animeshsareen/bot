import asyncio
import base64
import json
import logging
import time
from typing import Optional, Any, Dict, List, Tuple

import websockets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from .ws_manager import VenueWSHandler, WSConfig
from .venue_state import VenueState, OrderLevel
from .kalshi_client import KalshiClient

logger = logging.getLogger(__name__)

class KalshiWSHandler(VenueWSHandler):
    """
    WebSocket handler for Kalshi.

    Subscribes to both orderbook_delta (market data) and fill (user fills)
    channels on the authenticated WebSocket connection.
    """

    PROD_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    DEMO_WS_URL = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    def __init__(self, market_id: str, client: KalshiClient, config: WSConfig = None):
        super().__init__("kalshi", market_id, config or WSConfig())
        self.client = client
        self.sid: Optional[int] = None # Sequence ID for orderbook
        self._bid_map: Dict[int, int] = {}
        self._no_bid_map: Dict[int, int] = {}

        # Fill tracking via WebSocket
        self._fill_events: Dict[str, asyncio.Event] = {}   # order_id -> event
        self._fill_counts: Dict[str, int] = {}              # order_id -> cumulative filled qty
        self._fill_lock = asyncio.Lock()

        # Buffer for fills that arrive before we know the order_id
        # (fills that arrive during REST submission, before register_fill_watch)
        self._unmatched_fills: List[dict] = []

        # Authoritative position from Kalshi (updated on every fill)
        self.last_post_position: Optional[int] = None       # Latest post_position from fill WS

        # Determine WS URL based on client's base URL
        if "demo" in self.client.base_url:
            self.ws_url = self.DEMO_WS_URL
        else:
            self.ws_url = self.PROD_WS_URL

    def _generate_auth_headers(self) -> dict:
        """Generate RSA-PSS signed authentication headers for WebSocket."""
        # Match legacy bot timestamp exactly
        timestamp = str(int(time.time() * 1000))
        path = '/trade-api/ws/v2'
        message = f"{timestamp}GET{path}"

        signature = self.client.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return {
            "KALSHI-ACCESS-KEY": self.client.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def register_fill_watch(self, order_id: str) -> None:
        """Register interest in fills for an order_id.

        Also replays any buffered fills that arrived before this registration
        (handles the race where fill WS message arrives during REST submission).
        """
        if order_id not in self._fill_events:
            self._fill_events[order_id] = asyncio.Event()
            self._fill_counts[order_id] = 0

        # Replay any buffered fills that match this order_id
        remaining = []
        for fill_msg in self._unmatched_fills:
            if fill_msg.get("order_id") == order_id:
                count = fill_msg.get("count", 0)
                self._fill_counts[order_id] += count
                self._fill_events[order_id].set()
                logger.info(
                    "Replayed buffered fill for order_id=%s count=%d",
                    order_id, count,
                )
            else:
                remaining.append(fill_msg)
        self._unmatched_fills = remaining

        logger.debug("Registered fill watch for order_id=%s", order_id)

    async def wait_for_fill(self, order_id: str, timeout_sec: float = 3.0) -> int:
        """Wait for a fill notification for the given order_id.

        Args:
            order_id: The Kalshi exchange order ID to watch.
            timeout_sec: Maximum time to wait for a fill notification.

        Returns:
            Cumulative filled quantity (0 if timeout with no fills).
        """
        event = self._fill_events.get(order_id)
        if event is None:
            # Wasn't pre-registered; register now (may miss fills in race window)
            self.register_fill_watch(order_id)
            event = self._fill_events[order_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.info("Fill wait timeout for order_id=%s after %.1fs", order_id, timeout_sec)

        filled = self._fill_counts.get(order_id, 0)
        logger.info("Fill result for order_id=%s: qty=%d", order_id, filled)
        return filled

    def cleanup_fill_watch(self, order_id: str) -> None:
        """Remove fill tracking state for a completed order."""
        self._fill_events.pop(order_id, None)
        self._fill_counts.pop(order_id, None)

    def _process_fill_message(self, msg_body: dict) -> None:
        """Process a fill WebSocket message and update tracking state.

        Fill message format:
        {
            "trade_id": "...",
            "order_id": "...",
            "market_ticker": "...",
            "side": "yes" | "no",
            "yes_price": 75,
            "count": 278,
            "action": "buy",
            "ts": 1671899397,
            "post_position": 500
        }
        """
        order_id = msg_body.get("order_id")
        count = msg_body.get("count", 0)
        trade_id = msg_body.get("trade_id", "?")
        market_ticker = msg_body.get("market_ticker", "?")
        side = msg_body.get("side", "?")
        action = msg_body.get("action", "?")
        yes_price = msg_body.get("yes_price", 0)
        post_position = msg_body.get("post_position", None)

        logger.info(
            "KALSHI_WS_FILL: order_id=%s trade_id=%s ticker=%s side=%s "
            "action=%s price=%s count=%d post_position=%s",
            order_id, trade_id, market_ticker, side,
            action, yes_price, count, post_position,
        )

        # Update authoritative position from Kalshi
        if post_position is not None:
            self.last_post_position = post_position
            logger.info("Kalshi authoritative position updated: %d", post_position)

        if order_id and order_id in self._fill_events:
            self._fill_counts[order_id] = self._fill_counts.get(order_id, 0) + count
            self._fill_events[order_id].set()
            logger.info(
                "Fill watch signaled for order_id=%s cumulative_qty=%d",
                order_id, self._fill_counts[order_id],
            )
        elif order_id:
            # Buffer the fill — it may be for an order we're about to register
            # (race: fill arrives during REST submission before register_fill_watch)
            self._unmatched_fills.append(msg_body)
            # Cap buffer to prevent unbounded growth
            if len(self._unmatched_fills) > 100:
                self._unmatched_fills = self._unmatched_fills[-50:]
            logger.info(
                "Buffered fill for order_id=%s count=%d (no active watch yet)",
                order_id, count,
            )

    async def connect_and_subscribe(self) -> Any:
        """Connect and subscribe to orderbook + fill channels."""
        auth_headers = self._generate_auth_headers()
        ws = await websockets.connect(
            self.ws_url,
            additional_headers=auth_headers,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        )

        # Subscribe to orderbook
        orderbook_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": [self.market_id]
            }
        }
        await ws.send(json.dumps(orderbook_msg))

        # Subscribe to user fills (authenticated channel, market_tickers ignored by server)
        fill_msg = {
            "id": 2,
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"],
                "market_tickers": [self.market_id]
            }
        }
        await ws.send(json.dumps(fill_msg))

        logger.info("Kalshi WS subscribed to orderbook_delta + fill for %s", self.market_id)
        return ws

    async def process_message(self, message: dict) -> Optional[VenueState]:
        """Process Kalshi WS message (orderbook or fill)."""
        type_ = message.get("type")

        if type_ == "fill":
            msg_body = message.get("msg")
            if msg_body:
                self._process_fill_message(msg_body)
            return None

        if type_ == "orderbook_snapshot":
            # Handle snapshot
            msg_body = message.get("msg")
            if not msg_body: return None
            
            # Kalshi sends yes/no bids.
            # "yes" or "no" might be missing if empty.
            yes = msg_body.get("yes", [])
            no = msg_body.get("no", [])
            
            # Reset maps
            self._bid_map = {p: q for p, q in yes}
            self._no_bid_map = {p: q for p, q in no}

            bid_levels = self._convert_levels(yes) # Yes bids
            
            # Transform NO bids to YES asks
            derived_asks = []
            for p, q in no:
                # price in cents. 
                # If NO bid is 40, YES ask is 60.
                ask_price_cents = 100 - p
                derived_asks.append(OrderLevel(price=ask_price_cents/100.0, size=q))
            
            # Sort asks ascending (best ask is lowest price)
            derived_asks.sort(key=lambda x: x.price)
            
            # Sort bids descending (best bid is highest price)
            bid_levels.sort(key=lambda x: x.price, reverse=True)
            
            # Update state (snapshot)
            self.sid = message.get("sid")
            return self._state.with_orderbook(bid_levels, derived_asks)
            
        elif type_ == "orderbook_delta":
            # Handle delta
            msg_body = message.get("msg")
            if not msg_body: return None
            
            # Delta format: {"price": 96, "delta": -54, "side": "yes", ...}
            price = msg_body.get("price")
            delta = msg_body.get("delta")
            side = msg_body.get("side")
            
            if price is None or delta is None or side is None:
                return None
                
            if side == "yes":
                self._update_map(self._bid_map, price, delta)
            elif side == "no":
                self._update_map(self._no_bid_map, price, delta)
            
            # Reconstruct levels
            bids = self._map_to_levels(self._bid_map)
            # Asks from NO bids
            asks = []
            for p, q in self._no_bid_map.items():
                if q > 0:
                    ask_price = (100 - p) / 100.0
                    asks.append(OrderLevel(price=ask_price, size=q))
            asks.sort(key=lambda x: x.price)
            
            new_sid = message.get("sid")
            # Basic seq check could be added here
            self.sid = new_sid
            
            return self._state.with_orderbook(bids, asks)

        return None

    def _convert_levels(self, raw_levels: List[List]) -> List[OrderLevel]:
        """Convert [price, qty] list to OrderLevel list."""
        levels = []
        for p, q in raw_levels:
            # p is cents, q is count
            if q > 0:
                levels.append(OrderLevel(price=p/100.0, size=q))
        return levels

    def _update_map(self, book_map: Dict[int, int], price: int, delta: int):
        current = book_map.get(price, 0)
        new_qty = current + delta
        if new_qty <= 0:
            if price in book_map:
                del book_map[price]
        else:
            book_map[price] = new_qty

    def _map_to_levels(self, book_map: Dict[int, int]) -> List[OrderLevel]:
        levels = []
        for p, q in book_map.items():
            if q > 0:
                levels.append(OrderLevel(price=p/100.0, size=q))
        # Sort bids descending
        levels.sort(key=lambda x: x.price, reverse=True)
        return levels

    async def fetch_rest_orderbook(self) -> Optional[VenueState]:
        """
        Fetch orderbook snapshot via REST API (fallback when WS is stale).
        Uses KalshiClient.get_orderbook() for fresh data.

        Returns:
            VenueState with current orderbook snapshot, or None if fetch fails
        """
        try:
            raw_orderbook = await self.client.get_orderbook(
                ticker=self.market_id,
                depth=10  # Configurable, but 10 is reasonable default
            )

            if raw_orderbook is None:
                logger.warning(f"REST orderbook fetch returned None for {self.market_id}")
                return None

            bid_levels, ask_levels = KalshiClient.parse_orderbook_to_levels(raw_orderbook)

            current_ms = int(time.time() * 1000)
            best_bid = bid_levels[0].price if bid_levels else None
            best_ask = ask_levels[0].price if ask_levels else None
            best_bid_size = bid_levels[0].size if bid_levels else 0.0
            best_ask_size = ask_levels[0].size if ask_levels else 0.0

            return VenueState(
                venue="kalshi",
                market_id=self.market_id,
                last_update_ms=current_ms,
                best_bid=best_bid,
                best_ask=best_ask,
                best_bid_size=best_bid_size,
                best_ask_size=best_ask_size,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                is_fresh_after_reconnect=True,
            )
        except Exception as e:
            logger.error(f"REST orderbook fetch failed for {self.market_id}: {e}")
            return None

