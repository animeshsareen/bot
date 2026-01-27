
import asyncio
import json
import logging
import websockets
from typing import Dict, List, Optional
from .polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)

class PolymarketUserWS:
    """
    WebSocket handler for Polymarket User Channel (fills/orders).
    """
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.ws = None
        self._running = False
        
        # Fill tracking
        self._fill_events: Dict[str, asyncio.Event] = {}
        self._fill_counts: Dict[str, float] = {}  # cumulative qty
        
        # Buffer for fills arriving before watch registration
        self._unmatched_fills: List[dict] = []

    async def start(self):
        self._running = True
        logger.info("Initializing PolymarketUserWS...")
        asyncio.create_task(self._main_loop())

    async def stop(self):
        self._running = False
        if self.ws:
            await self.ws.close()

    async def _main_loop(self):
        logger.info("Starting Polymarket User WS loop...")
        while self._running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    self.ws = ws
                    if not await self._authenticate(ws):
                        logger.error("Polymarket User WS authentication failed, retrying in 5s...")
                        await asyncio.sleep(5)
                        continue
                        
                    logger.info("Polymarket User WS authenticated")
                    
                    async for message in ws:
                        if not self._running: break
                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode message: {message}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
            except Exception as e:
                logger.error(f"Polymarket User WS connection error: {e}")
                await asyncio.sleep(5)
    
    async def _authenticate(self, ws) -> bool:
        if not (self.client.api_key and self.client.api_secret and self.client.api_passphrase):
            logger.error("Missing Polymarket API credentials for User WS")
            return False

        msg = {
            "type": "user",
            "auth": {
                "apiKey": self.client.api_key,
                "secret": self.client.api_secret,
                "passphrase": self.client.api_passphrase
            }
        }
        await ws.send(json.dumps(msg))
        return True

    async def _process_message(self, data):
        # Polymarket can send list of events or single dict
        if isinstance(data, list):
            for item in data:
                await self._process_single_event(item)
        elif isinstance(data, dict):
            await self._process_single_event(data)

    async def _process_single_event(self, event: dict):
        event_type = event.get("event_type")
        if event_type == "trade":
            await self._handle_trade(event)
        elif event_type == "error":
            logger.error(f"Polymarket User WS Error: {event.get('message')}")
            
    async def _handle_trade(self, trade: dict):
        # We check both taker_order_id and maker_orders to find our order
        
        # 1. Check taker_order_id (likely for FAK/taker orders)
        taker_id = trade.get("taker_order_id")
        size = float(trade.get("size", 0))
        
        if taker_id:
            self._record_fill(taker_id, size, trade, "taker")
            
        # 2. Check maker_orders (if we provided liquidity)
        maker_orders = trade.get("maker_orders", [])
        for mo in maker_orders:
            oid = mo.get("order_id")
            matched_amt = float(mo.get("matched_amount", 0))
            if oid:
                self._record_fill(oid, matched_amt, trade, "maker")

    def _record_fill(self, order_id: str, quantity: float, context: dict, role: str):
        logger.info(f"POLY_USER_WS_FILL: order_id={order_id} qty={quantity} role={role} status={context.get('status')}")
        
        if order_id in self._fill_events:
            self._fill_counts[order_id] = self._fill_counts.get(order_id, 0.0) + quantity
            self._fill_events[order_id].set()
            logger.info(f"Fill watch signaled for order_id={order_id} cumulative_qty={self._fill_counts[order_id]}")
        else:
            # Buffer
            self._unmatched_fills.append({'order_id': order_id, 'qty': quantity, 'time': asyncio.get_event_loop().time()})
            # Clean old buffer
            now = asyncio.get_event_loop().time()
            self._unmatched_fills = [f for f in self._unmatched_fills if now - f['time'] < 60] # Keep for 60s
            if len(self._unmatched_fills) > 100:
                self._unmatched_fills.pop(0)
            logger.info(f"Buffered fill for order_id={order_id} count={quantity} (no active watch)")

    def register_fill_watch(self, order_id: str):
        if order_id not in self._fill_events:
            self._fill_events[order_id] = asyncio.Event()
            self._fill_counts[order_id] = 0.0
            logger.debug(f"Registered fill watch for poly order_id={order_id}")
            
        # Replay buffer
        remaining = []
        for fill in self._unmatched_fills:
            if fill['order_id'] == order_id:
                qty = fill['qty']
                self._fill_counts[order_id] += qty
                self._fill_events[order_id].set()
                logger.info(f"Replayed buffered Poly fill for {order_id}: {qty}")
            else:
                remaining.append(fill)
        self._unmatched_fills = remaining

    async def wait_for_fill(self, order_id: str, timeout_sec: float = 3.0) -> float:
        if order_id not in self._fill_events:
            self.register_fill_watch(order_id)
            
        event = self._fill_events[order_id]
        
        # If already filled, return immediately
        if self._fill_counts[order_id] > 0:
             # Wait a tiny bit more just in case more partials come in same burst? 
             # No, standard practice is return what we have or wait for event.
             pass

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.info(f"Poly fill wait timeout for order_id={order_id} after {timeout_sec}s")
            
        return self._fill_counts.get(order_id, 0.0)
        
    def cleanup_fill_watch(self, order_id: str):
        self._fill_events.pop(order_id, None)
        self._fill_counts.pop(order_id, None)
