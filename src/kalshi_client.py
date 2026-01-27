import asyncio
import base64
import json
import time
import uuid
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .venue_state import OrderLevel

# Configure logging
logger = logging.getLogger(__name__)

class KalshiClient:
    """
    Async client for Kalshi API (V2).
    
    Handles authentication via RSA signature and provides methods for
    market discovery, order management, and portfolio tracking.
    """
    
    # Production and Demo URLs
    PROD_URL = "https://api.elections.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    
    def __init__(self, key_id: str, private_key_path: str, base_url: Optional[str] = None):
        self.key_id = key_id
        self.private_key_path = private_key_path
        # Default to demo unless specified, safer for new users
        self.base_url = (base_url or self.DEMO_URL).rstrip("/")
        self.private_key = self._load_private_key()
        self.client = httpx.AsyncClient()

    def _load_private_key(self) -> rsa.RSAPrivateKey:
        """Load RSA private key from file."""
        try:
            with open(self.private_key_path, "rb") as f:
                return serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        except Exception as e:
            logger.error(f"Failed to load private key from {self.private_key_path}: {e}")
            raise

    def _get_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Construct authenticated request headers using RSA-PSS signing.

        Per Kalshi docs, the signed message is: timestamp_ms + METHOD + path_without_query
        No body is included. Query params must be stripped from the path.
        """
        timestamp = str(int(time.time() * 1000))

        # Strip query parameters from path before signing
        path_without_query = path.split('?')[0]
        msg = f"{timestamp}{method}{path_without_query}"

        signature = self.private_key.sign(
            msg.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        b64_signature = base64.b64encode(signature).decode('utf-8')

        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": b64_signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }

    async def request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Any:
        """Execute HTTP request."""
        url = f"{self.base_url}{endpoint}"
        path = f"/trade-api/v2{endpoint}"
        
        headers = self._get_headers(method, path)
        
        try:
            resp = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=10.0
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {method} {url}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get_markets(self, **kwargs) -> Dict:
        """
        Get markets.
        Supported filters: ticker, event_ticker, series_ticker, max_close_ts, min_close_ts, status, etc.
        """
        return await self.request("GET", "/markets", params=kwargs)

    async def get_market(self, ticker: str) -> Dict:
        """Get single market by ticker."""
        return await self.request("GET", f"/markets/{ticker}")

    async def get_orderbook(self, ticker: str, depth: int = 10) -> Dict:
        """Get market orderbook."""
        return await self.request("GET", f"/markets/{ticker}/orderbook", params={"depth": depth})

    async def get_balance(self) -> Dict:
        """Get account balance."""
        return await self.request("GET", "/portfolio/balance")

    async def get_fills(self, **kwargs) -> Dict:
        """Get fills history."""
        return await self.request("GET", "/portfolio/fills", params=kwargs)

    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        expiration_ts: Optional[int] = None,
        client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = None,
    ) -> Dict:
        """
        Create an order.

        Args:
            ticker: Market ticker
            action: 'buy' or 'sell'
            side: 'yes' or 'no'
            count: Number of contracts
            type: 'limit' or 'market'
            yes_price: Price for YES (1-99), required for limit
            no_price: Price for NO (1-99), alternative to yes_price
            expiration_ts: Unix timestamp (seconds)
            client_order_id: Unique ID
            time_in_force: 'fill_or_kill', 'good_till_canceled', or 'immediate_or_cancel'
        """
        if not client_order_id:
            client_order_id = str(uuid.uuid4())

        payload = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": type,
            "client_order_id": client_order_id
        }

        if type == "limit":
            if yes_price:
                payload["yes_price"] = yes_price
            elif no_price:
                payload["no_price"] = no_price
            else:
                raise ValueError("Limit order requires yes_price or no_price")

        if expiration_ts:
            payload["expiration_ts"] = expiration_ts

        if time_in_force:
            payload["time_in_force"] = time_in_force

        return await self.request("POST", "/portfolio/orders", data=payload)

    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        return await self.request("DELETE", f"/portfolio/orders/{order_id}")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    @staticmethod
    def parse_orderbook_to_levels(raw_orderbook: dict) -> tuple[tuple[OrderLevel, ...], tuple[OrderLevel, ...]]:
        """
        Convert raw Kalshi orderbook API response to bid/ask OrderLevel tuples.

        Kalshi API returns:
        - yes: List of [price_cents, qty] for YES bids
        - no: List of [price_cents, qty] for NO bids

        Conversion logic:
        - YES bids stay as bids
        - NO bids inverted to YES asks (ask_price = (100 - no_bid_price) / 100)

        Args:
            raw_orderbook: Dict with structure {"order_book": {"yes": [...], "no": [...]}}

        Returns:
            Tuple of (bid_levels, ask_levels), each a tuple of OrderLevel
        """
        try:
            order_book = raw_orderbook.get("order_book", {})
            yes_bids = order_book.get("yes", [])
            no_bids = order_book.get("no", [])

            # Parse YES bids directly (already bids)
            bid_levels = [
                OrderLevel(price=price_cents / 100.0, size=qty)
                for price_cents, qty in yes_bids
                if qty > 0
            ]

            # Convert NO bids to YES asks
            ask_levels = [
                OrderLevel(price=(100 - price_cents) / 100.0, size=qty)
                for price_cents, qty in no_bids
                if qty > 0
            ]

            # Sort: bids descending (best first), asks ascending (best first)
            bid_levels.sort(key=lambda x: x.price, reverse=True)
            ask_levels.sort(key=lambda x: x.price)

            return tuple(bid_levels), tuple(ask_levels)
        except Exception as e:
            logger.error(f"Failed to parse orderbook: {e}")
            return (), ()
