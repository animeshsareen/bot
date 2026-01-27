import os
import asyncio
import logging
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv


from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType, MarketOrderArgs
from py_clob_client.constants import POLYGON

# Monkeypatch to avoid Cloudflare blocking
import py_clob_client.http_helpers.helpers as http_helpers

original_overloadHeaders = http_helpers.overloadHeaders

def patched_overloadHeaders(method: str, headers: dict) -> dict:
    headers = original_overloadHeaders(method, headers)
    # Override User-Agent to mimic a browser
    headers["User-Agent"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    return headers

http_helpers.overloadHeaders = patched_overloadHeaders

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

class PolymarketClient:
    """
    Wrapper around py_clob_client for Polymarket interactions.
    """

    PROD_HOST = "https://clob.polymarket.com"
    STAGING_HOST = "https://clob-staging.polymarket.com"
    
    def __init__(self, host: Optional[str] = None, chain_id: int = POLYGON):
        # Default to staging unless specified
        self.host = host or self.STAGING_HOST
        self.chain_id = chain_id
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY") or os.getenv("PK")
        self.api_key = os.getenv("POLYMARKET_API_KEY")
        self.api_secret = os.getenv("POLYMARKET_API_SECRET")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")
        self.api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")
        self.proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS") or os.getenv("FUNDER_ADDRESS")
        self.signature_type = int(os.getenv("SIGNATURE_TYPE", "0"))

        if not self.private_key:
            logger.warning("No Polymarket private key found in env (POLYMARKET_PRIVATE_KEY or PK). Read-only mode may fail for L2.")

        if self.proxy_address:
            logger.info(f"Using Polymarket Proxy: {self.proxy_address}")

        # Initialize ClobClient
        # Note: creds are optional for public endpoints, but needed for trading
        self.client = ClobClient(
            host=self.host,
            key=self.private_key,
            chain_id=self.chain_id,
            creds=self._get_creds(),
            funder=self.proxy_address, # Use proxy as funder if provided
            signature_type=self.signature_type
        )
        
        # Determine if we have L2 access
        self.can_trade = bool(self.private_key and self.api_key and self.api_secret and self.api_passphrase)
        if not self.can_trade:
            logger.info("PolymarketClient initialized in READ-ONLY mode (missing API credentials).")

    async def warmup_market(self, token_ids: List[str]):
        """Warm up cache for tick size, neg risk, and fee rate."""
        logger.info(f"Warming up cache for {len(token_ids)} tokens...")
        loop = asyncio.get_running_loop()
        
        def _warmup_sync():
            for tid in token_ids:
                try:
                    self.client.get_tick_size(tid)
                    self.client.get_neg_risk(tid)
                    self.client.get_fee_rate_bps(tid)
                    logger.info(f"Warmed up {tid}")
                except Exception as e:
                    logger.error(f"Failed to warmup {tid}: {e}")
                    
        await loop.run_in_executor(None, _warmup_sync)


    def _get_creds(self):
        if self.api_key and self.api_secret and self.api_passphrase:
            from py_clob_client.clob_types import ApiCreds
            return ApiCreds(
                api_key=self.api_key,
                api_secret=self.api_secret,
                api_passphrase=self.api_passphrase
            )
        return None

    def get_market(self, condition_id: str) -> Dict:
        """Get market details by condition ID."""
        return self.client.get_market(condition_id)
        
    def get_market_by_slug(self, slug: str) -> Dict:
        """Get market details by slug (not directly in minimal client, might need REST or finding token via other means)."""
        # py_clob_client might not have get_market_by_slug convenient method?
        # It has `get_markets` with filters.
        # Let's assume we use it for finding markets.
        # This wrapper might need to extend functionality if ClobClient is limited.
        # For now, rely on `get_market` with ID (token_id).
        pass

    def get_orderbook(self, token_id: str) -> Dict:
        """Get orderbook for a token."""
        return self.client.get_order_book(token_id)

    def get_balance(self) -> Dict:
        """Get account balance (USDC on Polygon)."""
        # ClobClient doesn't directly expose on-chain balance usually, 
        # but might have `get_balance_allowance` or similar.
        # If not, we might need simple RPC call.
        # py_clob_client has `get_balance_allowance`.
        if not self.can_trade:
             return {"balance": 0.0}
             
        # This usually returns on-chain balance
        # Use BalanceAllowanceParams
        from py_clob_client.clob_types import BalanceAllowanceParams
        params = BalanceAllowanceParams(asset_type="COLLATERAL")
        return self.client.get_balance_allowance(params=params)

    async def create_order(
        self,
        token_id: str,
        side: str,  # 'BUY' or 'SELL'
        price: float,
        size: float,
        order_type: str = "FOK",
    ) -> Dict:
        """Create and post an order.

        Args:
            order_type: 'FOK' (Fill-Or-Kill) or 'GTC' (Good-Til-Cancelled).
                        Defaults to FOK to prevent resting orders.
        """
        if not self.can_trade:
            raise RuntimeError("Cannot trade without full API credentials")

        c_side = "BUY" if side.upper() == "BUY" else "SELL"

        ot_map = {"FOK": OrderType.FOK, "GTC": OrderType.GTC, "GTD": OrderType.GTD, "FAK": OrderType.FAK}
        ot = ot_map.get(order_type.upper(), OrderType.FOK)

        order_args = OrderArgs(
            price=price,
            size=size,
            side=c_side,
            token_id=token_id,
        )

        loop = asyncio.get_running_loop()
        def _exec():
            signed_order = self.client.create_order(order_args)
            return self.client.post_order(signed_order, orderType=ot)

        return await loop.run_in_executor(None, _exec)

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        return self.client.cancel(order_id)

    def cancel_all(self):
        """Cancel all orders."""
        return self.client.cancel_all()

    async def create_market_order(
        self,
        token_id: str,
        side: str,
        amount: float,  # USDC for BUY, Shares for SELL
        price: float = 0,
        order_type: str = "FOK",
    ) -> Dict:
        """Create and post a market order (mimics createAndPostMarketOrder).

        Args:
            token_id: The market token ID.
            side: 'BUY' or 'SELL'.
            amount: For BUY: Amount in USDC to spend.
                    For SELL: Number of shares to sell.
            price: Optional worst price (slippage protection). If 0, client calculates it.
            order_type: 'FOK' (default) or 'FAK'.
        """
        if not self.can_trade:
            raise RuntimeError("Cannot trade without full API credentials")

        c_side = "BUY" if side.upper() == "BUY" else "SELL"
        
        ot_map = {"FOK": OrderType.FOK, "GTC": OrderType.GTC, "GTD": OrderType.GTD, "FAK": OrderType.FAK}
        ot = ot_map.get(order_type.upper(), OrderType.FOK)

        # MarketOrderArgs takes 'amount' which maps to 'size' in standard OrderArgs
        # but the semantics differ by side as documented.
        market_order_args = MarketOrderArgs(
            token_id=token_id,
            side=c_side,
            amount=amount,
            price=price,
            order_type=ot,
        )

        # Create signed order (local)
        loop = asyncio.get_running_loop()
        def _exec():
            signed_order = self.client.create_market_order(market_order_args)
            # Post it
            return self.client.post_order(signed_order, orderType=ot)

        return await loop.run_in_executor(None, _exec)


