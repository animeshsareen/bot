"""Simple monitor to submit a Polymarket FAK order and watch fills via user WS.

Usage:
    python3 legacy_bot/scripts/monitor_poly_fill.py --token <token_id> --side BUY --price 0.5 --size 1

Notes:
- This script uses the same `PolymarketClient` and `PolymarketWSHandler` as the bot.
- Ensure your Polymarket API creds (API key/secret/passphrase and private key) are set in env.
"""

import argparse
import asyncio
import logging
import time
from typing import Optional
import sys
import pathlib

# Ensure package root (`legacy_bot`) is on sys.path so `src` package imports succeed
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.polymarket_client import PolymarketClient
from src.polymarket_ws import PolymarketWSHandler
from src.bot import _parse_poly_order_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("poly-monitor")


async def main(token_id: str, side: str, price: float, size: float, timeout: float = 5.0):
    client = PolymarketClient(host=PolymarketClient.STAGING_HOST)
    handler = PolymarketWSHandler(token_id, client)

    # Start WS handler task
    ws_task = handler.start()
    logger.info("Started PolymarketWSHandler for %s", token_id)

    # Give WS a moment to connect and subscribe
    await asyncio.sleep(1.5)

    loop = asyncio.get_event_loop()

    # Prepare to create order in executor (create_order is synchronous)
    def submit_order():
        return client.create_order(
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type='FAK',
        )

    logger.info("Submitting Polymarket order: %s %s @%s size=%s (FAK)", side, token_id, price, size)

    # Create order (do not await) but register WS watch first to reduce race
    # We'll attempt to get an order_id from the response and then wait if zero fill.
    try:
        raw = await loop.run_in_executor(None, submit_order)
    except Exception as e:
        logger.error("Order submission failed: %s", e, exc_info=True)
        await handler.stop()
        return

    parsed = _parse_poly_order_response(raw, price, size)
    order_id = parsed.get('order_id')
    filled = parsed.get('filled_qty', 0)
    status = parsed.get('status')

    logger.info("OrderResponse parsed: order_id=%s filled=%s status=%s", order_id, filled, status)

    if filled > 0:
        logger.info("Immediate REST fill observed: %d", filled)
    else:
        if not order_id:
            logger.warning("No order id returned by REST; cannot watch WS reliably")
        else:
            # Register watch and wait
            logger.info("Registering fill watch for order_id=%s and waiting up to %.1fs", order_id, timeout)
            handler.register_fill_watch(order_id)
            ws_filled = await handler.wait_for_fill(order_id, timeout_sec=timeout)
            handler.cleanup_fill_watch(order_id)
            if ws_filled > 0:
                logger.info("WS reported fill for order_id=%s qty=%d", order_id, ws_filled)
            else:
                logger.info("No WS fill observed for order_id=%s within %.1fs", order_id, timeout)

    # Cleanup
    logger.info("Shutting down WS handler")
    await handler.stop()
    try:
        await ws_task
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', required=True, help='Polymarket token id (asset id)')
    parser.add_argument('--side', default='BUY', choices=['BUY', 'SELL'])
    parser.add_argument('--price', type=float, required=True)
    parser.add_argument('--size', type=float, required=True)
    parser.add_argument('--timeout', type=float, default=5.0)
    args = parser.parse_args()

    asyncio.run(main(args.token, args.side, args.price, args.size, timeout=args.timeout))
