import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from src.kalshi_ws import KalshiWSHandler
from src.polymarket_ws import PolymarketWSHandler
from src.ws_manager import MarketDataManager, WSConfig
from src.opportunity import OpportunityDetector
from src.venue_state import MarketSnapshot

import time

@pytest.mark.asyncio
async def test_full_ingestion_to_detector_flow():
    """
    Test the flow from WS message ingestion to opportunity detection.
    """
    # 1. Setup Data Manager with Mock Handlers
    # We will manually trigger process_message on the handlers
    
    mock_kc = MagicMock()
    mock_pc = MagicMock()
    
    k_handler = KalshiWSHandler("ETH-24MAR-15M", mock_kc)
    p_handler = PolymarketWSHandler("poly_eth_token", mock_pc)
    
    manager = MarketDataManager(p_handler, k_handler, "eth-test")
    detector = OpportunityDetector(raw_profit_trigger_pct=1.0)
    
    # 2. Ingest Kalshi Snapshot
    k_snapshot_msg = {
        "type": "orderbook_snapshot",
        "sid": 1,
        "msg": {
            "yes": [[45, 100]], # YES Bid @ 0.45, size 100
            "no": [[50, 200]]   # NO Bid @ 0.50 (YES Ask @ 0.50), size 200
        }
    }
    new_state_k = await k_handler.process_message(k_snapshot_msg)
    await k_handler._update_state(new_state_k)
    
    # Check K state
    assert k_handler._state.best_ask == 0.50
    assert k_handler._state.best_ask_size == 200
    
    # 3. Ingest Polymarket Book Update
    p_book_msg = {
        "ev": "book",
        "b": [["0.44", "500"]],
        "a": [["0.48", "150"]], # YES Ask @ 0.48, size 150
        "timestamp": str(int(time.time() * 1000))
    }
    new_state_p = await p_handler.process_message(p_book_msg)
    await p_handler._update_state(new_state_p)
    
    # Check P state
    assert p_handler._state.best_ask == 0.48
    assert p_handler._state.best_ask_size == 150
    
    # 4. Get Snapshot from Manager
    # We need to manually set timestamps to be "fresh" for the manager check
    now = int(time.time() * 1000)
    k_handler._state = k_handler._state.with_orderbook(
        list(k_handler._state.bid_levels), 
        list(k_handler._state.ask_levels), 
        update_ms=now
    )
    p_handler._state = p_handler._state.with_orderbook(
        list(p_handler._state.bid_levels), 
        list(p_handler._state.ask_levels), 
        update_ms=now
    )
    
    snapshot = await manager.get_fresh_snapshot()
    assert snapshot.is_valid()
    
    # 5. Detect Opportunity
    # Side 1: Kalshi Ask 0.50 + Poly Ask 0.48 = 0.98 (2% profit)
    # Depth: min(200, 150) = 150
    
    trigger = detector.detect(snapshot, now_ms=now)
    
    assert trigger.triggered
    assert trigger.reason == "raw_profit_trigger"
    assert trigger.max_profit_qty == 150
    assert abs(trigger.avg_entry_price - 0.98) < 0.0001
    
    # 6. Apply Delta to Kalshi and re-test
    # Clear cooldown to allow immediate re-trigger
    detector.cooldown_tracker.clear()
    
    # Delta: NO Bid @ 0.50 increases by 50 (from 200 to 250)
    k_delta_msg = {
        "type": "orderbook_delta",
        "sid": 1,
        "msg": {
            "price": 50,
            "delta": 50,
            "side": "no"
        }
    }
    new_state_k2 = await k_handler.process_message(k_delta_msg)
    await k_handler._update_state(new_state_k2)
    
    # Poly Ask is still 150 @ 0.48
    # Kalshi Ask is now 250 @ 0.50
    # New depth: min(150, 250) = 150 (Still 150 because Poly leg is bottleneck)
    
    # Refresh K state timestamp
    k_handler._state = k_handler._state.with_orderbook(
        list(k_handler._state.bid_levels), 
        list(k_handler._state.ask_levels), 
        update_ms=now + 10
    )
    
    snapshot2 = await manager.get_fresh_snapshot()
    trigger2 = detector.detect(snapshot2, now_ms=now + 10)
    
    assert trigger2.max_profit_qty == 150 
    
    # 7. Add liquidity to Poly and see qty increase
    detector.cooldown_tracker.clear()
    
    p_book_msg_2 = {
        "ev": "book",
        "b": [["0.44", "500"]],
        "a": [["0.48", "300"]], # YES Ask @ 0.48, size 300
        "timestamp": str(now + 20)
    }
    new_state_p2 = await p_handler.process_message(p_book_msg_2)
    await p_handler._update_state(new_state_p2)
    
    snapshot3 = await manager.get_fresh_snapshot()
    trigger3 = detector.detect(snapshot3, now_ms=now + 25)
    
    # Poly Ask: 300 @ 0.48
    # Kalshi Ask: 250 @ 0.50
    # Depth: min(300, 250) = 250
    assert trigger3.max_profit_qty == 250

if __name__ == "__main__":
    # For quick manual run
    import asyncio
    asyncio.run(test_full_ingestion_to_detector_flow())
