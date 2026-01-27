
import unittest
from src.kalshi_ws import KalshiWSHandler
from src.venue_state import VenueState
from unittest.mock import MagicMock

class TestKalshiParsing(unittest.TestCase):
    def test_parse_snapshot(self):
        # Mock wrapper
        handler = KalshiWSHandler("test-market", MagicMock())
        handler._state = VenueState("kalshi", "test-market")
        
        # Simulated Snapshot
        # yes: bids for YES. [[price, qty]]
        # no: bids for NO.
        # If Yes Bid is 40cents (0.40).
        # If No Bid is 30cents (0.30) -> This implies YES Ask is 70cents (0.70).
        
        msg = {
            "type": "orderbook_snapshot",
            "msg": {
                "yes": [[40, 100], [39, 50]], # Bids: 0.40, 0.39
                "no": [[30, 100], [29, 50]]   # No Bids: 0.30, 0.29 -> Yes Asks: 0.70, 0.71
            }
        }
        
        new_state = asyncio.run(handler.process_message(msg))
        
        print(f"Best Bid (YES): {new_state.best_bid}")
        print(f"Best Ask (YES): {new_state.best_ask}")
        print(f"Bid Levels: {new_state.bid_levels}")
        print(f"Ask Levels: {new_state.ask_levels}")
        
        # Verify
        self.assertEqual(new_state.best_bid, 0.40)
        self.assertEqual(new_state.best_ask, 0.70)
        
    def test_parse_delta(self):
        # Mock wrapper
        handler = KalshiWSHandler("test-market", MagicMock())
        handler._state = VenueState("kalshi", "test-market")
        
        # Init with snapshot
        msg = {
            "type": "orderbook_snapshot",
            "msg": {
                "yes": [[40, 100]],
                "no": [[30, 100]]
            }
        }
        asyncio.run(handler.process_message(msg))
        
        # Delta: Update NO bid (side="no")
        # Change price 35 (NO Bid) to qty 50.
        # NO Bid 0.35 -> YES Ask 0.65.
        delta_msg = {
            "type": "orderbook_delta",
            "msg": {
                "price": 35,
                "delta": 50,
                "side": "no"
            }
        }
        new_state = asyncio.run(handler.process_message(delta_msg))
        
        print(f"Best Bid: {new_state.best_bid}")
        print(f"Best Ask: {new_state.best_ask}")
        
        self.assertEqual(new_state.best_ask, 0.65)

import asyncio
if __name__ == "__main__":
    unittest.main()
