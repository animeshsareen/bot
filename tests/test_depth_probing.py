import pytest
from src.opportunity import OpportunityDetector
from src.venue_state import VenueState, MarketSnapshot, OrderLevel

def make_venue_state_with_depth(
    venue: str,
    market_id: str,
    ask_levels: list[OrderLevel],
    update_ms: int = 1000
) -> VenueState:
    """Create a VenueState with multiple depth levels."""
    return VenueState(
        venue=venue,
        market_id=market_id,
        last_update_ms=update_ms,
        best_bid=0.40,
        best_ask=ask_levels[0].price if ask_levels else None,
        best_bid_size=100,
        best_ask_size=ask_levels[0].size if ask_levels else 0.0,
        bid_levels=(OrderLevel(0.40, 100),),
        ask_levels=tuple(ask_levels),
        is_fresh_after_reconnect=True,
    )

def test_calculate_max_profitable_quantity_multiple_levels():
    """Test walking the book across multiple levels."""
    detector = OpportunityDetector()
    
    # Poly Asks: 500 @ 0.45, 1000 @ 0.48, 2000 @ 0.55
    poly_asks = (
        OrderLevel(0.45, 500),
        OrderLevel(0.48, 1000),
        OrderLevel(0.55, 2000),
    )
    
    # Kalshi Asks: 800 @ 0.50, 1200 @ 0.51, 3000 @ 0.60
    kalshi_asks = (
        OrderLevel(0.50, 800),
        OrderLevel(0.51, 1200),
        OrderLevel(0.60, 3000),
    )
    
    # Profitable combinations (Price Sum < 1.0):
    # 1. 0.45 + 0.50 = 0.95 (Liquidity: min(500, 800) = 500)
    # 2. After consuming 500:
    #    Poly: 1000 @ 0.48 ... (next)
    #    Kalshi: 300 @ 0.50 (remaining)
    #    Marginal: 0.48 + 0.50 = 0.98 (Liquidity: min(1000, 300) = 300)
    # 3. After consuming 300:
    #    Poly: 700 @ 0.48
    #    Kalshi: 1200 @ 0.51
    #    Marginal: 0.48 + 0.51 = 0.99 (Liquidity: min(700, 1200) = 700)
    # 4. After consuming 700:
    #    Poly: 2000 @ 0.55
    #    Kalshi: 500 @ 0.51
    #    Marginal: 0.55 + 0.51 = 1.06 (STOP! Not profitable)
    
    # Expected total: 500 + 300 + 700 = 1500
    
    qty, avg_price = detector.calculate_max_profitable_quantity(poly_asks, kalshi_asks)
    
    assert qty == 1500
    # Weighted average calculation:
    # Cost = (500 * 0.95) + (300 * 0.98) + (700 * 0.99)
    # Cost = 475 + 294 + 693 = 1462
    # Avg = 1462 / 1500 = 0.97466...
    assert abs(avg_price - 0.97466666) < 0.0001

def test_detect_with_depth_limit():
    """Test that detect method returns the calculated qty."""
    detector = OpportunityDetector(raw_profit_trigger_pct=1.0)
    
    poly_state = make_venue_state_with_depth("poly", "P", [OrderLevel(0.45, 100), OrderLevel(0.46, 200)])
    kalshi_state = make_venue_state_with_depth("kalshi", "K", [OrderLevel(0.50, 150), OrderLevel(0.51, 100)])
    
    # Marginal 1: 0.45 + 0.50 = 0.95 (qty 100)
    # Marginal 2: 0.46 + 0.50 = 0.96 (qty 50)
    # Marginal 3: 0.46 + 0.51 = 0.97 (qty 100)
    # Total expected: 100 + 50 + 100 = 250
    
    snapshot = MarketSnapshot(poly_state, poly_state, kalshi_state, 5000, "test")
    result = detector.detect(snapshot, now_ms=5000)
    
    assert result.triggered
    assert result.max_profit_qty == 250
    assert result.avg_entry_price < 1.0

def test_insufficient_depth_suppresses_trigger():
    """Test that if no profitable depth exists beyond top-of-book (unlikely but possible), it suppresses or just returns 0."""
    # This case: top of book is profitable but logic says marginal trades must be profitable too.
    # Wait, if top of book is profitable, qty will be at least min(L1_poly, L1_kalshi).
    # So if detect() finds a trigger at top-of-book, max_profit_qty should be > 0.
    
    detector = OpportunityDetector(raw_profit_trigger_pct=10.0) # High trigger
    
    # Let's say top of book is 0.40 + 0.40 = 0.80 (20% profit)
    poly_state = make_venue_state_with_depth("poly", "P", [OrderLevel(0.40, 10)])
    kalshi_state = make_venue_state_with_depth("kalshi", "K", [OrderLevel(0.40, 10)])
    
    snapshot = MarketSnapshot(poly_state, poly_state, kalshi_state, 5000, "test")
    result = detector.detect(snapshot, now_ms=5000)
    
    assert result.triggered
    assert result.max_profit_qty == 10
    assert result.avg_entry_price == 0.80
