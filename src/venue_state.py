"""WebSocket ingestion with freshness guarantees for cross-venue arbitrage.

This module maintains continuously updated market state with explicit staleness
handling for both Polymarket and Kalshi venues.

Key concepts:
- VenueState: Per-venue market data with last_update_ms timestamp
- OrderLevel: Price/size pair for orderbook depth
- MarketSnapshot: Combined snapshot from both venues for arbitrage decisions
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


# Default staleness threshold from PRD config
# 5000ms (5s) to handle sporadic WS messages and prevent aggressive REST polling
# Matches legacy bot behavior
DEFAULT_STALE_MS = 5000


@dataclass(frozen=True)
class OrderLevel:
    """Single price level in the orderbook.

    Attributes:
        price: Price in dollars (0.0-1.0 for prediction markets)
        size: Number of contracts available at this price
    """

    price: float
    size: float

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("price must be non-negative")
        if self.size < 0:
            raise ValueError("size must be non-negative")


@dataclass(frozen=True)
class VenueState:
    """Per-venue market state with freshness tracking.

    Tracks best bid/ask (L1) and additional depth (L2) for one venue.
    All updates must come through immutable replacement to ensure thread safety.

    Attributes:
        venue: Venue identifier ("poly" or "kalshi")
        market_id: Market identifier (token_id for Poly, ticker for Kalshi)
        last_update_ms: Timestamp of last update in milliseconds since epoch
        best_bid: Best bid price (None if no bids)
        best_ask: Best ask price (None if no asks)
        best_bid_size: Size at best bid
        best_ask_size: Size at best ask
        bid_levels: L1 + L2 bid depth (sorted best to worst)
        ask_levels: L1 + L2 ask depth (sorted best to worst)
        is_fresh_after_reconnect: False until first fresh update after WS reconnect
    """

    venue: str
    market_id: str
    last_update_ms: int = 0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0
    bid_levels: tuple[OrderLevel, ...] = field(default_factory=tuple)
    ask_levels: tuple[OrderLevel, ...] = field(default_factory=tuple)
    is_fresh_after_reconnect: bool = True

    def __post_init__(self) -> None:
        if self.venue not in ("poly", "kalshi"):
            raise ValueError(f"venue must be 'poly' or 'kalshi', got {self.venue}")
        if not self.market_id:
            raise ValueError("market_id must not be empty")
        if self.last_update_ms < 0:
            raise ValueError("last_update_ms must be non-negative")

    def is_stale(self, now_ms: Optional[int] = None, stale_ms: int = DEFAULT_STALE_MS) -> bool:
        """Check if this state is stale.

        Args:
            now_ms: Current time in ms (defaults to actual current time)
            stale_ms: Staleness threshold in ms

        Returns:
            True if state is older than stale_ms or not fresh after reconnect
        """
        if not self.is_fresh_after_reconnect:
            return True
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return (now_ms - self.last_update_ms) > stale_ms

    def with_orderbook(
        self,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        update_ms: Optional[int] = None,
    ) -> "VenueState":
        """Create new state with updated orderbook.

        Args:
            bid_levels: New bid levels (sorted best to worst)
            ask_levels: New ask levels (sorted best to worst)
            update_ms: Update timestamp (defaults to current time)

        Returns:
            New VenueState with updated orderbook
        """
        if update_ms is None:
            update_ms = int(time.time() * 1000)

        best_bid = bid_levels[0].price if bid_levels else None
        best_ask = ask_levels[0].price if ask_levels else None
        best_bid_size = bid_levels[0].size if bid_levels else 0.0
        best_ask_size = ask_levels[0].size if ask_levels else 0.0

        return VenueState(
            venue=self.venue,
            market_id=self.market_id,
            last_update_ms=update_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            bid_levels=tuple(bid_levels),
            ask_levels=tuple(ask_levels),
            is_fresh_after_reconnect=True,
        )

    def with_bbo_update(
        self,
        best_bid: Optional[float],
        best_bid_size: float,
        best_ask: Optional[float],
        best_ask_size: float,
        update_ms: int,
    ) -> "VenueState":
        """Create new state with updated BBO while preserving existing depth levels.

        Used for Polymarket price_change events that only carry L1 (BBO) data.
        Unlike with_orderbook(), this preserves the existing bid_levels and
        ask_levels so that depth information is not destroyed.

        Args:
            best_bid: New best bid price (None if no bids)
            best_bid_size: Size at the new best bid
            best_ask: New best ask price (None if no asks)
            best_ask_size: Size at the new best ask
            update_ms: Update timestamp in milliseconds since epoch

        Returns:
            New VenueState with updated BBO values, existing depth levels,
            and is_fresh_after_reconnect=True
        """
        return VenueState(
            venue=self.venue,
            market_id=self.market_id,
            last_update_ms=update_ms,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            bid_levels=self.bid_levels,
            ask_levels=self.ask_levels,
            is_fresh_after_reconnect=True,
        )

    def mark_reconnecting(self) -> "VenueState":
        """Create new state marked as stale due to reconnect.

        On WS reconnect, cached data should be discarded until fresh update.

        Returns:
            New VenueState with is_fresh_after_reconnect=False
        """
        return VenueState(
            venue=self.venue,
            market_id=self.market_id,
            last_update_ms=self.last_update_ms,
            best_bid=self.best_bid,
            best_ask=self.best_ask,
            best_bid_size=self.best_bid_size,
            best_ask_size=self.best_ask_size,
            bid_levels=self.bid_levels,
            ask_levels=self.ask_levels,
            is_fresh_after_reconnect=False,
        )

    def total_bid_depth(self, max_levels: Optional[int] = None) -> float:
        """Calculate total bid depth across L1 + L2.

        Args:
            max_levels: Maximum levels to include (None for all)

        Returns:
            Sum of sizes across bid levels
        """
        levels = self.bid_levels[:max_levels] if max_levels else self.bid_levels
        return sum(level.size for level in levels)

    def total_ask_depth(self, max_levels: Optional[int] = None) -> float:
        """Calculate total ask depth across L1 + L2.

        Args:
            max_levels: Maximum levels to include (None for all)

        Returns:
            Sum of sizes across ask levels
        """
        levels = self.ask_levels[:max_levels] if max_levels else self.ask_levels
        return sum(level.size for level in levels)


@dataclass(frozen=True)
class MarketSnapshot:
    """Combined snapshot from both venues for arbitrage decisions.

    A snapshot is only valid if ALL FOUR LEGS are fresh:
    - poly_yes, poly_no, kalshi_yes (via kalshi state), kalshi_no (via kalshi state)

    Per-leg freshness ensures we never execute arbitrage with stale data on any side.

    Attributes:
        poly_yes_state: Polymarket YES token state
        poly_no_state: Polymarket NO token state
        kalshi_state: Kalshi venue state (includes both YES and NO sides)
        snapshot_ms: Timestamp when snapshot was taken
        aligned_market_id: Identifier for the aligned market pair
    """

    poly_yes_state: VenueState
    poly_no_state: VenueState
    kalshi_state: VenueState
    snapshot_ms: int
    aligned_market_id: str

    def __post_init__(self) -> None:
        if self.poly_yes_state.venue != "poly":
            raise ValueError("poly_yes_state must have venue='poly'")
        if self.poly_no_state.venue != "poly":
            raise ValueError("poly_no_state must have venue='poly'")
        if self.kalshi_state.venue != "kalshi":
            raise ValueError("kalshi_state must have venue='kalshi'")

    def is_valid(self, stale_ms: int = DEFAULT_STALE_MS) -> bool:
        """Check if snapshot is valid for trading.

        A snapshot is valid ONLY if ALL FOUR LEGS are fresh:
        1. poly_yes updated within stale_ms
        2. poly_no updated within stale_ms
        3. kalshi_yes (kalshi state) updated within stale_ms
        4. kalshi_no (kalshi state) updated within stale_ms
        5. All legs fresh after any reconnect

        Args:
            stale_ms: Staleness threshold in ms

        Returns:
            True if snapshot is valid for trading decisions
        """
        poly_yes_age = self.snapshot_ms - self.poly_yes_state.last_update_ms
        poly_no_age = self.snapshot_ms - self.poly_no_state.last_update_ms
        kalshi_age = self.snapshot_ms - self.kalshi_state.last_update_ms

        # ALL legs must be fresh
        if poly_yes_age > stale_ms or poly_no_age > stale_ms or kalshi_age > stale_ms:
            return False

        # ALL legs must be fresh after reconnect
        if not self.poly_yes_state.is_fresh_after_reconnect:
            return False
        if not self.poly_no_state.is_fresh_after_reconnect:
            return False
        if not self.kalshi_state.is_fresh_after_reconnect:
            return False

        # Enforce non-empty market IDs
        if not self.poly_yes_state.market_id or not self.poly_no_state.market_id or not self.kalshi_state.market_id:
            return False

        return True

    @property
    def poly_state(self) -> VenueState:
        """Deprecated: Use poly_yes_state or poly_no_state directly."""
        return self.poly_yes_state


def create_venue_state(venue: str, market_id: str) -> VenueState:
    """Factory function to create an initial venue state.

    Args:
        venue: Venue identifier ("poly" or "kalshi")
        market_id: Market identifier

    Returns:
        Fresh VenueState with no orderbook data
    """
    return VenueState(
        venue=venue,
        market_id=market_id,
        last_update_ms=0,
        is_fresh_after_reconnect=False,  # Not fresh until first WS update
    )


def create_snapshot(
    poly_yes_state: VenueState,
    poly_no_state: VenueState,
    kalshi_state: VenueState,
    aligned_market_id: str,
    snapshot_ms: Optional[int] = None,
) -> MarketSnapshot:
    """Create an immutable snapshot from current venue states.

    Args:
        poly_yes_state: Current Polymarket YES token state
        poly_no_state: Current Polymarket NO token state
        kalshi_state: Current Kalshi state
        aligned_market_id: Identifier for the aligned market pair
        snapshot_ms: Snapshot timestamp (defaults to current time)

    Returns:
        Immutable MarketSnapshot for decision making with per-leg freshness
    """
    if snapshot_ms is None:
        snapshot_ms = int(time.time() * 1000)

    return MarketSnapshot(
        poly_yes_state=poly_yes_state,
        poly_no_state=poly_no_state,
        kalshi_state=kalshi_state,
        snapshot_ms=snapshot_ms,
        aligned_market_id=aligned_market_id,
    )


def parse_kalshi_orderbook(
    yes_bids: list[tuple[str, str]],
    no_bids: list[tuple[str, str]],
) -> tuple[list[OrderLevel], list[OrderLevel]]:
    """Parse Kalshi orderbook response into OrderLevels.

    Kalshi returns yes_bids and no_bids. For YES side trading:
    - YES asks = NO bids (price inverted: ask_price = 1 - no_bid_price)
    - YES bids = YES bids directly

    Args:
        yes_bids: List of [price_str, size_str] for YES bids
        no_bids: List of [price_str, size_str] for NO bids

    Returns:
        Tuple of (bid_levels, ask_levels) sorted best to worst
    """
    # Parse YES bids directly
    bid_levels = []
    for price_str, size_str in yes_bids:
        price = float(price_str)
        size = float(size_str)
        if size > 0:
            bid_levels.append(OrderLevel(price=price, size=size))

    # YES asks = inverted NO bids
    ask_levels = []
    for price_str, size_str in no_bids:
        no_price = float(price_str)
        size = float(size_str)
        if size > 0:
            # Invert price: YES ask = 1 - NO bid
            yes_ask_price = 1.0 - no_price
            ask_levels.append(OrderLevel(price=yes_ask_price, size=size))

    # Sort: bids descending (best=highest), asks ascending (best=lowest)
    bid_levels.sort(key=lambda x: x.price, reverse=True)
    ask_levels.sort(key=lambda x: x.price)

    return bid_levels, ask_levels


def parse_poly_orderbook(
    bids: list[dict],
    asks: list[dict],
) -> tuple[list[OrderLevel], list[OrderLevel]]:
    """Parse Polymarket orderbook response into OrderLevels.

    Args:
        bids: List of {"price": str, "size": str} dicts
        asks: List of {"price": str, "size": str} dicts

    Returns:
        Tuple of (bid_levels, ask_levels) sorted best to worst
    """
    bid_levels = []
    for level in bids:
        price = float(level["price"])
        size = float(level["size"])
        if size > 0:
            bid_levels.append(OrderLevel(price=price, size=size))

    ask_levels = []
    for level in asks:
        price = float(level["price"])
        size = float(level["size"])
        if size > 0:
            ask_levels.append(OrderLevel(price=price, size=size))

    # Sort: bids descending (best=highest), asks ascending (best=lowest)
    bid_levels.sort(key=lambda x: x.price, reverse=True)
    ask_levels.sort(key=lambda x: x.price)

    return bid_levels, ask_levels
