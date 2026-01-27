"""Binary micro-arbitrage for cross-venue Y/N contracts.

Identifies arbitrage opportunities by buying YES on one venue and NO on another,
such that the combined average cost is < $0.99 (99 cents).
Uses a cooldown mechanism to avoid rapid re-triggering on the same opportunity.

Key concepts:
- cross_cost: Cost to buy YES on one venue and NO on the other
- raw_profit: 1 - cross_cost (profit if exactly one contract pays out $1)
- Trigger when raw_profit_trigger_pct met OR cross_cost <= max_combined_cost_cap
"""

from dataclasses import dataclass
from typing import Optional
import time
import logging

from .venue_state import MarketSnapshot


logger = logging.getLogger(__name__)


# Default config values from PRD
DEFAULT_RAW_PROFIT_TRIGGER_PCT = 2.0
DEFAULT_MAX_COMBINED_COST_CAP = 0.99
DEFAULT_COOLDOWN_MS = 500
DEFAULT_COOLDOWN_IMPROVEMENT_THRESHOLD = 0.002


@dataclass(frozen=True)
class OpportunityTrigger:
    """Result of opportunity detection.

    Attributes:
        triggered: Whether opportunity meets trigger criteria
        cross_cost: Cost to buy YES on one venue and NO on the other
        raw_profit_pct: Raw profit percentage (1 - cross_cost) * 100
        reason: Human-readable reason for trigger/skip
        poly_ask: Best ask from Polymarket (YES leg)
        kalshi_ask: Best ask from Kalshi (YES leg)
        snapshot_ms: Timestamp of the snapshot used
        strategy_id: Identifier for cooldown tracking
        strategy_path: Which arbitrage path (1: Buy Poly YES + buy Kalshi NO, 2: Buy Kalshi YES + Buy Poly NO)
    """

    triggered: bool
    cross_cost: Optional[float]
    raw_profit_pct: Optional[float]
    reason: str
    poly_ask: Optional[float]
    kalshi_ask: Optional[float]
    snapshot_ms: int
    strategy_id: str
    poly_ask_size: Optional[float] = 0.0
    kalshi_ask_size: Optional[float] = 0.0
    max_profit_qty: int = 0
    avg_entry_price: float = 0.0
    strategy_path: int = 0  # 1 or 2, only set when triggered

    def log_details(self) -> None:
        """Log trigger details at INFO level."""
        if self.triggered:
            logger.info(
                "TRIGGER: strategy=%s cross_cost=%.4f raw_profit_pct=%.2f%% "
                "poly_ask=%.4f (sz=%.1f) kalshi_ask=%.4f (sz=%.1f) qty=%d avg_price=%.4f reason=%s",
                self.strategy_id,
                self.cross_cost,
                self.raw_profit_pct,
                self.poly_ask,
                self.poly_ask_size,
                self.kalshi_ask,
                self.kalshi_ask_size,
                self.max_profit_qty,
                self.avg_entry_price,
                self.reason,
            )
        else:
            logger.info(
                "SKIP: strategy=%s cross_cost=%s raw_profit_pct=%s "
                "poly_ask=%s (sz=%s) kalshi_ask=%s (sz=%s) qty=%d reason=%s",
                self.strategy_id,
                f"{self.cross_cost:.4f}" if self.cross_cost else "N/A",
                f"{self.raw_profit_pct:.2f}%" if self.raw_profit_pct else "N/A",
                f"{self.poly_ask:.4f}" if self.poly_ask else "N/A",
                f"{self.poly_ask_size:.1f}" if self.poly_ask_size is not None else "N/A",
                f"{self.kalshi_ask:.4f}" if self.kalshi_ask else "N/A",
                f"{self.kalshi_ask_size:.1f}" if self.kalshi_ask_size is not None else "N/A",
                self.max_profit_qty,
                self.reason,
            )


class CooldownTracker:
    """Tracks cooldown state per strategy.

    Cooldown is 500ms unless cross_cost improves by >= 0.002 from last trigger.
    """

    def __init__(
        self,
        cooldown_ms: int = DEFAULT_COOLDOWN_MS,
        improvement_threshold: float = DEFAULT_COOLDOWN_IMPROVEMENT_THRESHOLD,
    ):
        self.cooldown_ms = cooldown_ms
        self.improvement_threshold = improvement_threshold
        # strategy_id -> (last_trigger_ms, last_cross_cost)
        self._last_triggers: dict[str, tuple[int, float]] = {}

    def is_on_cooldown(
        self,
        strategy_id: str,
        current_cross_cost: float,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Check if strategy is on cooldown.

        Args:
            strategy_id: Strategy identifier
            current_cross_cost: Current cross cost
            now_ms: Current time in ms (defaults to actual current time)

        Returns:
            True if on cooldown and opportunity should be skipped
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        if strategy_id not in self._last_triggers:
            return False

        last_trigger_ms, last_cross_cost = self._last_triggers[strategy_id]

        # Check if cooldown period has elapsed
        elapsed_ms = now_ms - last_trigger_ms
        if elapsed_ms >= self.cooldown_ms:
            return False

        # Check if cross_cost improved by threshold
        improvement = last_cross_cost - current_cross_cost
        if improvement >= self.improvement_threshold:
            return False

        return True

    def record_trigger(
        self,
        strategy_id: str,
        cross_cost: float,
        trigger_ms: Optional[int] = None,
    ) -> None:
        """Record a trigger for cooldown tracking.

        Args:
            strategy_id: Strategy identifier
            cross_cost: Cross cost at trigger time
            trigger_ms: Trigger timestamp (defaults to current time)
        """
        if trigger_ms is None:
            trigger_ms = int(time.time() * 1000)
        self._last_triggers[strategy_id] = (trigger_ms, cross_cost)

    def clear(self, strategy_id: Optional[str] = None) -> None:
        """Clear cooldown state.

        Args:
            strategy_id: Strategy to clear (None clears all)
        """
        if strategy_id is None:
            self._last_triggers.clear()
        elif strategy_id in self._last_triggers:
            del self._last_triggers[strategy_id]


class OpportunityDetector:
    """Detects binary arbitrage opportunities from market snapshots.

    Looks for cases where you can buy YES on one venue and NO on another,
    with a combined average cost < $0.99.
    """

    def __init__(
        self,
        raw_profit_trigger_pct: float = DEFAULT_RAW_PROFIT_TRIGGER_PCT,
        max_combined_cost_cap: float = DEFAULT_MAX_COMBINED_COST_CAP,
        cooldown_tracker: Optional[CooldownTracker] = None,
    ):
        """Initialize detector.

        Args:
            raw_profit_trigger_pct: Minimum raw profit % to trigger
            max_combined_cost_cap: Alternative trigger if cross_cost <= this
            cooldown_tracker: Cooldown tracker instance (creates default if None)
        """
        self.raw_profit_trigger_pct = raw_profit_trigger_pct
        self.max_combined_cost_cap = max_combined_cost_cap
        self.cooldown_tracker = cooldown_tracker or CooldownTracker()

    def detect(
        self,
        snapshot: MarketSnapshot,
        now_ms: Optional[int] = None,
    ) -> OpportunityTrigger:
        """Detect if snapshot presents a binary arbitrage opportunity.

        Evaluates two paths:
        - Path 1: Buy YES on Poly, buy NO on Kalshi (cost: poly_yes_ask + (1 - kalshi_yes_bid))
        - Path 2: Buy YES on Kalshi, buy NO on Poly (cost: kalshi_yes_ask + poly_no_ask)

        Takes the cheaper path and checks if cross_cost < $0.99.

        Args:
            snapshot: Market snapshot with per-leg states
            now_ms: Current time for cooldown check (defaults to actual time)

        Returns:
            OpportunityTrigger with detection result
        """
        strategy_id = snapshot.aligned_market_id

        # Check for missing data (per-leg freshness validation)
        poly_yes_ask = snapshot.poly_yes_state.best_ask
        poly_yes_bid = snapshot.poly_yes_state.best_bid
        poly_no_ask = snapshot.poly_no_state.best_ask
        poly_no_bid = snapshot.poly_no_state.best_bid
        kalshi_yes_ask = snapshot.kalshi_state.best_ask
        kalshi_yes_bid = snapshot.kalshi_state.best_bid

        if (poly_yes_ask is None or poly_no_ask is None or kalshi_yes_ask is None or
            poly_yes_bid is None or poly_no_bid is None or kalshi_yes_bid is None):
            return OpportunityTrigger(
                triggered=False,
                cross_cost=None,
                raw_profit_pct=None,
                reason="missing_ask_or_bid",
                poly_ask=poly_yes_ask,
                kalshi_ask=kalshi_yes_ask,
                poly_ask_size=None,
                kalshi_ask_size=None,
                snapshot_ms=snapshot.snapshot_ms,
                strategy_id=strategy_id,
            )

        # Get sizes for logging
        poly_yes_size = snapshot.poly_yes_state.best_ask_size
        kalshi_yes_size = snapshot.kalshi_state.best_ask_size
        poly_no_size = snapshot.poly_no_state.best_ask_size
        
        # For Path 1: Poly YES + Kalshi NO
        # Kalshi NO size is derived from Kalshi YES BID size
        kalshi_yes_bid_size = snapshot.kalshi_state.best_bid_size
        
        path_1_poly_size = poly_yes_size
        path_1_kalshi_size = kalshi_yes_bid_size # Kalshi NO size
        
        path_2_kalshi_size = kalshi_yes_size
        path_2_poly_size = poly_no_size

        # Compute cross cost: find minimum cost arbitrage
        # Option 1: Buy YES Poly, Buy NO Kalshi = poly_yes_ask + (1 - kalshi_yes_bid)
        #   Kalshi only trades YES, so buying NO = selling YES at bid = cost of (1 - bid)
        # Option 2: Buy YES Kalshi, Buy NO Poly = kalshi_yes_ask + poly_no_ask
        #   Polymarket trades YES and NO separately, so buy NO directly at poly_no_ask
        cost_poly_yes_kalshi_no = poly_yes_ask + (1.0 - kalshi_yes_bid)
        cost_kalshi_yes_poly_no = kalshi_yes_ask + poly_no_ask
        cross_cost = min(cost_poly_yes_kalshi_no, cost_kalshi_yes_poly_no)
        raw_profit_pct = (1.0 - cross_cost) * 100

        # Determine which path is cheaper to validate depth correctly
        path_1_cheaper = cost_poly_yes_kalshi_no < cost_kalshi_yes_poly_no

        # Check cooldown
        if self.cooldown_tracker.is_on_cooldown(strategy_id, cross_cost, now_ms):
            return OpportunityTrigger(
                triggered=False,
                cross_cost=cross_cost,
                raw_profit_pct=raw_profit_pct,
                reason="cooldown",
                poly_ask=poly_yes_ask,
                kalshi_ask=kalshi_yes_ask,
                poly_ask_size=path_1_poly_size if path_1_cheaper else path_2_poly_size,
                kalshi_ask_size=path_1_kalshi_size if path_1_cheaper else path_2_kalshi_size,
                snapshot_ms=snapshot.snapshot_ms,
                strategy_id=strategy_id,
            )

        # Check trigger conditions
        profit_trigger_met = raw_profit_pct >= self.raw_profit_trigger_pct
        cost_cap_met = cross_cost <= self.max_combined_cost_cap

        # Compute L1-based executable quantity (no VWAP walk)
        if path_1_cheaper:
            poly_l1 = path_1_poly_size
            kalshi_l1 = path_1_kalshi_size
        else:
            poly_l1 = path_2_poly_size
            kalshi_l1 = path_2_kalshi_size
        max_profit_qty = int(min(poly_l1, kalshi_l1, 5))
        avg_entry_price = cross_cost if max_profit_qty > 0 else 0.0

        if profit_trigger_met:
            self.cooldown_tracker.record_trigger(strategy_id, cross_cost, now_ms)
            return OpportunityTrigger(
                triggered=True,
                cross_cost=cross_cost,
                raw_profit_pct=raw_profit_pct,
                reason="raw_profit_trigger",
                poly_ask=poly_yes_ask,
                kalshi_ask=kalshi_yes_ask,
                poly_ask_size=path_1_poly_size if path_1_cheaper else path_2_poly_size,
                kalshi_ask_size=path_1_kalshi_size if path_1_cheaper else path_2_kalshi_size,
                snapshot_ms=snapshot.snapshot_ms,
                strategy_id=strategy_id,
                max_profit_qty=max_profit_qty,
                avg_entry_price=avg_entry_price,
                strategy_path=1 if path_1_cheaper else 2
            )

        if cost_cap_met:
            self.cooldown_tracker.record_trigger(strategy_id, cross_cost, now_ms)
            return OpportunityTrigger(
                triggered=True,
                cross_cost=cross_cost,
                raw_profit_pct=raw_profit_pct,
                reason="cost_cap_trigger",
                poly_ask=poly_yes_ask,
                kalshi_ask=kalshi_yes_ask,
                poly_ask_size=path_1_poly_size if path_1_cheaper else path_2_poly_size,
                kalshi_ask_size=path_1_kalshi_size if path_1_cheaper else path_2_kalshi_size,
                snapshot_ms=snapshot.snapshot_ms,
                strategy_id=strategy_id,
                max_profit_qty=max_profit_qty,
                avg_entry_price=avg_entry_price,
                strategy_path=1 if path_1_cheaper else 2
            )

        # No trigger
        return OpportunityTrigger(
            triggered=False,
            cross_cost=cross_cost,
            raw_profit_pct=raw_profit_pct,
            reason="below_threshold",
            poly_ask=poly_yes_ask,
            kalshi_ask=kalshi_yes_ask,
            poly_ask_size=path_1_poly_size if path_1_cheaper else path_2_poly_size,
            kalshi_ask_size=path_1_kalshi_size if path_1_cheaper else path_2_kalshi_size,
            snapshot_ms=snapshot.snapshot_ms,
            strategy_id=strategy_id,
            max_profit_qty=0,
            avg_entry_price=0.0
        )


def compute_cross_cost_yes_no(yes_price: float, no_price: float) -> float:
    """Compute cross cost for binary arbitrage (YES + NO).

    Args:
        yes_price: Price to buy YES on one venue
        no_price: Price to buy NO on another venue

    Returns:
        Combined cost to buy both legs
    """
    return yes_price + no_price


def compute_raw_profit_pct(cross_cost: float) -> float:
    """Compute raw profit percentage from cross cost.

    Args:
        cross_cost: Sum of best asks

    Returns:
        Raw profit percentage (1 - cross_cost) * 100
    """
    return (1.0 - cross_cost) * 100
