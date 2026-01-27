"""Chase-based reconciliation of short leg for cross-venue arbitrage.

Attempts to equalize fills before flattening excess exposure.
Submits taker chase orders on the short leg to match fills.

Key concepts:
- common_qty: Minimum of both venue fills (matched quantity)
- short_leg: Venue with fewer fills that needs chasing
- Chase: Submit aggressive taker orders to catch up
"""

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from enum import Enum
import asyncio
import time
import logging

from .venue_execution import (
    DualFillRecord, KalshiFillRecord, PolyFillRecord,
    OrderRequest, OrderResult, OrderSide, OrderType,
)
from .async_executor import ExecutorState


logger = logging.getLogger(__name__)


# Default config values from PRD
DEFAULT_MAX_CHASE_STEPS = 5
DEFAULT_MAX_HEDGE_RECONCILE_SEC = 10
DEFAULT_CHASE_STEP_CENTS_KALSHI = 1
DEFAULT_CHASE_STEP_DOLLARS_POLY = 0.01


class ShortLeg(Enum):
    """Identifies which leg is short (needs chasing)."""
    NONE = "none"
    KALSHI = "kalshi"
    POLY = "poly"


class ReconcileOutcome(Enum):
    """Outcome of reconciliation attempt."""
    EQUALIZED = "equalized"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    MAX_STEPS_REACHED = "max_steps_reached"
    CANCELLED = "cancelled"
    NO_CHASE_NEEDED = "no_chase_needed"


@dataclass(frozen=True)
class ChaseStep:
    """Record of a single chase step.

    Attributes:
        step_number: Step index (1-based)
        venue: Venue being chased
        price: Price used for chase order
        quantity_requested: Quantity requested
        quantity_filled: Quantity actually filled
        cumulative_filled: Total filled across all chase steps
        latency_ms: Time for this step
    """

    step_number: int
    venue: str
    price: float
    quantity_requested: int
    quantity_filled: int
    cumulative_filled: int
    latency_ms: float


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of chase-based reconciliation.

    Attributes:
        outcome: Final outcome of reconciliation
        initial_common_qty: Common qty before chasing
        final_common_qty: Common qty after chasing
        short_leg: Which leg was short
        chase_steps_taken: Number of chase steps executed
        chase_steps: Details of each chase step
        total_ms: Total time spent reconciling
        orders_cancelled: Number of orders cancelled before flatten
    """

    outcome: ReconcileOutcome
    initial_common_qty: int
    final_common_qty: int
    short_leg: ShortLeg
    chase_steps_taken: int
    chase_steps: tuple[ChaseStep, ...]
    total_ms: float
    orders_cancelled: int = 0

    def log_details(self) -> None:
        """Log reconciliation details at INFO level."""
        logger.info(
            "RECONCILE: outcome=%s initial_common=%d final_common=%d "
            "short_leg=%s steps=%d total_ms=%.1f cancelled=%d",
            self.outcome.value,
            self.initial_common_qty,
            self.final_common_qty,
            self.short_leg.value,
            self.chase_steps_taken,
            self.total_ms,
            self.orders_cancelled,
        )


# Type for order submission and cancellation callbacks
OrderSubmitter = Callable[[OrderRequest], Awaitable[OrderResult]]
OrderCanceller = Callable[[str], Awaitable[bool]]  # order_id -> success


def identify_short_leg(
    kalshi_filled: int,
    poly_filled: int,
) -> tuple[ShortLeg, int]:
    """Identify which leg is short and by how much.

    Args:
        kalshi_filled: Kalshi fill quantity
        poly_filled: Polymarket fill quantity

    Returns:
        Tuple of (short leg, deficit quantity)
    """
    if kalshi_filled == poly_filled:
        return ShortLeg.NONE, 0
    elif kalshi_filled < poly_filled:
        return ShortLeg.KALSHI, poly_filled - kalshi_filled
    else:
        return ShortLeg.POLY, kalshi_filled - poly_filled


def compute_chase_price(
    current_price: float,
    step: int,
    venue: str,
    chase_step_cents_kalshi: int = DEFAULT_CHASE_STEP_CENTS_KALSHI,
    chase_step_dollars_poly: float = DEFAULT_CHASE_STEP_DOLLARS_POLY,
) -> float:
    """Compute aggressive chase price for a step.

    Args:
        current_price: Current best ask price
        step: Chase step number (1-based)
        venue: Venue being chased
        chase_step_cents_kalshi: Price walk per step for Kalshi
        chase_step_dollars_poly: Price walk per step for Poly

    Returns:
        Aggressive price for chase order
    """
    if venue == "kalshi":
        # Walk up by cents per step
        return current_price + (step * chase_step_cents_kalshi / 100.0)
    else:
        # Walk up by dollars per step
        return current_price + (step * chase_step_dollars_poly)


class ChaseReconciler:
    """Chase-based reconciliation for imbalanced fills.

    Attempts to equalize fills by chasing the short leg.
    """

    def __init__(
        self,
        max_chase_steps: int = DEFAULT_MAX_CHASE_STEPS,
        max_reconcile_sec: float = DEFAULT_MAX_HEDGE_RECONCILE_SEC,
        chase_step_cents_kalshi: int = DEFAULT_CHASE_STEP_CENTS_KALSHI,
        chase_step_dollars_poly: float = DEFAULT_CHASE_STEP_DOLLARS_POLY,
    ):
        """Initialize reconciler.

        Args:
            max_chase_steps: Maximum chase steps before giving up
            max_reconcile_sec: Maximum time for reconciliation
            chase_step_cents_kalshi: Price walk per step for Kalshi
            chase_step_dollars_poly: Price walk per step for Poly
        """
        self.max_chase_steps = max_chase_steps
        self.max_reconcile_sec = max_reconcile_sec
        self.chase_step_cents_kalshi = chase_step_cents_kalshi
        self.chase_step_dollars_poly = chase_step_dollars_poly
        self._cancel_requested = False
        self._outstanding_orders: list[str] = []

    def request_cancel(self) -> None:
        """Request cancellation of reconciliation."""
        self._cancel_requested = True

    async def reconcile(
        self,
        dual_fill: DualFillRecord,
        kalshi_market_id: str,
        poly_market_id: str,
        side: OrderSide,
        get_kalshi_ask: Callable[[], Optional[float]],
        get_poly_ask: Callable[[], Optional[float]],
        submit_order: OrderSubmitter,
        cancel_order: Optional[OrderCanceller] = None,
        check_profitability: Optional[Callable[[float, float], bool]] = None,
    ) -> ReconciliationResult:
        """Attempt to reconcile imbalanced fills.

        Args:
            dual_fill: Current fill state
            kalshi_market_id: Kalshi market ticker
            poly_market_id: Polymarket token ID
            side: Order side for the chase (usually same as original)
            get_kalshi_ask: Callback to get fresh Kalshi ask
            get_poly_ask: Callback to get fresh Poly ask
            submit_order: Callback to submit orders
            cancel_order: Callback to cancel orders (optional)
            check_profitability: Optional callback to check if chase is still profitable

        Returns:
            ReconciliationResult with outcome details
        """
        start_ms = time.time() * 1000
        self._cancel_requested = False
        self._outstanding_orders = []

        kalshi_filled = dual_fill.kalshi_fill.filled_qty
        poly_filled = dual_fill.poly_fill.filled_qty if dual_fill.poly_fill else 0

        initial_common = min(kalshi_filled, poly_filled)
        short_leg, deficit = identify_short_leg(kalshi_filled, poly_filled)

        # No chase needed if balanced
        if short_leg == ShortLeg.NONE:
            return ReconciliationResult(
                outcome=ReconcileOutcome.NO_CHASE_NEEDED,
                initial_common_qty=initial_common,
                final_common_qty=initial_common,
                short_leg=short_leg,
                chase_steps_taken=0,
                chase_steps=(),
                total_ms=time.time() * 1000 - start_ms,
            )

        # Chase the short leg
        chase_steps: list[ChaseStep] = []
        cumulative_filled = 0
        # Chase the short leg
        chase_steps: list[ChaseStep] = []
        cumulative_filled = 0
        
        # Determine target market and price source
        if short_leg == ShortLeg.KALSHI:
            market_id = kalshi_market_id
            get_price = get_kalshi_ask
        else:
            market_id = poly_market_id
            get_price = get_poly_ask

        for step in range(1, self.max_chase_steps + 1):
            # Check cancellation
            if self._cancel_requested:
                break

            # Check timeout
            elapsed_sec = (time.time() * 1000 - start_ms) / 1000
            if elapsed_sec >= self.max_reconcile_sec:
                break

            # Get fresh price
            current_price = get_price()
            if current_price is None:
                # Can't chase without price
                break

            # Check profitability if callback provided
            # We need both prices to check cross-venue profit/loss
            if check_profitability:
                k_ask = get_kalshi_ask()
                p_ask = get_poly_ask()
                if k_ask is not None and p_ask is not None:
                    if not check_profitability(k_ask, p_ask):
                        # Stop chase if no longer profitable (or exceeds loss limit)
                        break

            # Compute chase price
            chase_price = compute_chase_price(
                current_price,
                step,
                short_leg.value,
                self.chase_step_cents_kalshi,
                self.chase_step_dollars_poly,
            )

            # Remaining deficit
            remaining = deficit - cumulative_filled
            if remaining <= 0:
                break

            # Submit chase order
            order = OrderRequest(
                venue=short_leg.value,
                market_id=market_id,
                side=side,
                price=chase_price,
                quantity=remaining,
                order_type=OrderType.IOC,
            )

            step_start = time.time() * 1000
            result = await submit_order(order)
            step_latency = time.time() * 1000 - step_start

            if result.order_id:
                self._outstanding_orders.append(result.order_id)

            cumulative_filled += result.filled_qty

            chase_steps.append(ChaseStep(
                step_number=step,
                venue=short_leg.value,
                price=chase_price,
                quantity_requested=remaining,
                quantity_filled=result.filled_qty,
                cumulative_filled=cumulative_filled,
                latency_ms=step_latency,
            ))

            # Check if fully reconciled
            if cumulative_filled >= deficit:
                break

        # Cancel outstanding orders before returning
        orders_cancelled = 0
        if cancel_order and self._outstanding_orders:
            for order_id in self._outstanding_orders:
                try:
                    if await cancel_order(order_id):
                        orders_cancelled += 1
                except Exception:
                    pass

        total_ms = time.time() * 1000 - start_ms
        final_common = initial_common + cumulative_filled

        # Determine outcome
        if cumulative_filled >= deficit:
            outcome = ReconcileOutcome.EQUALIZED
        elif self._cancel_requested:
            outcome = ReconcileOutcome.CANCELLED
        elif (total_ms / 1000) >= self.max_reconcile_sec:
            outcome = ReconcileOutcome.TIMEOUT
        elif len(chase_steps) >= self.max_chase_steps:
            outcome = ReconcileOutcome.MAX_STEPS_REACHED
        else:
            outcome = ReconcileOutcome.PARTIAL

        return ReconciliationResult(
            outcome=outcome,
            initial_common_qty=initial_common,
            final_common_qty=final_common,
            short_leg=short_leg,
            chase_steps_taken=len(chase_steps),
            chase_steps=tuple(chase_steps),
            total_ms=total_ms,
            orders_cancelled=orders_cancelled,
        )


def compute_common_qty(kalshi_filled: int, poly_filled: int) -> int:
    """Compute matched quantity across both venues.

    Args:
        kalshi_filled: Kalshi fill quantity
        poly_filled: Polymarket fill quantity

    Returns:
        Minimum of both fills
    """
    return min(kalshi_filled, poly_filled)


class SimpleReconciler:
    """Simple post-execution reconciliation of fills across venues.

    Handles basic reconciliation logic for fills without chase attempts.
    Accepts small imbalances but forces flattening on unhedged positions.

    Key philosophy:
    - Accept balanced fills or small imbalances (case 1-2)
    - Force flatten on unhedged positions (case 3)
    - Log error but flatten on impossible cases (case 4)
    """

    async def reconcile_fills(
        self,
        kalshi_fill: int,
        poly_fill: int,
    ) -> ExecutorState:
        """Reconcile fills from both venues.

        Implements 4-case logic:
        1. kalshi == poly > 0 → return DONE_EQUAL (balanced, good!)
        2. kalshi > poly > 0 → log warning, accept imbalance, return DONE_EQUAL
        3. kalshi > 0, poly == 0 → log error, unhedged position, return FLAT_FORCED
        4. poly > kalshi → log error, shouldn't happen, return FLAT_FORCED

        Args:
            kalshi_fill: Number of contracts filled on Kalshi
            poly_fill: Number of contracts filled on Polymarket

        Returns:
            ExecutorState indicating next action (DONE_EQUAL or FLAT_FORCED)
        """
        # Case 1: Balanced fills - ideal outcome
        if kalshi_fill == poly_fill and kalshi_fill > 0:
            logger.info(
                "RECONCILE: fills balanced kalshi=%d poly=%d status=DONE_EQUAL",
                kalshi_fill,
                poly_fill,
            )
            return ExecutorState.DONE_EQUAL

        # Case 2: Kalshi more filled than Poly, but Poly has some
        # Small imbalance acceptable - one leg filled more due to market conditions
        if kalshi_fill > poly_fill > 0:
            logger.warning(
                "RECONCILE: imbalanced fills (Poly short) kalshi=%d poly=%d "
                "diff=%d status=DONE_EQUAL accepting_imbalance",
                kalshi_fill,
                poly_fill,
                kalshi_fill - poly_fill,
            )
            return ExecutorState.DONE_EQUAL

        # Case 3: Kalshi has position but Poly completely unfilled (unhedged)
        if kalshi_fill > 0 and poly_fill == 0:
            logger.error(
                "RECONCILE: unhedged position kalshi=%d poly=0 status=FLAT_FORCED "
                "forcing_flatten",
                kalshi_fill,
            )
            return ExecutorState.FLAT_FORCED

        # Case 4: Poly more filled than Kalshi (shouldn't happen with gating)
        if poly_fill > kalshi_fill:
            logger.error(
                "RECONCILE: impossible state poly=%d kalshi=%d status=FLAT_FORCED "
                "poly_exceeds_kalshi",
                poly_fill,
                kalshi_fill,
            )
            return ExecutorState.FLAT_FORCED

        # Edge case: both zero (unlikely but handled)
        logger.warning(
            "RECONCILE: both venues zero kalshi=%d poly=%d status=DONE_EQUAL",
            kalshi_fill,
            poly_fill,
        )
        return ExecutorState.DONE_EQUAL
