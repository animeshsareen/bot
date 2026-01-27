"""Deterministic flatten safety valve for cross-venue arbitrage.

Force exit when equality cannot be achieved quickly or near expiry.
Flattens both venues to qty 0 using aggressive taker orders.

Key concepts:
- FLAT_FORCED: Outcome when flatten completes successfully
- 5s timeout: Must complete within 5 seconds
- Idempotent: Can be called multiple times safely
- Cancellable: All operations can be cancelled
"""

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable
from enum import Enum
import asyncio
import time
import logging

from .venue_execution import OrderRequest, OrderResult, OrderSide, OrderType
from .position import PositionState, Outcome


logger = logging.getLogger(__name__)


# Default config
DEFAULT_FLATTEN_TIMEOUT_SEC = 5.0
DEFAULT_FLATTEN_PRICE_BUFFER = 0.05  # 5 cents buffer for aggressive fill


class FlattenStatus(Enum):
    """Status of flatten operation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class FlattenOrder:
    """Record of a flatten order.

    Attributes:
        venue: Venue identifier
        quantity: Quantity to flatten
        price: Price used
        filled_qty: Quantity filled
        order_id: Exchange order ID
        latency_ms: Order latency
    """

    venue: str
    quantity: int
    price: float
    filled_qty: int
    order_id: Optional[str]
    latency_ms: float


@dataclass(frozen=True)
class FlattenResult:
    """Result of flatten operation.

    Attributes:
        status: Final status
        outcome: Position outcome (FLAT_FORCED if successful)
        kalshi_order: Kalshi flatten order (if any)
        poly_order: Poly flatten order (if any)
        total_ms: Total time spent
        final_kalshi_qty: Final Kalshi position
        final_poly_qty: Final Poly position
    """

    status: FlattenStatus
    outcome: Optional[Outcome]
    kalshi_order: Optional[FlattenOrder]
    poly_order: Optional[FlattenOrder]
    total_ms: float
    final_kalshi_qty: int
    final_poly_qty: int

    @property
    def is_flat(self) -> bool:
        """Whether position is completely flat."""
        return self.final_kalshi_qty == 0 and self.final_poly_qty == 0

    def log_details(self) -> None:
        """Log flatten details at INFO level."""
        logger.info(
            "FLATTEN: status=%s outcome=%s kalshi_qty=%d poly_qty=%d "
            "total_ms=%.1f is_flat=%s",
            self.status.value,
            self.outcome.name if self.outcome else "N/A",
            self.final_kalshi_qty,
            self.final_poly_qty,
            self.total_ms,
            self.is_flat,
        )


# Type for order submission
OrderSubmitter = Callable[[OrderRequest], Awaitable[OrderResult]]
OrderCanceller = Callable[[str], Awaitable[bool]]


class Flattener:
    """Deterministic flatten safety valve.

    Flattens both venues to zero using aggressive taker orders.
    Must complete within timeout, all operations cancellable.
    """

    def __init__(
        self,
        timeout_sec: float = DEFAULT_FLATTEN_TIMEOUT_SEC,
        price_buffer: float = DEFAULT_FLATTEN_PRICE_BUFFER,
    ):
        """Initialize flattener.

        Args:
            timeout_sec: Maximum time to complete flatten
            price_buffer: Extra price buffer for aggressive fills
        """
        self.timeout_sec = timeout_sec
        self.price_buffer = price_buffer
        self._cancel_requested = False
        self._outstanding_orders: list[str] = []
        self._flatten_count = 0  # For idempotency tracking

    def request_cancel(self) -> None:
        """Request cancellation of flatten operation."""
        self._cancel_requested = True

    def reset(self) -> None:
        """Reset flattener state for new operation."""
        self._cancel_requested = False
        self._outstanding_orders = []

    async def flatten(
        self,
        kalshi_qty: int,
        poly_qty: int,
        kalshi_bid: float,
        poly_bid: float,
        submit_order: OrderSubmitter,
        cancel_order: Optional[OrderCanceller] = None,
    ) -> FlattenResult:
        """Flatten both venues to zero.

        Args:
            kalshi_qty: Current Kalshi position to flatten
            poly_qty: Current Poly position to flatten
            kalshi_bid: Current Kalshi best bid (sell price)
            poly_bid: Current Poly best bid (sell price)
            submit_order: Callback to submit orders
            cancel_order: Callback to cancel orders

        Returns:
            FlattenResult with outcome
        """
        self._flatten_count += 1
        start_ms = time.time() * 1000
        self.reset()

        # Nothing to flatten
        if kalshi_qty == 0 and poly_qty == 0:
            return FlattenResult(
                status=FlattenStatus.COMPLETED,
                outcome=Outcome.FLAT_FORCED,
                kalshi_order=None,
                poly_order=None,
                total_ms=0.0,
                final_kalshi_qty=0,
                final_poly_qty=0,
            )

        kalshi_order_record = None
        poly_order_record = None
        final_kalshi = kalshi_qty
        final_poly = poly_qty
        
        timeout_ms = self.timeout_sec * 1000

        try:
            # Retry loop for flattening
            while True:
                # Check for completion
                if final_kalshi == 0 and final_poly == 0:
                    status = FlattenStatus.COMPLETED
                    outcome = Outcome.FLAT_FORCED
                    break
                
                # Check for timeout or cancel
                remaining_ms = timeout_ms - (time.time() * 1000 - start_ms)
                if remaining_ms <= 0 or self._cancel_requested:
                    status = FlattenStatus.TIMEOUT if remaining_ms <= 0 else FlattenStatus.CANCELLED
                    outcome = None
                    break
                
                # Submit orders for remaining quantity
                tasks = []
                
                if final_kalshi > 0:
                    kalshi_price = max(0.01, kalshi_bid - self.price_buffer)
                    kalshi_order = OrderRequest(
                        venue="kalshi",
                        market_id="flatten_kalshi",
                        side=OrderSide.SELL,
                        price=kalshi_price,
                        quantity=final_kalshi,
                        order_type=OrderType.IOC,
                    )
                    tasks.append(("kalshi", kalshi_order, submit_order(kalshi_order)))

                if final_poly > 0:
                    poly_price = max(0.01, poly_bid - self.price_buffer)
                    poly_order = OrderRequest(
                        venue="poly",
                        market_id="flatten_poly",
                        side=OrderSide.SELL,
                        price=poly_price,
                        quantity=final_poly,
                        order_type=OrderType.FAK,
                    )
                    tasks.append(("poly", poly_order, submit_order(poly_order)))

                # Execute concurrent orders
                for venue, order, coro in tasks:
                    if self._cancel_requested:
                        break

                    step_start = time.time() * 1000
                    try:
                        # Use small timeout for individual steps to allow overall timeout check
                        result = await asyncio.wait_for(
                            coro,
                            timeout=remaining_ms / 1000
                        )
                    except asyncio.TimeoutError:
                        continue 

                    step_latency = time.time() * 1000 - step_start

                    if result.order_id:
                        self._outstanding_orders.append(result.order_id)

                    order_record = FlattenOrder(
                        venue=venue,
                        quantity=order.quantity,
                        price=order.price,
                        filled_qty=result.filled_qty,
                        order_id=result.order_id,
                        latency_ms=step_latency,
                    )

                    if venue == "kalshi":
                        kalshi_order_record = order_record
                        final_kalshi -= result.filled_qty
                    else:
                        poly_order_record = order_record
                        final_poly -= result.filled_qty

                # If we still have remaining qty, wait a bit before retry to avoid spamming
                if final_kalshi > 0 or final_poly > 0:
                    await asyncio.sleep(0.1) # 100ms retry delay

            return FlattenResult(
                status=status,
                outcome=outcome,
                kalshi_order=kalshi_order_record,
                poly_order=poly_order_record,
                total_ms=time.time() * 1000 - start_ms,
                final_kalshi_qty=final_kalshi,
                final_poly_qty=final_poly,
            )

        except Exception as e:
            logger.error("Flatten failed: %s", e)
            return FlattenResult(
                status=FlattenStatus.FAILED,
                outcome=None,
                kalshi_order=kalshi_order_record,
                poly_order=poly_order_record,
                total_ms=time.time() * 1000 - start_ms,
                final_kalshi_qty=final_kalshi,
                final_poly_qty=final_poly,
            )

    def _make_result(
        self,
        status: FlattenStatus,
        outcome: Optional[Outcome],
        kalshi_order: Optional[FlattenOrder],
        poly_order: Optional[FlattenOrder],
        start_ms: float,
        final_kalshi: int,
        final_poly: int,
    ) -> FlattenResult:
        """Helper to create result."""
        return FlattenResult(
            status=status,
            outcome=outcome,
            kalshi_order=kalshi_order,
            poly_order=poly_order,
            total_ms=time.time() * 1000 - start_ms,
            final_kalshi_qty=final_kalshi,
            final_poly_qty=final_poly,
        )


def create_flatten_order(
    venue: str,
    quantity: int,
    bid_price: float,
    price_buffer: float = DEFAULT_FLATTEN_PRICE_BUFFER,
) -> OrderRequest:
    """Create a flatten order for a venue.

    Args:
        venue: Venue identifier
        quantity: Position to flatten
        bid_price: Current best bid
        price_buffer: Buffer below bid for aggressive fill

    Returns:
        OrderRequest for flatten
    """
    price = max(0.01, bid_price - price_buffer)
    order_type = OrderType.IOC if venue == "kalshi" else OrderType.FAK

    return OrderRequest(
        venue=venue,
        market_id=f"flatten_{venue}",
        side=OrderSide.SELL,
        price=price,
        quantity=quantity,
        order_type=order_type,
    )
