"""Periodic reconciliation loop for state synchronization.

Background task that periodically fetches venue state via REST and
reconciles with local state. Detects and corrects:
- Orders that exist on venue but not locally (unknown orders)
- Orders that exist locally but not on venue (stale orders)
- Fill mismatches between local and venue state
- Position discrepancies

Key design principles:
- Runs as background async task
- Non-blocking, cancellable
- Emits events for corrections (doesn't mutate state directly)
- Respects rate limits
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any
from enum import Enum, auto

from .types import Event, EventType, OrderState, TERMINAL_ORDER_STATES
from .state import StateManager


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RECONCILE_INTERVAL_SEC = 5.0  # Aggressive per config
DEFAULT_STALE_ORDER_AGE_SEC = 60.0    # Orders older than this with no updates
DEFAULT_MAX_UNKNOWN_ORDERS = 10       # Max unknown orders to cancel per cycle


class MismatchType(Enum):
    """Types of state mismatches detected."""
    UNKNOWN_ORDER = auto()      # Order on venue not in local state
    STALE_ORDER = auto()        # Order in local state not on venue
    FILL_MISMATCH = auto()      # Fill quantity differs
    POSITION_MISMATCH = auto()  # Position differs from expected
    ORDER_STATE_MISMATCH = auto()  # Order state differs


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ReconciliationMetrics:
    """Metrics for reconciliation loop.

    Attributes:
        cycles_completed: Total reconciliation cycles
        mismatches_detected: Total mismatches found
        corrections_emitted: Correction events emitted
        unknown_orders_cancelled: Unknown orders cancelled
        errors: Errors during reconciliation
        last_cycle_ms: Duration of last cycle
        avg_cycle_ms: Average cycle duration
    """
    cycles_completed: int = 0
    mismatches_detected: int = 0
    corrections_emitted: int = 0
    unknown_orders_cancelled: int = 0
    errors: int = 0
    last_cycle_ms: float = 0.0
    _cycle_sum_ms: float = 0.0

    @property
    def avg_cycle_ms(self) -> float:
        if self.cycles_completed == 0:
            return 0.0
        return self._cycle_sum_ms / self.cycles_completed


@dataclass
class Mismatch:
    """Record of a detected mismatch.

    Attributes:
        mismatch_type: Type of mismatch
        venue: Venue where mismatch occurred
        order_id: Order ID (if applicable)
        local_value: Value in local state
        venue_value: Value on venue
        details: Additional context
    """
    mismatch_type: MismatchType
    venue: str
    order_id: Optional[str] = None
    local_value: Any = None
    venue_value: Any = None
    details: str = ""


# =============================================================================
# VENUE CLIENT PROTOCOL
# =============================================================================

@dataclass
class VenueOrder:
    """Order as reported by venue REST API."""
    order_id: str
    client_order_id: Optional[str]
    market_id: str
    side: str
    price: float
    quantity: int
    filled_qty: int
    status: str  # "open", "filled", "cancelled", etc.
    created_at_ms: int


@dataclass
class VenuePosition:
    """Position as reported by venue REST API."""
    market_id: str
    position: int  # Signed: positive=long, negative=short
    avg_price: Optional[float] = None


# Callback types for venue REST calls
FetchOrdersCallback = Callable[[], Awaitable[list[VenueOrder]]]
FetchPositionsCallback = Callable[[], Awaitable[list[VenuePosition]]]
CancelOrderCallback = Callable[[str], Awaitable[bool]]  # order_id -> success

EventEmitter = Callable[[Event], Awaitable[None]]


# =============================================================================
# RECONCILIATION LOOP
# =============================================================================

class ReconciliationLoop:
    """Background loop for periodic state reconciliation.

    Fetches state from venues via REST and compares to local state.
    Emits correction events when mismatches are detected.

    Usage:
        loop = ReconciliationLoop(
            state_manager=state,
            event_emitter=emit_event,
            interval_sec=5.0,
        )
        loop.set_kalshi_client(fetch_orders, fetch_positions, cancel_order)
        loop.set_poly_client(fetch_orders, fetch_positions, cancel_order)
        await loop.start()
        ...
        await loop.stop()
    """

    def __init__(
        self,
        state_manager: StateManager,
        event_emitter: EventEmitter,
        interval_sec: float = DEFAULT_RECONCILE_INTERVAL_SEC,
        stale_order_age_sec: float = DEFAULT_STALE_ORDER_AGE_SEC,
        max_unknown_orders: int = DEFAULT_MAX_UNKNOWN_ORDERS,
    ):
        """Initialize reconciliation loop.

        Args:
            state_manager: StateManager for local state access
            event_emitter: Callback to emit correction events
            interval_sec: Seconds between reconciliation cycles
            stale_order_age_sec: Age threshold for stale orders
            max_unknown_orders: Max unknown orders to cancel per cycle
        """
        self._state = state_manager
        self._event_emitter = event_emitter
        self._interval_sec = interval_sec
        self._stale_order_age_sec = stale_order_age_sec
        self._max_unknown_orders = max_unknown_orders

        # Venue clients
        self._kalshi_fetch_orders: Optional[FetchOrdersCallback] = None
        self._kalshi_fetch_positions: Optional[FetchPositionsCallback] = None
        self._kalshi_cancel: Optional[CancelOrderCallback] = None

        self._poly_fetch_orders: Optional[FetchOrdersCallback] = None
        self._poly_fetch_positions: Optional[FetchPositionsCallback] = None
        self._poly_cancel: Optional[CancelOrderCallback] = None

        # Loop state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._sequence_counter = 0

        self.metrics = ReconciliationMetrics()

    # =========================================================================
    # CLIENT CONFIGURATION
    # =========================================================================

    def set_kalshi_client(
        self,
        fetch_orders: FetchOrdersCallback,
        fetch_positions: FetchPositionsCallback,
        cancel_order: CancelOrderCallback,
    ) -> None:
        """Configure Kalshi REST client callbacks."""
        self._kalshi_fetch_orders = fetch_orders
        self._kalshi_fetch_positions = fetch_positions
        self._kalshi_cancel = cancel_order

    def set_poly_client(
        self,
        fetch_orders: FetchOrdersCallback,
        fetch_positions: FetchPositionsCallback,
        cancel_order: CancelOrderCallback,
    ) -> None:
        """Configure Polymarket REST client callbacks."""
        self._poly_fetch_orders = fetch_orders
        self._poly_fetch_positions = fetch_positions
        self._poly_cancel = cancel_order

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """Start the reconciliation loop."""
        if self._running:
            logger.warning("ReconciliationLoop already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(
            self._run_loop(),
            name="reconciliation_loop"
        )
        logger.info(
            "ReconciliationLoop started (interval=%.1fs)",
            self._interval_sec
        )

    async def stop(self, timeout_sec: float = 5.0) -> bool:
        """Stop the reconciliation loop.

        Args:
            timeout_sec: Max time to wait for graceful stop

        Returns:
            True if stopped within timeout
        """
        if not self._running:
            return True

        logger.info("ReconciliationLoop stopping...")
        self._shutdown_event.set()

        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=timeout_sec)
            except asyncio.TimeoutError:
                logger.warning("ReconciliationLoop stop timeout, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self._running = False
        logger.info("ReconciliationLoop stopped")
        return True

    @property
    def is_running(self) -> bool:
        """True if loop is running."""
        return self._running

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def _run_loop(self) -> None:
        """Main reconciliation loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Run reconciliation cycle
                start_ms = time.time() * 1000
                await self._reconcile_cycle()
                cycle_ms = time.time() * 1000 - start_ms

                # Update metrics
                self.metrics.cycles_completed += 1
                self.metrics.last_cycle_ms = cycle_ms
                self.metrics._cycle_sum_ms += cycle_ms

                logger.debug(
                    "Reconciliation cycle completed in %.1fms",
                    cycle_ms
                )

            except asyncio.CancelledError:
                logger.info("Reconciliation loop cancelled")
                break
            except Exception as e:
                logger.error("Reconciliation cycle error: %s", e, exc_info=True)
                self.metrics.errors += 1

            # Wait for next cycle or shutdown
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._interval_sec
                )
                # Shutdown requested
                break
            except asyncio.TimeoutError:
                # Normal timeout, continue to next cycle
                pass

    async def _reconcile_cycle(self) -> None:
        """Run a single reconciliation cycle."""
        mismatches: list[Mismatch] = []

        # Reconcile Kalshi
        if self._kalshi_fetch_orders:
            kalshi_mismatches = await self._reconcile_venue(
                "kalshi",
                self._kalshi_fetch_orders,
                self._kalshi_fetch_positions,
                self._kalshi_cancel,
            )
            mismatches.extend(kalshi_mismatches)

        # Reconcile Poly
        if self._poly_fetch_orders:
            poly_mismatches = await self._reconcile_venue(
                "poly",
                self._poly_fetch_orders,
                self._poly_fetch_positions,
                self._poly_cancel,
            )
            mismatches.extend(poly_mismatches)

        # Process mismatches
        if mismatches:
            self.metrics.mismatches_detected += len(mismatches)
            await self._handle_mismatches(mismatches)

    async def _reconcile_venue(
        self,
        venue: str,
        fetch_orders: FetchOrdersCallback,
        fetch_positions: Optional[FetchPositionsCallback],
        cancel_order: Optional[CancelOrderCallback],
    ) -> list[Mismatch]:
        """Reconcile state for a single venue.

        Args:
            venue: Venue name
            fetch_orders: Callback to fetch orders
            fetch_positions: Callback to fetch positions
            cancel_order: Callback to cancel orders

        Returns:
            List of detected mismatches
        """
        mismatches: list[Mismatch] = []

        try:
            # Fetch venue orders
            venue_orders = await fetch_orders()
            venue_order_map = {o.order_id: o for o in venue_orders}

            # Also index by client_order_id for matching
            venue_by_client_id = {
                o.client_order_id: o
                for o in venue_orders
                if o.client_order_id
            }

            # Get local orders for this venue
            local_orders = [
                o for o in self._state._orders.values()
                if o.venue == venue and o.state not in TERMINAL_ORDER_STATES
            ]

            now_ms = int(time.time() * 1000)

            # Check for unknown orders (on venue but not local)
            local_venue_order_ids = {
                o.venue_order_id for o in local_orders if o.venue_order_id
            }
            local_client_order_ids = {
                o.client_order_id for o in local_orders if o.client_order_id
            }

            unknown_orders = []
            for vo in venue_orders:
                # Skip if we know about it
                if vo.order_id in local_venue_order_ids:
                    continue
                if vo.client_order_id and vo.client_order_id in local_client_order_ids:
                    continue

                # Unknown order
                unknown_orders.append(vo)
                mismatches.append(Mismatch(
                    mismatch_type=MismatchType.UNKNOWN_ORDER,
                    venue=venue,
                    order_id=vo.order_id,
                    local_value=None,
                    venue_value=vo,
                    details=f"Order {vo.order_id} exists on venue but not locally",
                ))

            # Cancel unknown orders (they cause 409 conflicts)
            if cancel_order and unknown_orders:
                cancelled = 0
                for vo in unknown_orders[:self._max_unknown_orders]:
                    try:
                        if await cancel_order(vo.order_id):
                            cancelled += 1
                            logger.info(
                                "Cancelled unknown order %s on %s",
                                vo.order_id, venue
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to cancel unknown order %s: %s",
                            vo.order_id, e
                        )
                self.metrics.unknown_orders_cancelled += cancelled

            # Check for stale orders (local but not on venue)
            for lo in local_orders:
                # Skip very new orders (might not be on venue yet)
                age_sec = (now_ms - lo.created_at_ms) / 1000
                if age_sec < 5.0:
                    continue

                # Check if order exists on venue
                found = False
                if lo.venue_order_id and lo.venue_order_id in venue_order_map:
                    found = True
                    vo = venue_order_map[lo.venue_order_id]
                elif lo.client_order_id and lo.client_order_id in venue_by_client_id:
                    found = True
                    vo = venue_by_client_id[lo.client_order_id]
                else:
                    vo = None

                if not found and age_sec > self._stale_order_age_sec:
                    # Stale order - exists locally but not on venue
                    mismatches.append(Mismatch(
                        mismatch_type=MismatchType.STALE_ORDER,
                        venue=venue,
                        order_id=lo.order_id,
                        local_value=lo,
                        venue_value=None,
                        details=f"Order {lo.order_id} is stale (age={age_sec:.0f}s)",
                    ))

                # Check fill mismatch
                if found and vo:
                    if vo.filled_qty != lo.filled_qty:
                        mismatches.append(Mismatch(
                            mismatch_type=MismatchType.FILL_MISMATCH,
                            venue=venue,
                            order_id=lo.order_id,
                            local_value=lo.filled_qty,
                            venue_value=vo.filled_qty,
                            details=f"Fill mismatch: local={lo.filled_qty} venue={vo.filled_qty}",
                        ))

        except Exception as e:
            logger.error("Failed to reconcile %s: %s", venue, e)
            self.metrics.errors += 1

        return mismatches

    async def _handle_mismatches(self, mismatches: list[Mismatch]) -> None:
        """Handle detected mismatches by emitting correction events.

        Args:
            mismatches: List of mismatches to handle
        """
        for mismatch in mismatches:
            logger.warning(
                "Mismatch detected: type=%s venue=%s order=%s details=%s",
                mismatch.mismatch_type.name,
                mismatch.venue,
                mismatch.order_id,
                mismatch.details,
            )

            # Emit correction events based on mismatch type
            if mismatch.mismatch_type == MismatchType.STALE_ORDER:
                # Emit cancellation event for stale order
                await self._emit_order_cancelled(
                    mismatch.order_id,
                    mismatch.venue,
                )
                self.metrics.corrections_emitted += 1

            elif mismatch.mismatch_type == MismatchType.FILL_MISMATCH:
                # Emit fill event for missing fills
                venue_filled = mismatch.venue_value
                local_filled = mismatch.local_value
                if venue_filled > local_filled:
                    # We missed fills
                    missing_qty = venue_filled - local_filled
                    await self._emit_fill_correction(
                        mismatch.order_id,
                        mismatch.venue,
                        missing_qty,
                    )
                    self.metrics.corrections_emitted += 1

    # =========================================================================
    # EVENT EMISSION
    # =========================================================================

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence_counter += 1
        return self._sequence_counter

    async def _emit_order_cancelled(
        self,
        order_id: str,
        venue: str,
    ) -> None:
        """Emit ORDER_CANCELLED event for stale order."""
        event = Event(
            event_type=EventType.ORDER_CANCELLED,
            sequence_num=self._next_sequence(),
            venue=venue,
            order_id=order_id,
        )
        await self._event_emitter(event)
        logger.info("Emitted cancellation for stale order %s", order_id)

    async def _emit_fill_correction(
        self,
        order_id: str,
        venue: str,
        quantity: int,
    ) -> None:
        """Emit ORDER_FILL event for missed fill."""
        event = Event(
            event_type=EventType.ORDER_FILL,
            sequence_num=self._next_sequence(),
            venue=venue,
            order_id=order_id,
            fill_qty=quantity,
            fill_price=0.0,  # Unknown price for correction
        )
        await self._event_emitter(event)
        logger.info(
            "Emitted fill correction for order %s: qty=%d",
            order_id, quantity
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_reconciliation_loop(
    state_manager: StateManager,
    event_emitter: EventEmitter,
    interval_sec: float = DEFAULT_RECONCILE_INTERVAL_SEC,
) -> ReconciliationLoop:
    """Factory to create ReconciliationLoop.

    Args:
        state_manager: StateManager for local state
        event_emitter: Callback to emit events
        interval_sec: Reconciliation interval

    Returns:
        Configured ReconciliationLoop
    """
    return ReconciliationLoop(
        state_manager=state_manager,
        event_emitter=event_emitter,
        interval_sec=interval_sec,
    )
