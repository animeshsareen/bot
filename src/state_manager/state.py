"""Single-writer state manager with atomic updates.

Central state store consumed by a single event-processing coroutine.
All mutations go through this module to maintain invariants.

Key design principles:
- Single-writer: only one coroutine mutates state
- Atomic updates: transactions succeed fully or rollback
- Immutable snapshots: readers get frozen copies
- Invariant enforcement: validates all transitions
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Callable
from contextlib import contextmanager

from .types import (
    Order, OrderState, Fill, Attempt, AttemptState, Inventory, Event, EventType, Snapshot,
    is_valid_order_transition, is_valid_attempt_transition,
    TERMINAL_ORDER_STATES, TERMINAL_ATTEMPT_STATES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class StateError(Exception):
    """Base exception for state errors."""
    pass


class InvalidTransitionError(StateError):
    """Raised when an invalid state transition is attempted."""
    pass


class InvariantViolationError(StateError):
    """Raised when a state invariant would be violated."""
    pass


class OutOfOrderEventError(StateError):
    """Raised when an event arrives out of sequence."""
    pass


class RollbackError(StateError):
    """Raised when rollback fails."""
    pass


class InventoryBandError(StateError):
    """Raised when inventory is outside acceptable band for new attempts."""
    pass


# =============================================================================
# TRANSACTION CONTEXT
# =============================================================================

@dataclass
class Transaction:
    """Captures state before mutation for rollback.

    Used internally by StateManager to enable atomic updates.
    """
    orders_snapshot: dict[str, Order] = field(default_factory=dict)
    attempts_snapshot: dict[str, Attempt] = field(default_factory=dict)
    inventory_snapshot: Optional[Inventory] = None
    fills_count: int = 0
    committed: bool = False

    def capture(self, state: "StateManager") -> None:
        """Capture current state for potential rollback."""
        self.orders_snapshot = dict(state._orders)
        self.attempts_snapshot = dict(state._attempts)
        self.inventory_snapshot = state._inventory
        self.fills_count = len(state._fills)

    def rollback(self, state: "StateManager") -> None:
        """Restore state from snapshot."""
        if self.committed:
            raise RollbackError("Cannot rollback committed transaction")
        state._orders = self.orders_snapshot
        state._attempts = self.attempts_snapshot
        state._inventory = self.inventory_snapshot
        # Truncate fills to pre-transaction count
        state._fills = state._fills[:self.fills_count]
        logger.warning("Transaction rolled back")

    def commit(self) -> None:
        """Mark transaction as committed (no rollback possible)."""
        self.committed = True


# =============================================================================
# STATE MANAGER
# =============================================================================

class StateManager:
    """Single-writer state manager with atomic updates.

    Maintains:
    - Orders: all orders by order_id
    - Attempts: all attempts by attempt_id
    - Inventory: current positions and balances
    - Fills: ordered list of all fills
    - Sequence tracking: for out-of-order detection

    Thread Safety:
    - Only the event consumer coroutine should call mutating methods
    - Snapshot methods return immutable copies safe for concurrent reads
    """

    def __init__(self, initial_inventory: Optional[Inventory] = None):
        """Initialize state manager.

        Args:
            initial_inventory: Starting inventory (defaults to zero)
        """
        self._orders: dict[str, Order] = {}
        self._attempts: dict[str, Attempt] = {}
        self._fills: list[Fill] = []
        self._inventory: Inventory = initial_inventory or Inventory()

        # Sequence tracking per venue for out-of-order detection
        self._last_sequence: dict[str, int] = {}  # venue -> last seq

        # Active attempt tracking (at most one active)
        self._active_attempt_id: Optional[str] = None

        # Transaction support
        self._current_txn: Optional[Transaction] = None

        # Metrics
        self._events_processed: int = 0
        self._events_rejected: int = 0

    # =========================================================================
    # TRANSACTION MANAGEMENT
    # =========================================================================

    @contextmanager
    def transaction(self):
        """Context manager for atomic updates.

        Usage:
            with state.transaction():
                state.update_order(...)
                state.update_inventory(...)
            # Auto-commits on success, auto-rollback on exception
        """
        if self._current_txn is not None:
            raise StateError("Nested transactions not supported")

        txn = Transaction()
        txn.capture(self)
        self._current_txn = txn

        try:
            yield txn
            txn.commit()
        except Exception as e:
            logger.error("Transaction failed, rolling back: %s", e)
            txn.rollback(self)
            raise
        finally:
            self._current_txn = None

    # =========================================================================
    # ORDER OPERATIONS
    # =========================================================================

    def create_order(
        self,
        venue: str,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        quantity: int,
        attempt_id: Optional[str] = None,
        leg: Optional[str] = None,
    ) -> Order:
        """Create a new order in PENDING state.

        Args:
            venue: "kalshi" or "poly"
            market_id: Market/token ID
            side: "buy" or "sell" (trade direction: BUY/SELL)
            outcome: "YES" or "NO" (which outcome token)
            price: Limit price
            quantity: Order quantity
            attempt_id: Parent attempt ID (optional)
            leg: Leg identifier for idempotency key (e.g., "YES" or "NO").
                Used with attempt_id to generate client_order_id.

        Returns:
            New Order in PENDING state

        Raises:
            InvariantViolationError: If quantity <= 0
        """
        if quantity <= 0:
            raise InvariantViolationError(f"Order quantity must be positive: {quantity}")

        order_id = f"{venue}_{uuid.uuid4().hex[:8]}"

        # Generate client_order_id for venue-side idempotency
        # Format: A-{attempt_suffix}-{VENUE}-{LEG}
        # e.g., A-abc12345-KALSHI-YES
        if attempt_id and leg:
            # Extract suffix from attempt_id (e.g., "attempt_abc12345" -> "abc12345")
            attempt_suffix = attempt_id.replace("attempt_", "")
            client_order_id = f"A-{attempt_suffix}-{venue.upper()}-{leg.upper()}"
        else:
            # Fallback: use order_id as client_order_id
            client_order_id = order_id

        order = Order(
            order_id=order_id,
            venue=venue,
            market_id=market_id,
            side=side,
            outcome=outcome,
            price=price,
            quantity=quantity,
            state=OrderState.PENDING,
            client_order_id=client_order_id,
            attempt_id=attempt_id,
        )

        self._orders[order_id] = order
        logger.debug("Created order %s (client_id=%s): %s %s %d @ %.4f on %s",
                    order_id, client_order_id, side, outcome, quantity, price, venue)
        return order

    def update_order_state(
        self,
        order_id: str,
        new_state: OrderState,
        venue_order_id: Optional[str] = None,
        filled_qty: Optional[int] = None,
        avg_fill_price: Optional[float] = None,
        error: Optional[str] = None,
    ) -> Order:
        """Update order to new state with validation.

        Args:
            order_id: Order to update
            new_state: Target state
            venue_order_id: Venue-assigned ID (on ACK)
            filled_qty: Updated fill quantity
            avg_fill_price: Average fill price
            error: Error message (on REJECTED)

        Returns:
            Updated Order

        Raises:
            InvalidTransitionError: If transition is invalid
            KeyError: If order not found
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order not found: {order_id}")

        if not is_valid_order_transition(order.state, new_state):
            raise InvalidTransitionError(
                f"Invalid order transition: {order.state} → {new_state} for {order_id}"
            )

        updates = {}
        if venue_order_id is not None:
            updates['venue_order_id'] = venue_order_id
        if filled_qty is not None:
            updates['filled_qty'] = filled_qty
        if avg_fill_price is not None:
            updates['avg_fill_price'] = avg_fill_price
        if error is not None:
            updates['error'] = error

        updated = order.with_state(new_state, **updates)
        self._orders[order_id] = updated

        logger.debug(
            "Order %s: %s → %s (filled=%d/%d)",
            order_id, order.state.value, new_state.value,
            updated.filled_qty, updated.quantity
        )
        return updated

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_orders_for_attempt(self, attempt_id: str) -> list[Order]:
        """Get all orders for an attempt."""
        return [o for o in self._orders.values() if o.attempt_id == attempt_id]

    # =========================================================================
    # FILL OPERATIONS
    # =========================================================================

    def record_fill(
        self,
        order_id: str,
        quantity: int,
        price: float,
        timestamp_ms: Optional[int] = None,
    ) -> Fill:
        """Record a fill and update order + inventory atomically.

        Args:
            order_id: Order that was filled
            quantity: Fill quantity
            price: Fill price
            timestamp_ms: Venue timestamp

        Returns:
            New Fill record

        Raises:
            KeyError: If order not found
            InvariantViolationError: If fill exceeds remaining qty
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order not found: {order_id}")

        if quantity <= 0:
            raise InvariantViolationError(f"Fill quantity must be positive: {quantity}")

        new_filled = order.filled_qty + quantity
        if new_filled > order.quantity:
            raise InvariantViolationError(
                f"Fill exceeds order qty: {new_filled} > {order.quantity} for {order_id}"
            )

        # Create fill record
        fill_id = f"fill_{uuid.uuid4().hex[:8]}"
        fill = Fill(
            fill_id=fill_id,
            order_id=order_id,
            venue=order.venue,
            quantity=quantity,
            price=price,
            timestamp_ms=timestamp_ms or int(time.time() * 1000),
        )
        self._fills.append(fill)

        # Update order state
        new_state = OrderState.FILLED if new_filled == order.quantity else OrderState.PARTIAL

        # Calculate new average fill price
        if order.avg_fill_price is not None:
            total_cost = (order.avg_fill_price * order.filled_qty) + (price * quantity)
            avg_price = total_cost / new_filled
        else:
            avg_price = price

        self.update_order_state(
            order_id,
            new_state,
            filled_qty=new_filled,
            avg_fill_price=avg_price,
        )

        # Update inventory
        self._inventory = self._inventory.with_fill(order.venue, quantity, price, order.side)

        # Update attempt fill counts if order belongs to attempt
        if order.attempt_id:
            self._update_attempt_fills(order.attempt_id, order.venue, quantity)

        logger.info(
            "Fill recorded: %s %d @ %.4f on %s (order %s now %d/%d)",
            order.side, quantity, price, order.venue, order_id,
            new_filled, order.quantity
        )
        return fill

    def _update_attempt_fills(self, attempt_id: str, venue: str, qty: int) -> None:
        """Update attempt fill counts after a fill."""
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            return

        if venue == "kalshi":
            updated = Attempt(
                attempt_id=attempt.attempt_id,
                state=attempt.state,
                strategy_path=attempt.strategy_path,
                target_qty=attempt.target_qty,
                kalshi_order_id=attempt.kalshi_order_id,
                poly_order_id=attempt.poly_order_id,
                kalshi_filled=attempt.kalshi_filled + qty,
                poly_filled=attempt.poly_filled,
                created_at_ms=attempt.created_at_ms,
                updated_at_ms=int(time.time() * 1000),
                completed_at_ms=attempt.completed_at_ms,
                error=attempt.error,
            )
        else:
            updated = Attempt(
                attempt_id=attempt.attempt_id,
                state=attempt.state,
                strategy_path=attempt.strategy_path,
                target_qty=attempt.target_qty,
                kalshi_order_id=attempt.kalshi_order_id,
                poly_order_id=attempt.poly_order_id,
                kalshi_filled=attempt.kalshi_filled,
                poly_filled=attempt.poly_filled + qty,
                created_at_ms=attempt.created_at_ms,
                updated_at_ms=int(time.time() * 1000),
                completed_at_ms=attempt.completed_at_ms,
                error=attempt.error,
            )

        self._attempts[attempt_id] = updated

    # =========================================================================
    # ATTEMPT OPERATIONS
    # =========================================================================

    # Default inventory band: blocks new attempts if |net_position| > this
    DEFAULT_INVENTORY_BAND = 5

    def can_start_attempt(self, max_delta: int = DEFAULT_INVENTORY_BAND) -> bool:
        """Check if inventory is within acceptable band to start a new attempt.

        Prevents arb loop from starting while hedger is still working to
        flatten an imbalanced position. This avoids compounding risk.

        Args:
            max_delta: Maximum allowed |net_position| to start new attempt.
                       Default is 5 shares (tight band).

        Returns:
            True if |net_position| <= max_delta, False otherwise
        """
        return abs(self._inventory.net_position) <= max_delta

    def create_attempt(
        self,
        strategy_path: int,
        target_qty: int,
        max_inventory_delta: int = DEFAULT_INVENTORY_BAND,
    ) -> Attempt:
        """Create a new arb attempt.

        Args:
            strategy_path: Arb strategy (1 or 2)
            target_qty: Target quantity for both legs
            max_inventory_delta: Maximum allowed |net_position| to start.
                                 Default is 5 shares.

        Returns:
            New Attempt in IDLE state

        Raises:
            InvariantViolationError: If another attempt is active
            InventoryBandError: If inventory |net_position| > max_inventory_delta
        """
        # Check inventory band first - prevents compounding imbalance
        if not self.can_start_attempt(max_inventory_delta):
            net = self._inventory.net_position
            raise InventoryBandError(
                f"Cannot start attempt: net_position={net} exceeds band ±{max_inventory_delta}"
            )

        if self._active_attempt_id is not None:
            active = self._attempts.get(self._active_attempt_id)
            if active and not active.is_terminal:
                raise InvariantViolationError(
                    f"Cannot create attempt: {self._active_attempt_id} still active"
                )

        attempt_id = f"attempt_{uuid.uuid4().hex[:8]}"
        attempt = Attempt(
            attempt_id=attempt_id,
            state=AttemptState.IDLE,
            strategy_path=strategy_path,
            target_qty=target_qty,
        )

        self._attempts[attempt_id] = attempt
        self._active_attempt_id = attempt_id

        logger.info("Created attempt %s: path=%d qty=%d", attempt_id, strategy_path, target_qty)
        return attempt

    def update_attempt_state(
        self,
        attempt_id: str,
        new_state: AttemptState,
        kalshi_order_id: Optional[str] = None,
        poly_order_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Attempt:
        """Update attempt to new state with validation.

        Args:
            attempt_id: Attempt to update
            new_state: Target state
            kalshi_order_id: Kalshi order ID
            poly_order_id: Poly order ID
            error: Error message

        Returns:
            Updated Attempt

        Raises:
            InvalidTransitionError: If transition is invalid
            KeyError: If attempt not found
        """
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            raise KeyError(f"Attempt not found: {attempt_id}")

        if not is_valid_attempt_transition(attempt.state, new_state):
            raise InvalidTransitionError(
                f"Invalid attempt transition: {attempt.state} → {new_state} for {attempt_id}"
            )

        updates = {}
        if kalshi_order_id is not None:
            updates['kalshi_order_id'] = kalshi_order_id
        if poly_order_id is not None:
            updates['poly_order_id'] = poly_order_id
        if error is not None:
            updates['error'] = error

        updated = attempt.with_state(new_state, **updates)
        self._attempts[attempt_id] = updated

        # Clear active attempt if terminal
        if updated.is_terminal and self._active_attempt_id == attempt_id:
            self._active_attempt_id = None

        logger.info(
            "Attempt %s: %s → %s (K=%d P=%d)",
            attempt_id, attempt.state.value, new_state.value,
            updated.kalshi_filled, updated.poly_filled
        )
        return updated

    def get_attempt(self, attempt_id: str) -> Optional[Attempt]:
        """Get attempt by ID."""
        return self._attempts.get(attempt_id)

    def get_active_attempt(self) -> Optional[Attempt]:
        """Get currently active attempt (if any)."""
        if self._active_attempt_id is None:
            return None
        return self._attempts.get(self._active_attempt_id)

    # =========================================================================
    # INVENTORY OPERATIONS
    # =========================================================================

    def get_inventory(self) -> Inventory:
        """Get current inventory (immutable copy)."""
        return self._inventory

    def set_inventory(self, inventory: Inventory) -> None:
        """Set inventory directly (for initialization or reconciliation)."""
        self._inventory = inventory
        logger.info(
            "Inventory set: K_pos=%d P_pos=%d K_bal=%.2f P_bal=%.2f",
            inventory.kalshi_position, inventory.poly_position,
            inventory.kalshi_balance, inventory.poly_balance
        )

    # =========================================================================
    # SEQUENCE TRACKING
    # =========================================================================

    def check_sequence(self, venue: str, sequence_num: int) -> bool:
        """Check if event sequence is valid (not out of order).

        Args:
            venue: Source venue
            sequence_num: Event sequence number

        Returns:
            True if sequence is valid (>= last seen)
        """
        last = self._last_sequence.get(venue, -1)
        if sequence_num < last:
            return False
        self._last_sequence[venue] = sequence_num
        return True

    def reject_if_out_of_order(self, event: Event) -> None:
        """Raise exception if event is out of order.

        Args:
            event: Event to check

        Raises:
            OutOfOrderEventError: If event sequence is out of order
        """
        if event.venue is None:
            return  # Internal events don't have venue sequence

        if not self.check_sequence(event.venue, event.sequence_num):
            self._events_rejected += 1
            raise OutOfOrderEventError(
                f"Out of order event: venue={event.venue} seq={event.sequence_num} "
                f"last={self._last_sequence.get(event.venue)}"
            )

    # =========================================================================
    # SNAPSHOTS
    # =========================================================================

    def snapshot(
        self,
        poly_yes_ask: Optional[float] = None,
        poly_yes_bid: Optional[float] = None,
        poly_no_ask: Optional[float] = None,
        poly_no_bid: Optional[float] = None,
        kalshi_yes_ask: Optional[float] = None,
        kalshi_yes_bid: Optional[float] = None,
        book_seq_kalshi: int = 0,
        book_seq_poly_yes: int = 0,
        book_seq_poly_no: int = 0,
    ) -> Snapshot:
        """Create immutable snapshot of current state.

        Args:
            poly_yes_ask: Polymarket YES ask price
            poly_yes_bid: Polymarket YES bid price
            poly_no_ask: Polymarket NO ask price
            poly_no_bid: Polymarket NO bid price
            kalshi_yes_ask: Kalshi YES ask price
            kalshi_yes_bid: Kalshi YES bid price
            book_seq_kalshi: Kalshi order book sequence number
            book_seq_poly_yes: Polymarket YES book sequence number
            book_seq_poly_no: Polymarket NO book sequence number

        Returns:
            Immutable Snapshot for engine decisions with version info
        """
        return Snapshot(
            poly_yes_ask=poly_yes_ask,
            poly_yes_bid=poly_yes_bid,
            poly_no_ask=poly_no_ask,
            poly_no_bid=poly_no_bid,
            kalshi_yes_ask=kalshi_yes_ask,
            kalshi_yes_bid=kalshi_yes_bid,
            inventory=self._inventory,
            active_attempt=self.get_active_attempt(),
            snapshot_ms=int(time.time() * 1000),
            book_seq_kalshi=book_seq_kalshi,
            book_seq_poly_yes=book_seq_poly_yes,
            book_seq_poly_no=book_seq_poly_no,
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    @property
    def events_processed(self) -> int:
        """Total events successfully processed."""
        return self._events_processed

    @property
    def events_rejected(self) -> int:
        """Total events rejected (out of order, invalid, etc)."""
        return self._events_rejected

    def record_event_processed(self) -> None:
        """Increment processed event counter."""
        self._events_processed += 1

    # =========================================================================
    # INVARIANT CHECKS
    # =========================================================================

    def check_invariants(self) -> list[str]:
        """Check all state invariants.

        Returns:
            List of violation descriptions (empty if all valid)
        """
        violations = []

        # Check: at most one active attempt
        active_count = sum(1 for a in self._attempts.values() if not a.is_terminal)
        if active_count > 1:
            violations.append(f"Multiple active attempts: {active_count}")

        # Check: all order fill counts are valid
        for order in self._orders.values():
            if order.filled_qty < 0:
                violations.append(f"Negative fill qty on order {order.order_id}")
            if order.filled_qty > order.quantity:
                violations.append(f"Fill exceeds order qty on {order.order_id}")

        # Check: attempt fill counts match order fills
        for attempt in self._attempts.values():
            orders = self.get_orders_for_attempt(attempt.attempt_id)
            k_fill = sum(o.filled_qty for o in orders if o.venue == "kalshi")
            p_fill = sum(o.filled_qty for o in orders if o.venue == "poly")
            if k_fill != attempt.kalshi_filled:
                violations.append(
                    f"Attempt {attempt.attempt_id} K fill mismatch: {attempt.kalshi_filled} vs {k_fill}"
                )
            if p_fill != attempt.poly_filled:
                violations.append(
                    f"Attempt {attempt.attempt_id} P fill mismatch: {attempt.poly_filled} vs {p_fill}"
                )

        return violations

    def assert_invariants(self) -> None:
        """Assert all invariants hold.

        Raises:
            InvariantViolationError: If any invariant is violated
        """
        violations = self.check_invariants()
        if violations:
            raise InvariantViolationError(f"Invariant violations: {violations}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def derive_client_order_id(attempt_id: str, venue: str, leg: str) -> str:
    """Derive idempotent client_order_id from attempt, venue, and leg.

    Format: A-{attempt_suffix}-{VENUE}-{LEG}
    e.g., A-abc12345-KALSHI-YES

    This enables:
    - Venue-side idempotency (retry submits won't create duplicates)
    - Duplicate submit prevention
    - Easy correlation of orders to attempts

    Args:
        attempt_id: The attempt ID (e.g., "attempt_abc12345")
        venue: Venue name ("kalshi" or "poly")
        leg: Leg identifier ("YES" or "NO")

    Returns:
        Formatted client_order_id string
    """
    attempt_suffix = attempt_id.replace("attempt_", "")
    return f"A-{attempt_suffix}-{venue.upper()}-{leg.upper()}"


def create_state_manager(
    kalshi_balance: float = 0.0,
    poly_balance: float = 0.0,
) -> StateManager:
    """Factory to create StateManager with initial balances.

    Args:
        kalshi_balance: Initial Kalshi balance
        poly_balance: Initial Poly balance

    Returns:
        Configured StateManager
    """
    inventory = Inventory(
        kalshi_balance=kalshi_balance,
        poly_balance=poly_balance,
    )
    return StateManager(initial_inventory=inventory)
