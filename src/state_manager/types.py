"""Core types for the State Manager FSM.

Single source of truth for order lifecycle, attempt tracking, and state machine.

Key design principles:
- All dataclasses immutable (frozen=True) for thread safety
- Explicit state machine with defined transitions
- Events carry enough context for single-writer to apply atomically
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional
import time


# =============================================================================
# ORDER STATES
# =============================================================================

class OrderState(Enum):
    """Lifecycle states for a single order.

    State Machine:
        PENDING → SENT → ACKED → FILLED
                       ↘ PARTIAL → FILLED
                       ↘ CANCELLED
                       ↘ REJECTED
                       ↘ EXPIRED
    """
    PENDING = "pending"      # Created, not yet submitted
    SENT = "sent"            # Submitted to venue, awaiting ack
    ACKED = "acked"          # Venue acknowledged, on book or IOC pending
    PARTIAL = "partial"      # Partially filled
    FILLED = "filled"        # Fully filled (terminal)
    CANCELLED = "cancelled"  # Cancelled by us or venue (terminal)
    REJECTED = "rejected"    # Rejected by venue (terminal)
    EXPIRED = "expired"      # Time expired (terminal)


# Terminal order states - no further transitions allowed
TERMINAL_ORDER_STATES = frozenset({
    OrderState.FILLED,
    OrderState.CANCELLED,
    OrderState.REJECTED,
    OrderState.EXPIRED,
})


# Valid order state transitions
ORDER_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PENDING: frozenset({OrderState.SENT, OrderState.CANCELLED}),
    OrderState.SENT: frozenset({OrderState.ACKED, OrderState.REJECTED, OrderState.EXPIRED}),
    OrderState.ACKED: frozenset({OrderState.PARTIAL, OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}),
    OrderState.PARTIAL: frozenset({OrderState.FILLED, OrderState.CANCELLED, OrderState.EXPIRED}),
    # Terminal states have no valid transitions
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


def is_valid_order_transition(from_state: OrderState, to_state: OrderState) -> bool:
    """Check if an order state transition is valid."""
    return to_state in ORDER_TRANSITIONS.get(from_state, frozenset())


# =============================================================================
# ATTEMPT STATES
# =============================================================================

class AttemptState(Enum):
    """Lifecycle states for an arb attempt (paired orders).

    State Machine:
        IDLE → EXECUTING → RECONCILING → DONE_BALANCED
                                       → DONE_IMBALANCED
                                       → FLATTENING → DONE_FLAT
        Any state except IDLE can → FAILED
    """
    IDLE = "idle"                    # No active attempt
    EXECUTING = "executing"          # Submitting orders
    RECONCILING = "reconciling"      # Checking fills, possibly chasing
    DONE_BALANCED = "done_balanced"  # Fills matched (terminal success)
    DONE_IMBALANCED = "done_imbalanced"  # Imbalanced but accepted (terminal)
    FLATTENING = "flattening"        # Closing unhedged position
    DONE_FLAT = "done_flat"          # Flattening complete (terminal)
    FAILED = "failed"                # Attempt failed (terminal error)


# Terminal attempt states
TERMINAL_ATTEMPT_STATES = frozenset({
    AttemptState.DONE_BALANCED,
    AttemptState.DONE_IMBALANCED,
    AttemptState.DONE_FLAT,
    AttemptState.FAILED,
})


# Valid attempt state transitions
ATTEMPT_TRANSITIONS: dict[AttemptState, frozenset[AttemptState]] = {
    AttemptState.IDLE: frozenset({AttemptState.EXECUTING}),
    AttemptState.EXECUTING: frozenset({AttemptState.RECONCILING, AttemptState.FLATTENING, AttemptState.FAILED}),
    AttemptState.RECONCILING: frozenset({AttemptState.DONE_BALANCED, AttemptState.DONE_IMBALANCED, AttemptState.FLATTENING, AttemptState.FAILED}),
    AttemptState.FLATTENING: frozenset({AttemptState.DONE_FLAT, AttemptState.FAILED}),
    # Terminal states
    AttemptState.DONE_BALANCED: frozenset({AttemptState.IDLE}),  # Can reset to IDLE for next attempt
    AttemptState.DONE_IMBALANCED: frozenset({AttemptState.IDLE}),
    AttemptState.DONE_FLAT: frozenset({AttemptState.IDLE}),
    AttemptState.FAILED: frozenset({AttemptState.IDLE}),
}


def is_valid_attempt_transition(from_state: AttemptState, to_state: AttemptState) -> bool:
    """Check if an attempt state transition is valid."""
    return to_state in ATTEMPT_TRANSITIONS.get(from_state, frozenset())


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """All events that can affect state.

    Events are produced by venue adapters and processed by state manager.
    Each event type maps to specific state transitions.
    """
    # Order lifecycle events (from venues)
    ORDER_SENT = auto()          # Order submitted to venue
    ORDER_ACKED = auto()         # Venue acknowledged order
    ORDER_FILL = auto()          # Fill received (partial or full)
    ORDER_CANCELLED = auto()     # Order cancelled
    ORDER_REJECTED = auto()      # Order rejected by venue
    ORDER_EXPIRED = auto()       # Order expired

    # Attempt lifecycle events (internal)
    ATTEMPT_START = auto()       # New arb attempt initiated
    ATTEMPT_EXECUTE = auto()     # Orders being submitted
    ATTEMPT_RECONCILE = auto()   # Reconciliation started
    ATTEMPT_COMPLETE = auto()    # Attempt finished (success)
    ATTEMPT_FLATTEN = auto()     # Flattening needed
    ATTEMPT_FAIL = auto()        # Attempt failed

    # Inventory events
    INVENTORY_UPDATE = auto()    # Position or balance change

    # System events
    MARKET_ROLLOVER = auto()     # Market expired, new market
    SHUTDOWN = auto()            # Graceful shutdown requested


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class Order:
    """Immutable order record.

    Attributes:
        order_id: Unique identifier (local, before venue assigns one)
        client_order_id: Idempotent ID for venue submission (format: A-{attempt_id}-{venue}-{leg})
        venue_order_id: Venue-assigned order ID (after ack)
        venue: "kalshi" or "poly"
        market_id: Market/token ID
        side: "buy" or "sell" (trade direction: BUY/SELL)
        outcome: "YES" or "NO" (which outcome token is being traded)
        price: Limit price (dollars for poly, dollars for kalshi)
        quantity: Requested quantity
        filled_qty: Cumulative filled quantity
        state: Current order state
        created_at_ms: Creation timestamp
        updated_at_ms: Last update timestamp
        attempt_id: Parent attempt ID
    """
    order_id: str
    venue: str
    market_id: str
    side: str
    outcome: str  # "YES" or "NO" - which outcome token
    price: float
    quantity: int
    state: OrderState = OrderState.PENDING
    client_order_id: Optional[str] = None
    venue_order_id: Optional[str] = None
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    attempt_id: Optional[str] = None
    error: Optional[str] = None

    def with_state(self, new_state: OrderState, **updates) -> "Order":
        """Create new Order with updated state."""
        if not is_valid_order_transition(self.state, new_state):
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")
        return Order(
            order_id=self.order_id,
            venue=self.venue,
            market_id=self.market_id,
            side=self.side,
            outcome=self.outcome,
            price=self.price,
            quantity=self.quantity,
            state=new_state,
            client_order_id=self.client_order_id,
            venue_order_id=updates.get('venue_order_id', self.venue_order_id),
            filled_qty=updates.get('filled_qty', self.filled_qty),
            avg_fill_price=updates.get('avg_fill_price', self.avg_fill_price),
            created_at_ms=self.created_at_ms,
            updated_at_ms=int(time.time() * 1000),
            attempt_id=self.attempt_id,
            error=updates.get('error', self.error),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.state in TERMINAL_ORDER_STATES

    @property
    def remaining_qty(self) -> int:
        """Quantity remaining to fill."""
        return self.quantity - self.filled_qty


@dataclass(frozen=True)
class Fill:
    """Record of a single fill.

    Attributes:
        fill_id: Unique fill identifier
        order_id: Parent order ID
        venue: "kalshi" or "poly"
        quantity: Fill quantity
        price: Fill price
        timestamp_ms: Venue timestamp
        received_at_ms: When we received the fill event
    """
    fill_id: str
    order_id: str
    venue: str
    quantity: int
    price: float
    timestamp_ms: int
    received_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(frozen=True)
class Attempt:
    """Record of an arb attempt (paired orders).

    Attributes:
        attempt_id: Unique attempt identifier
        state: Current attempt state
        strategy_path: Arb strategy (1 or 2)
        kalshi_order_id: Kalshi order ID
        poly_order_id: Poly order ID
        target_qty: Target quantity for both legs
        kalshi_filled: Kalshi fill quantity
        poly_filled: Poly fill quantity
        created_at_ms: Attempt start time
        updated_at_ms: Last update time
        completed_at_ms: When attempt reached terminal state
    """
    attempt_id: str
    state: AttemptState
    strategy_path: int
    target_qty: int
    kalshi_order_id: Optional[str] = None
    poly_order_id: Optional[str] = None
    kalshi_filled: int = 0
    poly_filled: int = 0
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    updated_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    completed_at_ms: Optional[int] = None
    error: Optional[str] = None

    def with_state(self, new_state: AttemptState, **updates) -> "Attempt":
        """Create new Attempt with updated state."""
        if not is_valid_attempt_transition(self.state, new_state):
            raise ValueError(f"Invalid transition: {self.state} → {new_state}")

        completed_at = self.completed_at_ms
        if new_state in TERMINAL_ATTEMPT_STATES and completed_at is None:
            completed_at = int(time.time() * 1000)

        return Attempt(
            attempt_id=self.attempt_id,
            state=new_state,
            strategy_path=self.strategy_path,
            target_qty=self.target_qty,
            kalshi_order_id=updates.get('kalshi_order_id', self.kalshi_order_id),
            poly_order_id=updates.get('poly_order_id', self.poly_order_id),
            kalshi_filled=updates.get('kalshi_filled', self.kalshi_filled),
            poly_filled=updates.get('poly_filled', self.poly_filled),
            created_at_ms=self.created_at_ms,
            updated_at_ms=int(time.time() * 1000),
            completed_at_ms=completed_at,
            error=updates.get('error', self.error),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if attempt is in terminal state."""
        return self.state in TERMINAL_ATTEMPT_STATES

    @property
    def is_balanced(self) -> bool:
        """Check if fills are balanced."""
        return self.kalshi_filled == self.poly_filled and self.kalshi_filled > 0

    @property
    def common_qty(self) -> int:
        """Matched quantity across both venues."""
        return min(self.kalshi_filled, self.poly_filled)


@dataclass(frozen=True)
class Inventory:
    """Position and balance tracking.

    Attributes:
        kalshi_position: Net Kalshi contracts (positive=long, negative=short)
        poly_position: Net Poly contracts
        kalshi_balance: Available Kalshi balance (cents)
        poly_balance: Available Poly balance (USDC)
        last_update_ms: Last update timestamp
    """
    kalshi_position: int = 0
    poly_position: int = 0
    kalshi_balance: float = 0.0
    poly_balance: float = 0.0
    last_update_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def with_fill(self, venue: str, qty: int, price: float, side: str) -> "Inventory":
        """Create new Inventory reflecting a fill.

        Args:
            venue: "kalshi" or "poly"
            qty: Fill quantity (always positive)
            price: Fill price
            side: "buy" or "sell"
        """
        cost = qty * price
        sign = 1 if side == "buy" else -1

        if venue == "kalshi":
            return Inventory(
                kalshi_position=self.kalshi_position + (sign * qty),
                poly_position=self.poly_position,
                kalshi_balance=self.kalshi_balance - (sign * cost),
                poly_balance=self.poly_balance,
                last_update_ms=int(time.time() * 1000),
            )
        else:
            return Inventory(
                kalshi_position=self.kalshi_position,
                poly_position=self.poly_position + (sign * qty),
                kalshi_balance=self.kalshi_balance,
                poly_balance=self.poly_balance - (sign * cost),
                last_update_ms=int(time.time() * 1000),
            )

    @property
    def is_flat(self) -> bool:
        """Check if inventory is flat (no positions)."""
        return self.kalshi_position == 0 and self.poly_position == 0

    @property
    def net_position(self) -> int:
        """Net position across venues (should be 0 if hedged)."""
        return self.kalshi_position + self.poly_position


# =============================================================================
# EVENTS
# =============================================================================

@dataclass(frozen=True)
class Event:
    """Base event for state machine.

    Attributes:
        event_type: Type of event
        timestamp_ms: When event occurred (venue time if available)
        received_at_ms: When we received the event
        sequence_num: Monotonic sequence number for ordering
        venue: Source venue (if applicable)
        payload: Event-specific data
    """
    event_type: EventType
    sequence_num: int
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    received_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    venue: Optional[str] = None

    # Order events
    order_id: Optional[str] = None
    venue_order_id: Optional[str] = None
    fill_qty: Optional[int] = None
    fill_price: Optional[float] = None
    error: Optional[str] = None

    # Attempt events
    attempt_id: Optional[str] = None
    strategy_path: Optional[int] = None
    target_qty: Optional[int] = None


# =============================================================================
# SNAPSHOT (for passing to engine)
# =============================================================================

# Default staleness threshold for snapshots
DEFAULT_SNAPSHOT_STALE_MS = 250


@dataclass(frozen=True)
class Snapshot:
    """Immutable snapshot of current state for engine decisions.

    This is passed to the arb engine for opportunity detection.
    Combines venue state with inventory for complete picture.

    Versioned snapshots prevent "stale intent" races where:
    1. Arb loop sees opportunity, creates snapshot
    2. Book updates while arb loop decides
    3. Executor receives command with outdated snapshot
    4. Executor submits based on OLD prices -> bad fill

    The sequence numbers (book_seq_*) track book updates since snapshot creation.
    Use is_stale() to detect when market data may have changed.
    """
    # Market data (from venue_state.py MarketSnapshot)
    poly_yes_ask: Optional[float] = None
    poly_yes_bid: Optional[float] = None
    poly_no_ask: Optional[float] = None
    poly_no_bid: Optional[float] = None
    kalshi_yes_ask: Optional[float] = None
    kalshi_yes_bid: Optional[float] = None

    # Inventory
    inventory: Inventory = field(default_factory=Inventory)

    # Current attempt (if any)
    active_attempt: Optional[Attempt] = None

    # Timestamp
    snapshot_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    # Book sequence numbers for staleness detection
    book_seq_kalshi: int = 0
    book_seq_poly_yes: int = 0
    book_seq_poly_no: int = 0

    @property
    def snapshot_id(self) -> tuple[int, int, int, int]:
        """Unique snapshot identifier: (kalshi_seq, poly_yes_seq, poly_no_seq, timestamp_ms).

        Can be used to compare snapshots and detect if market data has changed.
        Two snapshots with identical snapshot_id were created from the same market state.
        """
        return (self.book_seq_kalshi, self.book_seq_poly_yes, self.book_seq_poly_no, self.snapshot_ms)

    def is_stale(self, max_age_ms: int = DEFAULT_SNAPSHOT_STALE_MS) -> bool:
        """Check if snapshot is older than max_age_ms.

        Args:
            max_age_ms: Maximum allowed age in milliseconds (default: 250ms)

        Returns:
            True if snapshot is stale (older than max_age_ms)
        """
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - self.snapshot_ms
        return age_ms > max_age_ms

    @property
    def can_trade(self) -> bool:
        """Check if we can initiate a new trade."""
        if self.active_attempt is not None and not self.active_attempt.is_terminal:
            return False
        return True


# =============================================================================
# FSM TRANSITION TABLE SUMMARY
# =============================================================================
"""
Order State Machine:
┌─────────┐   sent    ┌──────┐   ack    ┌───────┐
│ PENDING │ ────────> │ SENT │ ──────> │ ACKED │
└─────────┘           └──────┘          └───────┘
     │                    │                 │
     │ cancel             │ reject          │ fill/partial/cancel/expire
     ↓                    ↓                 ↓
┌───────────┐        ┌──────────┐    ┌─────────────────┐
│ CANCELLED │        │ REJECTED │    │ PARTIAL/FILLED/ │
└───────────┘        └──────────┘    │ CANCELLED/EXPIRED│
                                      └─────────────────┘

Attempt State Machine:
┌──────┐  start  ┌───────────┐  submit  ┌─────────────┐
│ IDLE │ ──────> │ EXECUTING │ ───────> │ RECONCILING │
└──────┘         └───────────┘          └─────────────┘
   ↑                  │ fail                 │
   │                  ↓                      │ balanced/imbalanced
   │             ┌────────┐                  ↓
   │             │ FAILED │          ┌───────────────────┐
   │             └────────┘          │ DONE_BALANCED/    │
   │                 ↑               │ DONE_IMBALANCED   │
   │                 │               └───────────────────┘
   │                 │ fail                  │
   │            ┌────────────┐               │ flatten needed
   │            │ FLATTENING │ <─────────────┘
   │            └────────────┘
   │                 │ complete
   │                 ↓
   │            ┌───────────┐
   └────────────│ DONE_FLAT │
                └───────────┘

Event → State Transition Mapping:
- ORDER_SENT: PENDING → SENT
- ORDER_ACKED: SENT → ACKED
- ORDER_FILL: ACKED/PARTIAL → PARTIAL/FILLED
- ORDER_CANCELLED: PENDING/ACKED/PARTIAL → CANCELLED
- ORDER_REJECTED: SENT → REJECTED
- ORDER_EXPIRED: ACKED/PARTIAL → EXPIRED
- ATTEMPT_START: IDLE → EXECUTING
- ATTEMPT_RECONCILE: EXECUTING → RECONCILING
- ATTEMPT_COMPLETE: RECONCILING → DONE_BALANCED/DONE_IMBALANCED
- ATTEMPT_FLATTEN: RECONCILING/EXECUTING → FLATTENING
- ATTEMPT_FAIL: any (except IDLE) → FAILED
"""
