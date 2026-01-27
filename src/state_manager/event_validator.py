"""Event validation and ordering for State Manager.

Validates incoming events before they reach the state manager.
Ensures type safety, required fields, ordering, and idempotency.

Key responsibilities:
- Validate event structure and required fields
- Check sequence numbers for ordering
- Detect and reject duplicates
- Log validation failures with context
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Set
from collections import OrderedDict

from .types import Event, EventType, OrderState, AttemptState

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class MissingFieldError(ValidationError):
    """Raised when required field is missing."""
    pass


class InvalidFieldError(ValidationError):
    """Raised when field has invalid value."""
    pass


class OutOfOrderError(ValidationError):
    """Raised when event is out of sequence."""
    pass


class DuplicateEventError(ValidationError):
    """Raised when event is a duplicate."""
    pass


class StaleEventError(ValidationError):
    """Raised when event is too old."""
    pass


# =============================================================================
# REQUIRED FIELDS BY EVENT TYPE
# =============================================================================

# Maps event type to required fields
REQUIRED_FIELDS: dict[EventType, set[str]] = {
    # Order events require order_id
    EventType.ORDER_SENT: {'order_id'},
    EventType.ORDER_ACKED: {'order_id', 'venue_order_id'},
    EventType.ORDER_FILL: {'order_id', 'fill_qty', 'fill_price'},
    EventType.ORDER_CANCELLED: {'order_id'},
    EventType.ORDER_REJECTED: {'order_id'},
    EventType.ORDER_EXPIRED: {'order_id'},

    # Attempt events require attempt_id
    EventType.ATTEMPT_START: {'attempt_id', 'strategy_path', 'target_qty'},
    EventType.ATTEMPT_EXECUTE: {'attempt_id'},
    EventType.ATTEMPT_RECONCILE: {'attempt_id'},
    EventType.ATTEMPT_COMPLETE: {'attempt_id'},
    EventType.ATTEMPT_FLATTEN: {'attempt_id'},
    EventType.ATTEMPT_FAIL: {'attempt_id'},

    # Inventory events
    EventType.INVENTORY_UPDATE: set(),

    # System events
    EventType.MARKET_ROLLOVER: set(),
    EventType.SHUTDOWN: set(),
}


# Events that require venue
VENUE_REQUIRED_EVENTS: set[EventType] = {
    EventType.ORDER_SENT,
    EventType.ORDER_ACKED,
    EventType.ORDER_FILL,
    EventType.ORDER_CANCELLED,
    EventType.ORDER_REJECTED,
    EventType.ORDER_EXPIRED,
}


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Result of event validation.

    Attributes:
        valid: Whether event passed validation
        event: The validated event (possibly normalized)
        error: Error message if invalid
        error_type: Type of validation error
    """
    valid: bool
    event: Optional[Event] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    @staticmethod
    def success(event: Event) -> "ValidationResult":
        return ValidationResult(valid=True, event=event)

    @staticmethod
    def failure(error: str, error_type: str) -> "ValidationResult":
        return ValidationResult(valid=False, error=error, error_type=error_type)


# =============================================================================
# LRU CACHE FOR DUPLICATE DETECTION
# =============================================================================

class LRUSet:
    """LRU-evicting set for duplicate detection.

    Maintains a bounded set of recently seen items, evicting oldest
    when capacity is exceeded.
    """

    def __init__(self, max_size: int = 10000):
        self._max_size = max_size
        self._items: OrderedDict[str, None] = OrderedDict()

    def add(self, item: str) -> bool:
        """Add item to set.

        Args:
            item: Item to add

        Returns:
            True if item was new, False if duplicate
        """
        if item in self._items:
            # Move to end (most recently seen)
            self._items.move_to_end(item)
            return False

        self._items[item] = None

        # Evict oldest if over capacity
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

        return True

    def __contains__(self, item: str) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)


# =============================================================================
# EVENT VALIDATOR
# =============================================================================

@dataclass
class ValidationMetrics:
    """Metrics for validation monitoring."""
    total_events: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    duplicates_rejected: int = 0
    out_of_order_rejected: int = 0
    stale_rejected: int = 0
    missing_field_rejected: int = 0
    invalid_field_rejected: int = 0


class EventValidator:
    """Validates and filters incoming events.

    Checks:
    1. Required fields present
    2. Field values valid
    3. Sequence ordering per venue
    4. Duplicate detection
    5. Staleness (event age)

    Usage:
        validator = EventValidator()
        result = validator.validate(event)
        if result.valid:
            process(result.event)
        else:
            log_error(result.error)
    """

    def __init__(
        self,
        max_event_age_ms: int = 60000,  # 1 minute default
        duplicate_cache_size: int = 10000,
        allow_out_of_order: bool = False,
    ):
        """Initialize validator.

        Args:
            max_event_age_ms: Maximum age for events (older rejected as stale)
            duplicate_cache_size: Size of duplicate detection cache
            allow_out_of_order: If True, don't reject out-of-order events
        """
        self._max_event_age_ms = max_event_age_ms
        self._allow_out_of_order = allow_out_of_order

        # Per-venue sequence tracking
        self._last_sequence: dict[str, int] = {}

        # Duplicate detection
        self._seen_events = LRUSet(max_size=duplicate_cache_size)

        # Metrics
        self._metrics = ValidationMetrics()

    @property
    def metrics(self) -> ValidationMetrics:
        """Get validation metrics."""
        return self._metrics

    def validate(self, event: Event) -> ValidationResult:
        """Validate an event.

        Args:
            event: Event to validate

        Returns:
            ValidationResult with success/failure and details
        """
        self._metrics.total_events += 1

        try:
            # 1. Check required fields
            self._check_required_fields(event)

            # 2. Check field values
            self._check_field_values(event)

            # 3. Check for duplicates
            self._check_duplicate(event)

            # 4. Check staleness
            self._check_staleness(event)

            # 5. Check sequence ordering
            self._check_sequence(event)

            self._metrics.valid_events += 1
            return ValidationResult.success(event)

        except MissingFieldError as e:
            self._metrics.invalid_events += 1
            self._metrics.missing_field_rejected += 1
            logger.warning("Event validation failed (missing field): %s", e)
            return ValidationResult.failure(str(e), "missing_field")

        except InvalidFieldError as e:
            self._metrics.invalid_events += 1
            self._metrics.invalid_field_rejected += 1
            logger.warning("Event validation failed (invalid field): %s", e)
            return ValidationResult.failure(str(e), "invalid_field")

        except DuplicateEventError as e:
            self._metrics.invalid_events += 1
            self._metrics.duplicates_rejected += 1
            logger.debug("Duplicate event rejected: %s", e)
            return ValidationResult.failure(str(e), "duplicate")

        except StaleEventError as e:
            self._metrics.invalid_events += 1
            self._metrics.stale_rejected += 1
            logger.warning("Stale event rejected: %s", e)
            return ValidationResult.failure(str(e), "stale")

        except OutOfOrderError as e:
            self._metrics.invalid_events += 1
            self._metrics.out_of_order_rejected += 1
            logger.warning("Out-of-order event rejected: %s", e)
            return ValidationResult.failure(str(e), "out_of_order")

    def _check_required_fields(self, event: Event) -> None:
        """Check that all required fields are present."""
        required = REQUIRED_FIELDS.get(event.event_type, set())

        for field_name in required:
            value = getattr(event, field_name, None)
            if value is None:
                raise MissingFieldError(
                    f"Missing required field '{field_name}' for {event.event_type.name}"
                )

        # Check venue for venue-required events
        if event.event_type in VENUE_REQUIRED_EVENTS and event.venue is None:
            raise MissingFieldError(
                f"Missing venue for {event.event_type.name}"
            )

    def _check_field_values(self, event: Event) -> None:
        """Check that field values are valid."""
        # Venue must be valid
        if event.venue is not None and event.venue not in ("kalshi", "poly"):
            raise InvalidFieldError(f"Invalid venue: {event.venue}")

        # Sequence must be non-negative
        if event.sequence_num < 0:
            raise InvalidFieldError(f"Negative sequence number: {event.sequence_num}")

        # Fill quantity must be positive
        if event.fill_qty is not None and event.fill_qty <= 0:
            raise InvalidFieldError(f"Fill quantity must be positive: {event.fill_qty}")

        # Fill price must be non-negative
        if event.fill_price is not None and event.fill_price < 0:
            raise InvalidFieldError(f"Fill price must be non-negative: {event.fill_price}")

        # Strategy path must be 1 or 2
        if event.strategy_path is not None and event.strategy_path not in (1, 2):
            raise InvalidFieldError(f"Invalid strategy path: {event.strategy_path}")

        # Target qty must be positive
        if event.target_qty is not None and event.target_qty <= 0:
            raise InvalidFieldError(f"Target quantity must be positive: {event.target_qty}")

    def _check_duplicate(self, event: Event) -> None:
        """Check if event is a duplicate."""
        # Create unique event fingerprint
        fingerprint = self._event_fingerprint(event)

        if not self._seen_events.add(fingerprint):
            raise DuplicateEventError(f"Duplicate event: {fingerprint}")

    def _check_staleness(self, event: Event) -> None:
        """Check if event is too old."""
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - event.timestamp_ms

        if age_ms > self._max_event_age_ms:
            raise StaleEventError(
                f"Event too old: age={age_ms}ms max={self._max_event_age_ms}ms"
            )

    def _check_sequence(self, event: Event) -> None:
        """Check event sequence ordering."""
        if self._allow_out_of_order:
            return

        if event.venue is None:
            return  # Internal events don't have venue sequence

        last_seq = self._last_sequence.get(event.venue, -1)

        if event.sequence_num < last_seq:
            raise OutOfOrderError(
                f"Out of order: venue={event.venue} seq={event.sequence_num} last={last_seq}"
            )

        self._last_sequence[event.venue] = event.sequence_num

    def _event_fingerprint(self, event: Event) -> str:
        """Create unique fingerprint for duplicate detection."""
        # Combine event type, venue, and relevant IDs
        parts = [
            event.event_type.name,
            str(event.sequence_num),
            event.venue or "internal",
        ]

        if event.order_id:
            parts.append(event.order_id)
        if event.attempt_id:
            parts.append(event.attempt_id)
        if event.fill_qty is not None:
            parts.append(f"fill_{event.fill_qty}_{event.fill_price}")

        return ":".join(parts)

    def reset_sequence(self, venue: Optional[str] = None) -> None:
        """Reset sequence tracking.

        Args:
            venue: Venue to reset (None for all venues)
        """
        if venue is None:
            self._last_sequence.clear()
            logger.info("Reset all venue sequences")
        else:
            self._last_sequence.pop(venue, None)
            logger.info("Reset sequence for venue: %s", venue)

    def clear_duplicate_cache(self) -> None:
        """Clear duplicate detection cache."""
        self._seen_events = LRUSet(max_size=self._seen_events._max_size)
        logger.info("Cleared duplicate cache")


# =============================================================================
# EVENT QUEUE WITH BACKPRESSURE
# =============================================================================

@dataclass
class QueueMetrics:
    """Metrics for event queue monitoring."""
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    high_watermark: int = 0


class BoundedEventQueue:
    """Bounded event queue with backpressure support.

    Provides:
    - Maximum capacity with drop policy
    - Backpressure signaling
    - Priority for critical events
    - Metrics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        high_watermark_pct: float = 0.8,
        drop_policy: str = "oldest",  # "oldest" or "newest"
    ):
        """Initialize queue.

        Args:
            max_size: Maximum queue capacity
            high_watermark_pct: Percentage at which backpressure is signaled
            drop_policy: Which events to drop when full
        """
        self._max_size = max_size
        self._high_watermark = int(max_size * high_watermark_pct)
        self._drop_policy = drop_policy
        self._queue: list[Event] = []
        self._metrics = QueueMetrics()

    @property
    def metrics(self) -> QueueMetrics:
        """Get queue metrics."""
        return self._metrics

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return len(self._queue) >= self._max_size

    @property
    def backpressure(self) -> bool:
        """Check if backpressure is signaled."""
        return len(self._queue) >= self._high_watermark

    def enqueue(self, event: Event) -> bool:
        """Add event to queue.

        Args:
            event: Event to enqueue

        Returns:
            True if enqueued, False if dropped
        """
        if self.is_full:
            if self._drop_policy == "oldest":
                # Drop oldest to make room
                self._queue.pop(0)
                self._metrics.dropped += 1
                logger.warning("Queue full, dropped oldest event")
            else:
                # Drop this (newest) event
                self._metrics.dropped += 1
                logger.warning("Queue full, dropped new event: %s", event.event_type.name)
                return False

        self._queue.append(event)
        self._metrics.enqueued += 1

        # Track high watermark
        if len(self._queue) > self._metrics.high_watermark:
            self._metrics.high_watermark = len(self._queue)

        return True

    def dequeue(self) -> Optional[Event]:
        """Remove and return next event.

        Returns:
            Next event or None if empty
        """
        if not self._queue:
            return None

        event = self._queue.pop(0)
        self._metrics.dequeued += 1
        return event

    def peek(self) -> Optional[Event]:
        """Return next event without removing.

        Returns:
            Next event or None if empty
        """
        if not self._queue:
            return None
        return self._queue[0]

    def clear(self) -> int:
        """Clear all events from queue.

        Returns:
            Number of events cleared
        """
        count = len(self._queue)
        self._queue.clear()
        return count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_validator(
    max_event_age_ms: int = 60000,
    allow_out_of_order: bool = False,
) -> EventValidator:
    """Factory to create configured EventValidator.

    Args:
        max_event_age_ms: Maximum event age
        allow_out_of_order: Allow out-of-order events

    Returns:
        Configured EventValidator
    """
    return EventValidator(
        max_event_age_ms=max_event_age_ms,
        allow_out_of_order=allow_out_of_order,
    )


def create_event_queue(
    max_size: int = 1000,
    drop_oldest: bool = True,
) -> BoundedEventQueue:
    """Factory to create configured event queue.

    Args:
        max_size: Maximum queue capacity
        drop_oldest: If True, drop oldest when full; else drop newest

    Returns:
        Configured BoundedEventQueue
    """
    return BoundedEventQueue(
        max_size=max_size,
        drop_policy="oldest" if drop_oldest else "newest",
    )
