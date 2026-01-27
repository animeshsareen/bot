"""Event consumer coroutine for State Manager.

Single async coroutine that processes events from the queue,
validates them, applies FSM transitions, and updates state atomically.

Key design principles:
- Single-writer: only this coroutine mutates state
- Event-driven: all state changes flow through events
- Graceful shutdown: drains queue before exiting
- Error recovery: bad events go to dead-letter, consumer continues
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Any
from enum import Enum

from .types import (
    Event, EventType, OrderState, AttemptState,
    TERMINAL_ORDER_STATES, TERMINAL_ATTEMPT_STATES,
)
from .state import StateManager, StateError, InvalidTransitionError, OutOfOrderEventError
from .event_validator import EventValidator, ValidationResult, BoundedEventQueue
from .commands import SubmitOrderCommand, FlattenCommand, Command
from .command_executor import CommandQueue

logger = logging.getLogger(__name__)


# =============================================================================
# CONSUMER STATE
# =============================================================================

class ConsumerState(Enum):
    """State of the event consumer."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"


# =============================================================================
# HEALTH METRICS
# =============================================================================

@dataclass
class ConsumerMetrics:
    """Health metrics for consumer monitoring."""
    events_processed: int = 0
    events_failed: int = 0
    events_dead_lettered: int = 0
    processing_errors: int = 0
    avg_processing_ms: float = 0.0
    max_processing_ms: float = 0.0
    last_event_ms: int = 0
    uptime_ms: int = 0
    queue_depth_samples: list[int] = field(default_factory=list)

    @property
    def events_per_sec(self) -> float:
        """Average events per second."""
        if self.uptime_ms == 0:
            return 0.0
        return self.events_processed / (self.uptime_ms / 1000)

    @property
    def avg_queue_depth(self) -> float:
        """Average queue depth from samples."""
        if not self.queue_depth_samples:
            return 0.0
        return sum(self.queue_depth_samples) / len(self.queue_depth_samples)


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

@dataclass(frozen=True)
class DeadLetterEntry:
    """Entry in dead letter queue."""
    event: Event
    error: str
    timestamp_ms: int
    attempts: int = 1


class DeadLetterQueue:
    """Queue for events that failed processing.

    Events that repeatedly fail are moved here for manual inspection.
    """

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._entries: list[DeadLetterEntry] = []

    def add(self, event: Event, error: str, attempts: int = 1) -> None:
        """Add event to dead letter queue."""
        entry = DeadLetterEntry(
            event=event,
            error=error,
            timestamp_ms=int(time.time() * 1000),
            attempts=attempts,
        )
        self._entries.append(entry)

        # Evict oldest if over capacity
        while len(self._entries) > self._max_size:
            self._entries.pop(0)

        logger.error(
            "Event dead-lettered: type=%s error=%s attempts=%d",
            event.event_type.name, error, attempts
        )

    def get_all(self) -> list[DeadLetterEntry]:
        """Get all dead letter entries."""
        return list(self._entries)

    def clear(self) -> int:
        """Clear dead letter queue.

        Returns:
            Number of entries cleared
        """
        count = len(self._entries)
        self._entries.clear()
        return count

    @property
    def size(self) -> int:
        return len(self._entries)


# =============================================================================
# EVENT HANDLERS
# =============================================================================

class EventHandlers:
    """Maps event types to handler functions.

    Each handler takes (state_manager, event) and applies the appropriate
    state transitions. Optionally emits commands to CommandQueue.
    """

    def __init__(
        self,
        state: StateManager,
        command_queue: Optional[CommandQueue] = None,
    ):
        self._state = state
        self._command_queue = command_queue
        self._handlers: dict[EventType, Callable[[Event], None]] = {
            EventType.ORDER_SENT: self._handle_order_sent,
            EventType.ORDER_ACKED: self._handle_order_acked,
            EventType.ORDER_FILL: self._handle_order_fill,
            EventType.ORDER_CANCELLED: self._handle_order_cancelled,
            EventType.ORDER_REJECTED: self._handle_order_rejected,
            EventType.ORDER_EXPIRED: self._handle_order_expired,
            EventType.ATTEMPT_START: self._handle_attempt_start,
            EventType.ATTEMPT_EXECUTE: self._handle_attempt_execute,
            EventType.ATTEMPT_RECONCILE: self._handle_attempt_reconcile,
            EventType.ATTEMPT_COMPLETE: self._handle_attempt_complete,
            EventType.ATTEMPT_FLATTEN: self._handle_attempt_flatten,
            EventType.ATTEMPT_FAIL: self._handle_attempt_fail,
            EventType.INVENTORY_UPDATE: self._handle_inventory_update,
            EventType.MARKET_ROLLOVER: self._handle_market_rollover,
            EventType.SHUTDOWN: self._handle_shutdown,
        }

    def handle(self, event: Event) -> None:
        """Dispatch event to appropriate handler."""
        handler = self._handlers.get(event.event_type)
        if handler is None:
            raise ValueError(f"No handler for event type: {event.event_type}")
        handler(event)

    # --- Order handlers ---

    def _handle_order_sent(self, event: Event) -> None:
        """Handle ORDER_SENT: PENDING → SENT."""
        self._state.update_order_state(
            event.order_id,
            OrderState.SENT,
        )

    def _handle_order_acked(self, event: Event) -> None:
        """Handle ORDER_ACKED: SENT → ACKED."""
        self._state.update_order_state(
            event.order_id,
            OrderState.ACKED,
            venue_order_id=event.venue_order_id,
        )

    def _handle_order_fill(self, event: Event) -> None:
        """Handle ORDER_FILL: record fill and update state.

        After recording the fill, checks if a hedge is needed:
        - If attempt is imbalanced after fill, may emit hedge command
        """
        self._state.record_fill(
            event.order_id,
            event.fill_qty,
            event.fill_price,
            timestamp_ms=event.timestamp_ms,
        )

        # Check if hedge is needed after this fill
        self._maybe_emit_hedge_command(event)

    def _handle_order_cancelled(self, event: Event) -> None:
        """Handle ORDER_CANCELLED: * → CANCELLED."""
        self._state.update_order_state(
            event.order_id,
            OrderState.CANCELLED,
        )

    def _handle_order_rejected(self, event: Event) -> None:
        """Handle ORDER_REJECTED: SENT → REJECTED."""
        self._state.update_order_state(
            event.order_id,
            OrderState.REJECTED,
            error=event.error,
        )

    def _handle_order_expired(self, event: Event) -> None:
        """Handle ORDER_EXPIRED: ACKED/PARTIAL → EXPIRED."""
        self._state.update_order_state(
            event.order_id,
            OrderState.EXPIRED,
        )

    # --- Attempt handlers ---

    def _handle_attempt_start(self, event: Event) -> None:
        """Handle ATTEMPT_START: create new attempt."""
        # Create attempt
        attempt = self._state.create_attempt(
            strategy_path=event.strategy_path,
            target_qty=event.target_qty,
        )
        # If attempt_id in event differs from created, update mapping
        # (in practice, attempt_id should be generated here)
        logger.debug("Created attempt %s from event", attempt.attempt_id)

    def _handle_attempt_execute(self, event: Event) -> None:
        """Handle ATTEMPT_EXECUTE: IDLE → EXECUTING."""
        self._state.update_attempt_state(
            event.attempt_id,
            AttemptState.EXECUTING,
        )

    def _handle_attempt_reconcile(self, event: Event) -> None:
        """Handle ATTEMPT_RECONCILE: EXECUTING → RECONCILING."""
        self._state.update_attempt_state(
            event.attempt_id,
            AttemptState.RECONCILING,
        )

    def _handle_attempt_complete(self, event: Event) -> None:
        """Handle ATTEMPT_COMPLETE: RECONCILING → DONE_*."""
        attempt = self._state.get_attempt(event.attempt_id)
        if attempt is None:
            raise KeyError(f"Attempt not found: {event.attempt_id}")

        # Determine outcome based on fills
        if attempt.is_balanced:
            new_state = AttemptState.DONE_BALANCED
        else:
            new_state = AttemptState.DONE_IMBALANCED

        self._state.update_attempt_state(event.attempt_id, new_state)

    def _handle_attempt_flatten(self, event: Event) -> None:
        """Handle ATTEMPT_FLATTEN: * → FLATTENING.

        Also emits FlattenCommand to command queue if configured.
        """
        self._state.update_attempt_state(
            event.attempt_id,
            AttemptState.FLATTENING,
        )

        # Emit flatten command
        self._emit_flatten_command(event)

    def _handle_attempt_fail(self, event: Event) -> None:
        """Handle ATTEMPT_FAIL: * → FAILED."""
        self._state.update_attempt_state(
            event.attempt_id,
            AttemptState.FAILED,
            error=event.error,
        )

    # --- Command emission helpers ---

    def _maybe_emit_hedge_command(self, event: Event) -> None:
        """Check if a hedge is needed after fill and emit command.

        Called after ORDER_FILL is processed. If the fill creates an
        imbalance that needs hedging, emits a SubmitOrderCommand.
        """
        if self._command_queue is None:
            return

        # Get the order that was filled
        order = self._state.get_order(event.order_id)
        if order is None or order.attempt_id is None:
            return

        # Get the parent attempt
        attempt = self._state.get_attempt(order.attempt_id)
        if attempt is None or attempt.is_terminal:
            return

        # Check for imbalance that needs hedging
        # Only hedge when one leg is fully filled and the other is not
        imbalance = abs(attempt.kalshi_filled - attempt.poly_filled)
        if imbalance == 0:
            return

        # Determine which venue needs hedging
        if attempt.kalshi_filled > attempt.poly_filled:
            # Need to hedge on poly
            hedge_venue = "poly"
            hedge_qty = attempt.kalshi_filled - attempt.poly_filled
            # If Kalshi filled YES, we need to buy YES on Poly (or sell NO)
            hedge_side = "buy"  # Simplified - real logic depends on strategy
        else:
            # Need to hedge on kalshi
            hedge_venue = "kalshi"
            hedge_qty = attempt.poly_filled - attempt.kalshi_filled
            hedge_side = "buy"  # Simplified

        # Only emit hedge if order on filled side is terminal (fully done)
        filled_order = None
        if order.venue == "kalshi" and attempt.kalshi_filled >= attempt.target_qty:
            filled_order = order
        elif order.venue == "poly" and attempt.poly_filled >= attempt.target_qty:
            filled_order = order

        if filled_order is None:
            # Still waiting for more fills on the triggering side
            return

        # Get the other leg's order to find market_id
        other_orders = [
            o for o in self._state.get_orders_for_attempt(attempt.attempt_id)
            if o.venue == hedge_venue
        ]
        if not other_orders:
            logger.warning("No order found for hedge venue %s", hedge_venue)
            return

        other_order = other_orders[0]

        # Create hedge command
        from .state import derive_client_order_id
        client_order_id = derive_client_order_id(
            attempt.attempt_id, hedge_venue, "HEDGE"
        )

        cmd = SubmitOrderCommand(
            attempt_id=attempt.attempt_id,
            venue=hedge_venue,
            market_id=other_order.market_id,
            side=hedge_side,
            outcome=other_order.outcome,  # Use same outcome as original order
            quantity=hedge_qty,
            price=other_order.price,  # Use same price as original order
            client_order_id=client_order_id,
            order_id=f"{hedge_venue}_hedge_{attempt.attempt_id[-8:]}",
        )

        logger.info(
            "Emitting hedge command: %s %d on %s (imbalance=%d)",
            hedge_side, hedge_qty, hedge_venue, imbalance
        )

        # Enqueue asynchronously (fire-and-forget since we're sync here)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._command_queue.put(cmd))
        except RuntimeError:
            # No event loop running - queue directly if possible
            logger.warning("No event loop for hedge command emission")

    def _emit_flatten_command(self, event: Event) -> None:
        """Emit FlattenCommand when ATTEMPT_FLATTEN is processed."""
        if self._command_queue is None:
            return

        attempt = self._state.get_attempt(event.attempt_id)
        if attempt is None:
            return

        # Determine what needs flattening based on imbalance
        inventory = self._state.get_inventory()

        # Flatten the venue with excess position
        if inventory.kalshi_position > inventory.poly_position:
            flatten_venue = "kalshi"
            flatten_qty = inventory.kalshi_position - inventory.poly_position
            flatten_side = "sell"  # Sell to reduce long position
        elif inventory.poly_position > inventory.kalshi_position:
            flatten_venue = "poly"
            flatten_qty = inventory.poly_position - inventory.kalshi_position
            flatten_side = "sell"
        else:
            # Already balanced
            return

        if flatten_qty <= 0:
            return

        # Get market_id from attempt's orders
        orders = self._state.get_orders_for_attempt(attempt.attempt_id)
        market_orders = [o for o in orders if o.venue == flatten_venue]
        if not market_orders:
            logger.warning("No orders found for flatten on %s", flatten_venue)
            return

        market_id = market_orders[0].market_id

        cmd = FlattenCommand(
            venue=flatten_venue,
            quantity=flatten_qty,
            side=flatten_side,
            market_id=market_id,
            attempt_id=attempt.attempt_id,
            reason="attempt_flatten",
        )

        logger.info(
            "Emitting flatten command: %s %d on %s",
            flatten_side, flatten_qty, flatten_venue
        )

        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._command_queue.put(cmd))
        except RuntimeError:
            logger.warning("No event loop for flatten command emission")

    # --- Other handlers ---

    def _handle_inventory_update(self, event: Event) -> None:
        """Handle INVENTORY_UPDATE: refresh inventory from external source."""
        # This would typically fetch latest balances from venues
        # For now, just log
        logger.info("Inventory update event received")

    def _handle_market_rollover(self, event: Event) -> None:
        """Handle MARKET_ROLLOVER: market expired, reset for new market."""
        logger.info("Market rollover event - clearing active attempts")
        # Force-complete any active attempts
        active = self._state.get_active_attempt()
        if active and not active.is_terminal:
            self._state.update_attempt_state(
                active.attempt_id,
                AttemptState.FAILED,
                error="market_rollover",
            )

    def _handle_shutdown(self, event: Event) -> None:
        """Handle SHUTDOWN: graceful shutdown requested."""
        logger.info("Shutdown event received")
        # Caller should check for this and stop the consumer


# =============================================================================
# EVENT CONSUMER
# =============================================================================

class EventConsumer:
    """Main event consumer coroutine.

    Single async task that:
    1. Pulls events from queue
    2. Validates events
    3. Applies state transitions
    4. Tracks metrics
    5. Handles errors gracefully

    Usage:
        consumer = EventConsumer(state, validator, queue)
        task = asyncio.create_task(consumer.run())
        ...
        await consumer.shutdown()
    """

    def __init__(
        self,
        state: StateManager,
        validator: EventValidator,
        queue: BoundedEventQueue,
        command_queue: Optional[CommandQueue] = None,
        poll_interval_ms: int = 10,
        max_retries: int = 3,
        metrics_interval_ms: int = 5000,
    ):
        """Initialize consumer.

        Args:
            state: StateManager instance
            validator: EventValidator instance
            queue: Event queue to consume from
            command_queue: Optional CommandQueue to emit commands to
            poll_interval_ms: How often to poll queue when empty
            max_retries: Max retries before dead-lettering
            metrics_interval_ms: How often to sample metrics
        """
        self._state = state
        self._validator = validator
        self._queue = queue
        self._command_queue = command_queue
        self._poll_interval_ms = poll_interval_ms
        self._max_retries = max_retries
        self._metrics_interval_ms = metrics_interval_ms

        self._handlers = EventHandlers(state, command_queue)
        self._dead_letter = DeadLetterQueue()
        self._metrics = ConsumerMetrics()

        self._consumer_state = ConsumerState.STOPPED
        self._shutdown_event = asyncio.Event()
        self._started_at_ms: Optional[int] = None

        # Retry tracking
        self._retry_counts: dict[str, int] = {}  # event fingerprint -> retry count

    @property
    def state(self) -> ConsumerState:
        """Current consumer state."""
        return self._consumer_state

    @property
    def metrics(self) -> ConsumerMetrics:
        """Consumer health metrics."""
        return self._metrics

    @property
    def dead_letters(self) -> DeadLetterQueue:
        """Dead letter queue."""
        return self._dead_letter

    @property
    def is_running(self) -> bool:
        """Check if consumer is running."""
        return self._consumer_state in (ConsumerState.RUNNING, ConsumerState.DRAINING)

    async def run(self) -> None:
        """Main consumer loop.

        Runs until shutdown() is called. Processes events from queue,
        validates, applies transitions, handles errors.
        """
        self._consumer_state = ConsumerState.STARTING
        self._started_at_ms = int(time.time() * 1000)
        self._shutdown_event.clear()

        logger.info("Event consumer starting")

        self._consumer_state = ConsumerState.RUNNING
        last_metrics_ms = self._started_at_ms

        try:
            while not self._shutdown_event.is_set():
                # Process next event
                event = self._queue.dequeue()

                if event is None:
                    # Queue empty, wait before polling again
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=self._poll_interval_ms / 1000,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Process the event
                await self._process_event(event)

                # Sample metrics periodically
                now_ms = int(time.time() * 1000)
                if now_ms - last_metrics_ms >= self._metrics_interval_ms:
                    self._sample_metrics()
                    last_metrics_ms = now_ms

            # Draining phase: process remaining events
            self._consumer_state = ConsumerState.DRAINING
            logger.info("Consumer draining queue (%d events)", self._queue.size)

            while True:
                event = self._queue.dequeue()
                if event is None:
                    break
                await self._process_event(event)

        except asyncio.CancelledError:
            logger.info("Consumer cancelled")
            raise

        except Exception as e:
            logger.error("Consumer fatal error: %s", e, exc_info=True)
            raise

        finally:
            self._consumer_state = ConsumerState.STOPPED
            self._update_uptime()
            logger.info(
                "Consumer stopped: processed=%d failed=%d dead_lettered=%d",
                self._metrics.events_processed,
                self._metrics.events_failed,
                self._metrics.events_dead_lettered,
            )

    async def _process_event(self, event: Event) -> None:
        """Process a single event."""
        start_ms = int(time.time() * 1000)

        try:
            # Validate event
            result = self._validator.validate(event)
            if not result.valid:
                self._metrics.events_failed += 1
                logger.warning(
                    "Event validation failed: %s - %s",
                    event.event_type.name, result.error
                )
                return

            # Check for shutdown event
            if event.event_type == EventType.SHUTDOWN:
                self._handlers.handle(event)
                self._shutdown_event.set()
                return

            # Apply state transition atomically
            with self._state.transaction():
                self._state.reject_if_out_of_order(event)
                self._handlers.handle(event)
                self._state.record_event_processed()

            # Success
            self._metrics.events_processed += 1
            self._metrics.last_event_ms = start_ms
            self._clear_retry_count(event)

            # Update processing time stats
            processing_ms = int(time.time() * 1000) - start_ms
            self._update_processing_stats(processing_ms)

        except OutOfOrderEventError as e:
            self._metrics.events_failed += 1
            logger.warning("Out of order event: %s", e)

        except InvalidTransitionError as e:
            self._metrics.events_failed += 1
            self._handle_retry_or_dead_letter(event, str(e))

        except KeyError as e:
            self._metrics.events_failed += 1
            self._handle_retry_or_dead_letter(event, f"Missing entity: {e}")

        except StateError as e:
            self._metrics.events_failed += 1
            self._metrics.processing_errors += 1
            self._handle_retry_or_dead_letter(event, str(e))

        except Exception as e:
            self._metrics.events_failed += 1
            self._metrics.processing_errors += 1
            logger.error("Unexpected error processing event: %s", e, exc_info=True)
            self._handle_retry_or_dead_letter(event, str(e))

    def _handle_retry_or_dead_letter(self, event: Event, error: str) -> None:
        """Handle failed event: retry or dead-letter."""
        fingerprint = self._event_fingerprint(event)
        retry_count = self._retry_counts.get(fingerprint, 0) + 1
        self._retry_counts[fingerprint] = retry_count

        if retry_count >= self._max_retries:
            # Max retries exceeded, dead-letter
            self._dead_letter.add(event, error, retry_count)
            self._metrics.events_dead_lettered += 1
            del self._retry_counts[fingerprint]
        else:
            # Re-queue for retry
            self._queue.enqueue(event)
            logger.info(
                "Event retry %d/%d: %s",
                retry_count, self._max_retries, event.event_type.name
            )

    def _clear_retry_count(self, event: Event) -> None:
        """Clear retry count after successful processing."""
        fingerprint = self._event_fingerprint(event)
        self._retry_counts.pop(fingerprint, None)

    def _event_fingerprint(self, event: Event) -> str:
        """Create unique fingerprint for retry tracking."""
        parts = [
            event.event_type.name,
            str(event.sequence_num),
            event.venue or "internal",
        ]
        if event.order_id:
            parts.append(event.order_id)
        if event.attempt_id:
            parts.append(event.attempt_id)
        return ":".join(parts)

    def _update_processing_stats(self, processing_ms: int) -> None:
        """Update processing time statistics."""
        # Running average
        n = self._metrics.events_processed
        if n == 1:
            self._metrics.avg_processing_ms = float(processing_ms)
        else:
            self._metrics.avg_processing_ms = (
                self._metrics.avg_processing_ms * (n - 1) + processing_ms
            ) / n

        # Max
        if processing_ms > self._metrics.max_processing_ms:
            self._metrics.max_processing_ms = float(processing_ms)

    def _sample_metrics(self) -> None:
        """Sample current metrics."""
        self._metrics.queue_depth_samples.append(self._queue.size)
        # Keep only last 100 samples
        if len(self._metrics.queue_depth_samples) > 100:
            self._metrics.queue_depth_samples.pop(0)

    def _update_uptime(self) -> None:
        """Update uptime metric."""
        if self._started_at_ms:
            self._metrics.uptime_ms = int(time.time() * 1000) - self._started_at_ms

    async def shutdown(self, timeout_sec: float = 5.0) -> bool:
        """Request graceful shutdown.

        Args:
            timeout_sec: Maximum time to wait for shutdown

        Returns:
            True if shutdown completed within timeout
        """
        logger.info("Shutdown requested")
        self._consumer_state = ConsumerState.STOPPING
        self._shutdown_event.set()

        # Wait for consumer to stop
        start = time.time()
        while self._consumer_state != ConsumerState.STOPPED:
            if time.time() - start > timeout_sec:
                logger.warning("Shutdown timeout after %.1fs", timeout_sec)
                return False
            await asyncio.sleep(0.1)

        return True

    def enqueue(self, event: Event) -> bool:
        """Enqueue an event for processing.

        Convenience method to add events to the queue.

        Args:
            event: Event to enqueue

        Returns:
            True if enqueued, False if dropped
        """
        return self._queue.enqueue(event)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_consumer(
    state: StateManager,
    validator: Optional[EventValidator] = None,
    queue: Optional[BoundedEventQueue] = None,
    command_queue: Optional[CommandQueue] = None,
    queue_size: int = 1000,
) -> EventConsumer:
    """Factory to create configured EventConsumer.

    Args:
        state: StateManager instance
        validator: EventValidator (created if None)
        queue: Event queue (created if None)
        command_queue: Optional CommandQueue for emitting commands
        queue_size: Queue size if creating new queue

    Returns:
        Configured EventConsumer
    """
    if validator is None:
        validator = EventValidator()
    if queue is None:
        queue = BoundedEventQueue(max_size=queue_size)

    return EventConsumer(state, validator, queue, command_queue)
