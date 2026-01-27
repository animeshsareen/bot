"""Command queue and executor for order submission.

Implements the command queue pattern that separates:
- Events: "what happened" (ORDER_FILL, ORDER_ACKED, etc.)
- Commands: "what to do" (SubmitOrder, CancelOrder, Flatten)

The StateManager/EventConsumer emits commands after processing events.
The CommandExecutor consumes commands, acquires locks, and submits to venues.

Key design principles:
- CommandQueue provides backpressure (bounded queue)
- CommandExecutor never mutates state directly
- Executor emits events back to EventQueue after REST calls
- Locks acquired before submission, held until terminal or timeout
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Awaitable
from enum import Enum, auto

from .commands import (
    Command,
    CommandType,
    SubmitOrderCommand,
    CancelOrderCommand,
    FlattenCommand,
)
from .locks import LockManager, LockTimeoutError
from .types import Event, EventType


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_QUEUE_SIZE = 100
DEFAULT_COMMAND_STALE_MS = 250
DEFAULT_SHUTDOWN_DRAIN_SEC = 5.0


class DropPolicy(Enum):
    """Policy for handling full queue."""
    DROP_OLDEST = auto()  # Drop oldest command when full
    DROP_NEWEST = auto()  # Reject new command when full
    BLOCK = auto()        # Block until space available


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class CommandQueueMetrics:
    """Metrics for command queue operations.

    Attributes:
        commands_enqueued: Total commands added to queue
        commands_dropped: Commands dropped due to full queue
        commands_dequeued: Commands successfully dequeued
        stale_commands: Commands rejected as stale
        current_depth: Current queue depth
    """
    commands_enqueued: int = 0
    commands_dropped: int = 0
    commands_dequeued: int = 0
    stale_commands: int = 0
    current_depth: int = 0


@dataclass
class ExecutorMetrics:
    """Metrics for command executor.

    Attributes:
        commands_processed: Successfully processed commands
        commands_failed: Commands that failed execution
        lock_timeouts: Lock acquisition timeouts
        avg_latency_ms: Average command processing latency
        max_latency_ms: Maximum command processing latency
    """
    commands_processed: int = 0
    commands_failed: int = 0
    lock_timeouts: int = 0
    submit_successes: int = 0
    submit_failures: int = 0
    cancel_successes: int = 0
    cancel_failures: int = 0
    flatten_successes: int = 0
    flatten_failures: int = 0
    _latency_sum_ms: float = 0.0
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.commands_processed == 0:
            return 0.0
        return self._latency_sum_ms / self.commands_processed


# =============================================================================
# COMMAND QUEUE
# =============================================================================

class CommandQueue:
    """Thread-safe bounded queue for commands with backpressure.

    Provides:
    - Configurable max size
    - Drop policy when full
    - Stale command rejection
    - Metrics tracking

    Usage:
        queue = CommandQueue(max_size=100)
        await queue.put(SubmitOrderCommand(...))
        cmd = await queue.get()
    """

    def __init__(
        self,
        max_size: int = DEFAULT_QUEUE_SIZE,
        drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
        stale_ms: int = DEFAULT_COMMAND_STALE_MS,
    ):
        """Initialize command queue.

        Args:
            max_size: Maximum queue size
            drop_policy: What to do when queue is full
            stale_ms: Commands older than this are rejected
        """
        self.max_size = max_size
        self.drop_policy = drop_policy
        self.stale_ms = stale_ms

        self._queue: asyncio.Queue[Command] = asyncio.Queue(maxsize=max_size)
        self._lock = asyncio.Lock()
        self.metrics = CommandQueueMetrics()

    @property
    def qsize(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """True if queue is empty."""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """True if queue is at max capacity."""
        return self._queue.full()

    async def put(self, command: Command) -> bool:
        """Add command to queue.

        Args:
            command: Command to enqueue

        Returns:
            True if command was enqueued, False if dropped
        """
        # Check staleness
        if hasattr(command, 'is_stale') and command.is_stale(self.stale_ms):
            self.metrics.stale_commands += 1
            logger.warning(
                "Dropping stale command: %s (age > %dms)",
                command.command_type.name, self.stale_ms
            )
            return False

        async with self._lock:
            if self._queue.full():
                if self.drop_policy == DropPolicy.DROP_OLDEST:
                    # Remove oldest
                    try:
                        dropped = self._queue.get_nowait()
                        self.metrics.commands_dropped += 1
                        logger.warning(
                            "Queue full, dropped oldest: %s",
                            dropped.command_type.name
                        )
                    except asyncio.QueueEmpty:
                        pass
                elif self.drop_policy == DropPolicy.DROP_NEWEST:
                    self.metrics.commands_dropped += 1
                    logger.warning(
                        "Queue full, dropping new: %s",
                        command.command_type.name
                    )
                    return False
                # BLOCK policy falls through to put()

            try:
                self._queue.put_nowait(command)
                self.metrics.commands_enqueued += 1
                self.metrics.current_depth = self._queue.qsize()
                logger.debug(
                    "Enqueued command: %s (depth=%d)",
                    command.command_type.name, self.metrics.current_depth
                )
                return True
            except asyncio.QueueFull:
                # Shouldn't happen with DROP_OLDEST, but handle anyway
                self.metrics.commands_dropped += 1
                return False

    async def get(self, timeout_sec: Optional[float] = None) -> Optional[Command]:
        """Get next command from queue.

        Args:
            timeout_sec: Max time to wait (None = wait forever)

        Returns:
            Command if available, None if timeout
        """
        try:
            if timeout_sec is not None:
                command = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout_sec
                )
            else:
                command = await self._queue.get()

            self.metrics.commands_dequeued += 1
            self.metrics.current_depth = self._queue.qsize()
            return command

        except asyncio.TimeoutError:
            return None

    def get_nowait(self) -> Optional[Command]:
        """Get command without waiting.

        Returns:
            Command if available, None if empty
        """
        try:
            command = self._queue.get_nowait()
            self.metrics.commands_dequeued += 1
            self.metrics.current_depth = self._queue.qsize()
            return command
        except asyncio.QueueEmpty:
            return None

    def task_done(self) -> None:
        """Mark current task as done (for join())."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait until all commands are processed."""
        await self._queue.join()

    def clear(self) -> int:
        """Clear all pending commands.

        Returns:
            Number of commands cleared
        """
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        self.metrics.current_depth = 0
        return count


# =============================================================================
# VENUE CLIENT PROTOCOL
# =============================================================================

# Type aliases for venue client callbacks
SubmitCallback = Callable[[SubmitOrderCommand], Awaitable[tuple[bool, Optional[str], Optional[str]]]]
"""Returns (success, venue_order_id, error_message)"""

CancelCallback = Callable[[CancelOrderCommand], Awaitable[tuple[bool, Optional[str]]]]
"""Returns (success, error_message)"""

FlattenCallback = Callable[[FlattenCommand], Awaitable[tuple[bool, Optional[str]]]]
"""Returns (success, error_message)"""

EventEmitter = Callable[[Event], Awaitable[None]]
"""Callback to emit events back to event queue"""


# =============================================================================
# COMMAND EXECUTOR
# =============================================================================

class CommandExecutor:
    """Consumes commands from queue and executes against venues.

    The executor:
    1. Pulls commands from CommandQueue
    2. Acquires appropriate locks from LockManager
    3. Submits to venues via REST clients
    4. Emits events back to EventQueue

    Never mutates StateManager directly - only emits events.

    Usage:
        executor = CommandExecutor(
            command_queue=queue,
            lock_manager=locks,
            event_emitter=emit_event,
        )
        executor.set_kalshi_client(submit_fn, cancel_fn)
        executor.set_poly_client(submit_fn, cancel_fn)
        await executor.start()
    """

    def __init__(
        self,
        command_queue: CommandQueue,
        lock_manager: LockManager,
        event_emitter: EventEmitter,
        stale_ms: int = DEFAULT_COMMAND_STALE_MS,
    ):
        """Initialize command executor.

        Args:
            command_queue: Queue to consume commands from
            lock_manager: Lock manager for order/attempt locks
            event_emitter: Callback to emit events
            stale_ms: Reject commands older than this
        """
        self.command_queue = command_queue
        self.lock_manager = lock_manager
        self.event_emitter = event_emitter
        self.stale_ms = stale_ms

        # Venue clients (set via setters)
        self._kalshi_submit: Optional[SubmitCallback] = None
        self._kalshi_cancel: Optional[CancelCallback] = None
        self._poly_submit: Optional[SubmitCallback] = None
        self._poly_cancel: Optional[CancelCallback] = None
        self._flatten: Optional[FlattenCallback] = None

        # Executor state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._sequence_counter = 0

        self.metrics = ExecutorMetrics()

    # =========================================================================
    # CLIENT CONFIGURATION
    # =========================================================================

    def set_kalshi_client(
        self,
        submit: SubmitCallback,
        cancel: CancelCallback,
    ) -> None:
        """Configure Kalshi client callbacks."""
        self._kalshi_submit = submit
        self._kalshi_cancel = cancel

    def set_poly_client(
        self,
        submit: SubmitCallback,
        cancel: CancelCallback,
    ) -> None:
        """Configure Polymarket client callbacks."""
        self._poly_submit = submit
        self._poly_cancel = cancel

    def set_flatten_callback(self, flatten: FlattenCallback) -> None:
        """Configure flatten callback."""
        self._flatten = flatten

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """Start the executor loop."""
        if self._running:
            logger.warning("CommandExecutor already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="command_executor")
        logger.info("CommandExecutor started")

    async def stop(self, drain_timeout_sec: float = DEFAULT_SHUTDOWN_DRAIN_SEC) -> None:
        """Stop the executor gracefully.

        Args:
            drain_timeout_sec: Time to wait for queue to drain
        """
        if not self._running:
            return

        logger.info("CommandExecutor stopping...")
        self._shutdown_event.set()

        # Wait for queue to drain
        if not self.command_queue.empty:
            logger.info("Draining command queue...")
            try:
                await asyncio.wait_for(
                    self.command_queue.join(),
                    timeout=drain_timeout_sec
                )
            except asyncio.TimeoutError:
                remaining = self.command_queue.qsize
                logger.warning(
                    "Queue drain timeout, %d commands remaining", remaining
                )

        # Cancel the task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._running = False
        logger.info("CommandExecutor stopped")

    @property
    def is_running(self) -> bool:
        """True if executor is running."""
        return self._running

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def _run_loop(self) -> None:
        """Main executor loop."""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next command with timeout (for shutdown check)
                command = await self.command_queue.get(timeout_sec=0.1)
                if command is None:
                    continue

                # Process command
                start_ms = int(time.time() * 1000)
                try:
                    await self._process_command(command)
                    self.metrics.commands_processed += 1
                except Exception as e:
                    logger.error("Command processing failed: %s", e)
                    self.metrics.commands_failed += 1
                finally:
                    self.command_queue.task_done()
                    latency_ms = int(time.time() * 1000) - start_ms
                    self.metrics._latency_sum_ms += latency_ms
                    self.metrics.max_latency_ms = max(
                        self.metrics.max_latency_ms, latency_ms
                    )

            except asyncio.CancelledError:
                logger.info("Executor loop cancelled")
                break
            except Exception as e:
                logger.exception("Unexpected error in executor loop: %s", e)
                await asyncio.sleep(0.1)  # Avoid tight loop on error

    async def _process_command(self, command: Command) -> None:
        """Process a single command."""
        # Check staleness
        if hasattr(command, 'is_stale') and command.is_stale(self.stale_ms):
            logger.warning(
                "Skipping stale command: %s", command.command_type.name
            )
            self.metrics.commands_failed += 1
            return

        if isinstance(command, SubmitOrderCommand):
            await self._handle_submit(command)
        elif isinstance(command, CancelOrderCommand):
            await self._handle_cancel(command)
        elif isinstance(command, FlattenCommand):
            await self._handle_flatten(command)
        else:
            logger.error("Unknown command type: %s", type(command))

    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================

    async def _handle_submit(self, cmd: SubmitOrderCommand) -> None:
        """Handle order submission command."""
        logger.info(
            "Processing submit: %s %s %d @ %.4f on %s",
            cmd.side, cmd.market_id, cmd.quantity, cmd.price, cmd.venue
        )

        # Get the appropriate submit callback
        if cmd.venue == "kalshi":
            submit_fn = self._kalshi_submit
        elif cmd.venue == "poly":
            submit_fn = self._poly_submit
        else:
            logger.error("Unknown venue: %s", cmd.venue)
            await self._emit_order_rejected(cmd, f"Unknown venue: {cmd.venue}")
            return

        if submit_fn is None:
            logger.error("No submit callback configured for %s", cmd.venue)
            await self._emit_order_rejected(cmd, "No client configured")
            return

        # Acquire venue+market lock before submission
        try:
            async with self.lock_manager.acquire_order_lock(
                venue=cmd.venue,
                market_id=cmd.market_id,
                holder=cmd.client_order_id,
            ):
                # Emit ORDER_SENT event
                await self._emit_order_sent(cmd)

                # Submit to venue
                success, venue_order_id, error = await submit_fn(cmd)

                if success:
                    # Emit ORDER_ACKED event
                    await self._emit_order_acked(cmd, venue_order_id)
                    self.metrics.submit_successes += 1
                else:
                    # Emit ORDER_REJECTED event
                    await self._emit_order_rejected(cmd, error)
                    self.metrics.submit_failures += 1

        except LockTimeoutError as e:
            logger.error("Lock timeout for submit: %s", e)
            self.metrics.lock_timeouts += 1
            await self._emit_order_rejected(cmd, f"Lock timeout: {e}")

    async def _handle_cancel(self, cmd: CancelOrderCommand) -> None:
        """Handle order cancellation command."""
        logger.info("Processing cancel: %s on %s", cmd.order_id, cmd.venue)

        # Get the appropriate cancel callback
        if cmd.venue == "kalshi":
            cancel_fn = self._kalshi_cancel
        elif cmd.venue == "poly":
            cancel_fn = self._poly_cancel
        else:
            logger.error("Unknown venue: %s", cmd.venue)
            return

        if cancel_fn is None:
            logger.error("No cancel callback configured for %s", cmd.venue)
            return

        # Submit cancellation (no lock needed - venue handles idempotency)
        success, error = await cancel_fn(cmd)

        if success:
            await self._emit_order_cancelled(cmd)
            self.metrics.cancel_successes += 1
        else:
            logger.warning("Cancel failed for %s: %s", cmd.order_id, error)
            self.metrics.cancel_failures += 1

    async def _handle_flatten(self, cmd: FlattenCommand) -> None:
        """Handle flatten command."""
        logger.info(
            "Processing flatten: %s %d on %s (%s)",
            cmd.side, cmd.quantity, cmd.venue, cmd.reason
        )

        if self._flatten is None:
            logger.error("No flatten callback configured")
            self.metrics.flatten_failures += 1
            return

        success, error = await self._flatten(cmd)

        if success:
            logger.info("Flatten succeeded for %s", cmd.venue)
            self.metrics.flatten_successes += 1
        else:
            logger.error("Flatten failed for %s: %s", cmd.venue, error)
            self.metrics.flatten_failures += 1

    # =========================================================================
    # EVENT EMISSION
    # =========================================================================

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence_counter += 1
        return self._sequence_counter

    async def _emit_order_sent(self, cmd: SubmitOrderCommand) -> None:
        """Emit ORDER_SENT event."""
        event = Event(
            event_type=EventType.ORDER_SENT,
            sequence_num=self._next_sequence(),
            venue=cmd.venue,
            order_id=cmd.order_id,
            attempt_id=cmd.attempt_id,
        )
        await self.event_emitter(event)

    async def _emit_order_acked(
        self, cmd: SubmitOrderCommand, venue_order_id: Optional[str]
    ) -> None:
        """Emit ORDER_ACKED event."""
        event = Event(
            event_type=EventType.ORDER_ACKED,
            sequence_num=self._next_sequence(),
            venue=cmd.venue,
            order_id=cmd.order_id,
            venue_order_id=venue_order_id,
            attempt_id=cmd.attempt_id,
        )
        await self.event_emitter(event)

    async def _emit_order_rejected(
        self, cmd: SubmitOrderCommand, error: Optional[str]
    ) -> None:
        """Emit ORDER_REJECTED event."""
        event = Event(
            event_type=EventType.ORDER_REJECTED,
            sequence_num=self._next_sequence(),
            venue=cmd.venue,
            order_id=cmd.order_id,
            error=error,
            attempt_id=cmd.attempt_id,
        )
        await self.event_emitter(event)

    async def _emit_order_cancelled(self, cmd: CancelOrderCommand) -> None:
        """Emit ORDER_CANCELLED event."""
        event = Event(
            event_type=EventType.ORDER_CANCELLED,
            sequence_num=self._next_sequence(),
            venue=cmd.venue,
            order_id=cmd.order_id,
        )
        await self.event_emitter(event)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_command_executor(
    lock_manager: LockManager,
    event_emitter: EventEmitter,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    drop_policy: DropPolicy = DropPolicy.DROP_OLDEST,
    stale_ms: int = DEFAULT_COMMAND_STALE_MS,
) -> tuple[CommandQueue, CommandExecutor]:
    """Factory to create CommandQueue and CommandExecutor pair.

    Args:
        lock_manager: Lock manager for coordination
        event_emitter: Callback to emit events
        queue_size: Maximum queue size
        drop_policy: Policy when queue is full
        stale_ms: Staleness threshold

    Returns:
        Tuple of (CommandQueue, CommandExecutor)
    """
    queue = CommandQueue(
        max_size=queue_size,
        drop_policy=drop_policy,
        stale_ms=stale_ms,
    )

    executor = CommandExecutor(
        command_queue=queue,
        lock_manager=lock_manager,
        event_emitter=event_emitter,
        stale_ms=stale_ms,
    )

    return queue, executor
