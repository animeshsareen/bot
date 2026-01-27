"""Unit tests for command_executor module.

Tests cover:
- CommandQueue: put/get, drop policies, staleness, metrics
- CommandExecutor: lifecycle, command handling, lock integration, event emission
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch

from src.command_executor import (
    CommandQueue,
    CommandExecutor,
    DropPolicy,
    CommandQueueMetrics,
    ExecutorMetrics,
    create_command_executor,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_COMMAND_STALE_MS,
)
from src.commands import (
    SubmitOrderCommand,
    CancelOrderCommand,
    FlattenCommand,
)
from src.locks import LockManager, LockTimeoutError
from src.types import Event, EventType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def lock_manager():
    """Create a LockManager for testing."""
    return LockManager(lock_timeout_sec=1.0, warn_threshold_sec=0.5)


@pytest.fixture
def event_emitter():
    """Create a mock event emitter."""
    return AsyncMock()


@pytest.fixture
def command_queue():
    """Create a CommandQueue with default settings."""
    return CommandQueue(max_size=10, drop_policy=DropPolicy.DROP_OLDEST)


@pytest.fixture
def command_executor(lock_manager, event_emitter):
    """Create a CommandExecutor for testing."""
    queue = CommandQueue(max_size=10)
    executor = CommandExecutor(
        command_queue=queue,
        lock_manager=lock_manager,
        event_emitter=event_emitter,
        stale_ms=DEFAULT_COMMAND_STALE_MS,
    )
    return executor


@pytest.fixture
def submit_order_cmd():
    """Create a sample SubmitOrderCommand."""
    return SubmitOrderCommand(
        attempt_id="A-123",
        venue="kalshi",
        market_id="TEST-MKT",
        side="buy",
        quantity=10,
        price=0.50,
        client_order_id="A-123-KALSHI-YES",
        order_id="ORD-001",
    )


@pytest.fixture
def cancel_order_cmd():
    """Create a sample CancelOrderCommand."""
    return CancelOrderCommand(
        order_id="ORD-001",
        venue="kalshi",
        venue_order_id="VENUE-123",
        reason="test_cancel",
    )


@pytest.fixture
def flatten_cmd():
    """Create a sample FlattenCommand."""
    return FlattenCommand(
        venue="poly",
        quantity=5,
        side="sell",
        market_id="TEST-MKT",
        attempt_id="A-123",
        reason="imbalance",
    )


# =============================================================================
# COMMAND QUEUE TESTS
# =============================================================================

class TestCommandQueue:
    """Tests for CommandQueue."""

    def test_initialization(self):
        """Test queue initialization with default and custom params."""
        # Default
        q = CommandQueue()
        assert q.max_size == DEFAULT_QUEUE_SIZE
        assert q.drop_policy == DropPolicy.DROP_OLDEST
        assert q.stale_ms == DEFAULT_COMMAND_STALE_MS
        assert q.empty
        assert not q.full
        assert q.qsize == 0

        # Custom
        q2 = CommandQueue(max_size=5, drop_policy=DropPolicy.DROP_NEWEST, stale_ms=500)
        assert q2.max_size == 5
        assert q2.drop_policy == DropPolicy.DROP_NEWEST
        assert q2.stale_ms == 500

    @pytest.mark.asyncio
    async def test_put_get_basic(self, submit_order_cmd):
        """Test basic put/get operations."""
        q = CommandQueue(max_size=10)

        # Put command
        success = await q.put(submit_order_cmd)
        assert success
        assert q.qsize == 1
        assert not q.empty
        assert q.metrics.commands_enqueued == 1

        # Get command
        cmd = await q.get()
        assert cmd == submit_order_cmd
        assert q.qsize == 0
        assert q.empty
        assert q.metrics.commands_dequeued == 1

    @pytest.mark.asyncio
    async def test_put_drop_oldest(self, submit_order_cmd):
        """Test DROP_OLDEST policy when queue is full."""
        q = CommandQueue(max_size=2, drop_policy=DropPolicy.DROP_OLDEST)

        cmd1 = SubmitOrderCommand(
            attempt_id="A-1", venue="kalshi", market_id="M1",
            side="buy", quantity=1, price=0.5, client_order_id="C1", order_id="O1"
        )
        cmd2 = SubmitOrderCommand(
            attempt_id="A-2", venue="kalshi", market_id="M2",
            side="buy", quantity=1, price=0.5, client_order_id="C2", order_id="O2"
        )
        cmd3 = SubmitOrderCommand(
            attempt_id="A-3", venue="kalshi", market_id="M3",
            side="buy", quantity=1, price=0.5, client_order_id="C3", order_id="O3"
        )

        # Fill queue
        await q.put(cmd1)
        await q.put(cmd2)
        assert q.full
        assert q.qsize == 2

        # Add third - should drop oldest (cmd1)
        success = await q.put(cmd3)
        assert success
        assert q.qsize == 2
        assert q.metrics.commands_dropped == 1

        # Get remaining commands - should be cmd2 and cmd3
        first = await q.get()
        second = await q.get()
        assert first == cmd2
        assert second == cmd3

    @pytest.mark.asyncio
    async def test_put_drop_newest(self):
        """Test DROP_NEWEST policy when queue is full."""
        q = CommandQueue(max_size=2, drop_policy=DropPolicy.DROP_NEWEST)

        cmd1 = SubmitOrderCommand(
            attempt_id="A-1", venue="kalshi", market_id="M1",
            side="buy", quantity=1, price=0.5, client_order_id="C1", order_id="O1"
        )
        cmd2 = SubmitOrderCommand(
            attempt_id="A-2", venue="kalshi", market_id="M2",
            side="buy", quantity=1, price=0.5, client_order_id="C2", order_id="O2"
        )
        cmd3 = SubmitOrderCommand(
            attempt_id="A-3", venue="kalshi", market_id="M3",
            side="buy", quantity=1, price=0.5, client_order_id="C3", order_id="O3"
        )

        # Fill queue
        await q.put(cmd1)
        await q.put(cmd2)
        assert q.full

        # Try to add third - should be rejected
        success = await q.put(cmd3)
        assert not success
        assert q.qsize == 2
        assert q.metrics.commands_dropped == 1

        # Get commands - should be original two
        first = await q.get()
        second = await q.get()
        assert first == cmd1
        assert second == cmd2

    @pytest.mark.asyncio
    async def test_stale_command_rejection(self):
        """Test that stale commands are rejected."""
        q = CommandQueue(max_size=10, stale_ms=50)

        # Create stale command
        stale_cmd = SubmitOrderCommand(
            attempt_id="A-1",
            venue="kalshi",
            market_id="M1",
            side="buy",
            quantity=1,
            price=0.5,
            client_order_id="C1",
            order_id="O1",
            created_at_ms=int(time.time() * 1000) - 100,  # 100ms ago
        )

        # Should be rejected as stale
        success = await q.put(stale_cmd)
        assert not success
        assert q.metrics.stale_commands == 1
        assert q.empty

    @pytest.mark.asyncio
    async def test_get_with_timeout(self):
        """Test get() with timeout."""
        q = CommandQueue(max_size=10)

        # Get from empty queue with timeout
        cmd = await q.get(timeout_sec=0.1)
        assert cmd is None

    @pytest.mark.asyncio
    async def test_get_nowait(self, submit_order_cmd):
        """Test get_nowait() method."""
        q = CommandQueue(max_size=10)

        # Empty queue
        cmd = q.get_nowait()
        assert cmd is None

        # Add command
        await q.put(submit_order_cmd)

        # Get without waiting
        cmd = q.get_nowait()
        assert cmd == submit_order_cmd
        assert q.empty

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing all pending commands."""
        q = CommandQueue(max_size=10)

        # Add multiple commands
        for i in range(5):
            cmd = SubmitOrderCommand(
                attempt_id=f"A-{i}",
                venue="kalshi",
                market_id=f"M{i}",
                side="buy",
                quantity=1,
                price=0.5,
                client_order_id=f"C{i}",
                order_id=f"O{i}",
            )
            await q.put(cmd)

        assert q.qsize == 5

        # Clear
        cleared = q.clear()
        assert cleared == 5
        assert q.empty
        assert q.metrics.current_depth == 0

    @pytest.mark.asyncio
    async def test_task_done_and_join(self, submit_order_cmd):
        """Test task_done() and join() methods."""
        q = CommandQueue(max_size=10)

        # Add commands
        await q.put(submit_order_cmd)

        # Create task that will consume and mark done
        async def consumer():
            cmd = await q.get()
            await asyncio.sleep(0.01)
            q.task_done()

        task = asyncio.create_task(consumer())

        # Join should complete after task_done()
        await asyncio.wait_for(q.join(), timeout=0.5)
        await task

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        q = CommandQueue(max_size=10)

        assert q.metrics.commands_enqueued == 0
        assert q.metrics.commands_dequeued == 0
        assert q.metrics.commands_dropped == 0
        assert q.metrics.stale_commands == 0
        assert q.metrics.current_depth == 0


# =============================================================================
# COMMAND EXECUTOR TESTS
# =============================================================================

class TestCommandExecutor:
    """Tests for CommandExecutor."""

    def test_initialization(self, lock_manager, event_emitter):
        """Test executor initialization."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(
            command_queue=queue,
            lock_manager=lock_manager,
            event_emitter=event_emitter,
            stale_ms=200,
        )

        assert executor.command_queue == queue
        assert executor.lock_manager == lock_manager
        assert executor.event_emitter == event_emitter
        assert executor.stale_ms == 200
        assert not executor.is_running
        assert executor.metrics.commands_processed == 0

    def test_client_configuration(self, command_executor):
        """Test setting venue client callbacks."""
        kalshi_submit = AsyncMock()
        kalshi_cancel = AsyncMock()
        poly_submit = AsyncMock()
        poly_cancel = AsyncMock()
        flatten = AsyncMock()

        command_executor.set_kalshi_client(kalshi_submit, kalshi_cancel)
        command_executor.set_poly_client(poly_submit, poly_cancel)
        command_executor.set_flatten_callback(flatten)

        assert command_executor._kalshi_submit == kalshi_submit
        assert command_executor._kalshi_cancel == kalshi_cancel
        assert command_executor._poly_submit == poly_submit
        assert command_executor._poly_cancel == poly_cancel
        assert command_executor._flatten == flatten

    @pytest.mark.asyncio
    async def test_lifecycle_start_stop(self, command_executor):
        """Test executor start/stop lifecycle."""
        # Start
        await command_executor.start()
        assert command_executor.is_running
        assert command_executor._task is not None

        # Stop
        await command_executor.stop(drain_timeout_sec=1.0)
        assert not command_executor.is_running

    @pytest.mark.asyncio
    async def test_start_already_running(self, command_executor):
        """Test that starting when already running is a no-op."""
        await command_executor.start()
        assert command_executor.is_running

        # Start again - should log warning
        await command_executor.start()
        assert command_executor.is_running

        await command_executor.stop()

    @pytest.mark.asyncio
    async def test_handle_submit_success(self, lock_manager, event_emitter, submit_order_cmd):
        """Test successful order submission."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock venue client
        kalshi_submit = AsyncMock(return_value=(True, "VENUE-123", None))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())

        # Process submit command
        await executor._handle_submit(submit_order_cmd)

        # Verify venue client was called
        kalshi_submit.assert_called_once_with(submit_order_cmd)

        # Verify events emitted (ORDER_SENT, ORDER_ACKED)
        assert event_emitter.call_count == 2
        sent_event = event_emitter.call_args_list[0][0][0]
        acked_event = event_emitter.call_args_list[1][0][0]

        assert sent_event.event_type == EventType.ORDER_SENT
        assert sent_event.order_id == submit_order_cmd.order_id
        assert acked_event.event_type == EventType.ORDER_ACKED
        assert acked_event.venue_order_id == "VENUE-123"

        # Verify metrics
        assert executor.metrics.submit_successes == 1
        assert executor.metrics.submit_failures == 0

    @pytest.mark.asyncio
    async def test_handle_submit_failure(self, lock_manager, event_emitter, submit_order_cmd):
        """Test failed order submission."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock venue client with failure
        kalshi_submit = AsyncMock(return_value=(False, None, "Insufficient balance"))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())

        # Process submit command
        await executor._handle_submit(submit_order_cmd)

        # Verify events emitted (ORDER_SENT, ORDER_REJECTED)
        assert event_emitter.call_count == 2
        sent_event = event_emitter.call_args_list[0][0][0]
        rejected_event = event_emitter.call_args_list[1][0][0]

        assert sent_event.event_type == EventType.ORDER_SENT
        assert rejected_event.event_type == EventType.ORDER_REJECTED
        assert rejected_event.error == "Insufficient balance"

        # Verify metrics
        assert executor.metrics.submit_successes == 0
        assert executor.metrics.submit_failures == 1

    @pytest.mark.asyncio
    async def test_handle_submit_unknown_venue(self, lock_manager, event_emitter):
        """Test submit with unknown venue."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Create command with invalid venue
        cmd = SubmitOrderCommand(
            attempt_id="A-123",
            venue="unknown_venue",
            market_id="M1",
            side="buy",
            quantity=1,
            price=0.5,
            client_order_id="C1",
            order_id="O1",
        )

        await executor._handle_submit(cmd)

        # Should emit ORDER_REJECTED
        assert event_emitter.call_count == 1
        rejected_event = event_emitter.call_args[0][0]
        assert rejected_event.event_type == EventType.ORDER_REJECTED
        assert "Unknown venue" in rejected_event.error

    @pytest.mark.asyncio
    async def test_handle_submit_no_client_configured(self, lock_manager, event_emitter, submit_order_cmd):
        """Test submit when no client callback is configured."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Don't configure any client callbacks
        await executor._handle_submit(submit_order_cmd)

        # Should emit ORDER_REJECTED
        assert event_emitter.call_count == 1
        rejected_event = event_emitter.call_args[0][0]
        assert rejected_event.event_type == EventType.ORDER_REJECTED
        assert "No client configured" in rejected_event.error

    @pytest.mark.asyncio
    async def test_handle_submit_lock_timeout(self, event_emitter, submit_order_cmd):
        """Test submit when lock acquisition times out."""
        # Use very short timeout
        lock_manager = LockManager(lock_timeout_sec=0.01)
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock slow venue client
        async def slow_submit(cmd):
            await asyncio.sleep(0.1)
            return (True, "VENUE-123", None)

        executor.set_kalshi_client(slow_submit, AsyncMock())

        # Start first submission (will hold lock)
        task1 = asyncio.create_task(executor._handle_submit(submit_order_cmd))

        # Wait a bit for lock to be acquired
        await asyncio.sleep(0.005)

        # Create second command for same venue+market
        cmd2 = SubmitOrderCommand(
            attempt_id="A-456",
            venue="kalshi",
            market_id="TEST-MKT",
            side="sell",
            quantity=5,
            price=0.55,
            client_order_id="A-456-KALSHI-NO",
            order_id="ORD-002",
        )

        # Try to submit - should timeout
        await executor._handle_submit(cmd2)

        # Verify metrics
        assert executor.metrics.lock_timeouts == 1

        # Clean up
        await task1

    @pytest.mark.asyncio
    async def test_handle_cancel_success(self, lock_manager, event_emitter, cancel_order_cmd):
        """Test successful order cancellation."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock venue client
        kalshi_cancel = AsyncMock(return_value=(True, None))
        executor.set_kalshi_client(AsyncMock(), kalshi_cancel)

        # Process cancel command
        await executor._handle_cancel(cancel_order_cmd)

        # Verify venue client was called
        kalshi_cancel.assert_called_once_with(cancel_order_cmd)

        # Verify ORDER_CANCELLED event emitted
        assert event_emitter.call_count == 1
        cancelled_event = event_emitter.call_args[0][0]
        assert cancelled_event.event_type == EventType.ORDER_CANCELLED
        assert cancelled_event.order_id == cancel_order_cmd.order_id

        # Verify metrics
        assert executor.metrics.cancel_successes == 1
        assert executor.metrics.cancel_failures == 0

    @pytest.mark.asyncio
    async def test_handle_cancel_failure(self, lock_manager, event_emitter, cancel_order_cmd):
        """Test failed order cancellation."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock venue client with failure
        kalshi_cancel = AsyncMock(return_value=(False, "Order not found"))
        executor.set_kalshi_client(AsyncMock(), kalshi_cancel)

        # Process cancel command
        await executor._handle_cancel(cancel_order_cmd)

        # Verify metrics
        assert executor.metrics.cancel_successes == 0
        assert executor.metrics.cancel_failures == 1

    @pytest.mark.asyncio
    async def test_handle_flatten_success(self, lock_manager, event_emitter, flatten_cmd):
        """Test successful flatten operation."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock flatten callback
        flatten_callback = AsyncMock(return_value=(True, None))
        executor.set_flatten_callback(flatten_callback)

        # Process flatten command
        await executor._handle_flatten(flatten_cmd)

        # Verify callback was called
        flatten_callback.assert_called_once_with(flatten_cmd)

        # Verify metrics
        assert executor.metrics.flatten_successes == 1
        assert executor.metrics.flatten_failures == 0

    @pytest.mark.asyncio
    async def test_handle_flatten_failure(self, lock_manager, event_emitter, flatten_cmd):
        """Test failed flatten operation."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Mock flatten callback with failure
        flatten_callback = AsyncMock(return_value=(False, "No position to flatten"))
        executor.set_flatten_callback(flatten_callback)

        # Process flatten command
        await executor._handle_flatten(flatten_cmd)

        # Verify metrics
        assert executor.metrics.flatten_successes == 0
        assert executor.metrics.flatten_failures == 1

    @pytest.mark.asyncio
    async def test_handle_flatten_no_callback(self, lock_manager, event_emitter, flatten_cmd):
        """Test flatten when no callback is configured."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Don't configure flatten callback
        await executor._handle_flatten(flatten_cmd)

        # Verify failure metric incremented
        assert executor.metrics.flatten_failures == 1

    @pytest.mark.asyncio
    async def test_process_stale_command(self, lock_manager, event_emitter):
        """Test that stale commands are skipped."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter, stale_ms=50)

        # Create stale command
        stale_cmd = SubmitOrderCommand(
            attempt_id="A-1",
            venue="kalshi",
            market_id="M1",
            side="buy",
            quantity=1,
            price=0.5,
            client_order_id="C1",
            order_id="O1",
            created_at_ms=int(time.time() * 1000) - 100,  # 100ms ago
        )

        # Process stale command
        await executor._process_command(stale_cmd)

        # Should not emit any events
        event_emitter.assert_not_called()

        # Should increment failure metric
        assert executor.metrics.commands_failed == 1

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_commands(self, lock_manager, event_emitter, submit_order_cmd):
        """Test full executor lifecycle with command processing."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Configure venue client
        kalshi_submit = AsyncMock(return_value=(True, "VENUE-123", None))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())

        # Start executor
        await executor.start()

        # Add command to queue
        await queue.put(submit_order_cmd)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop executor
        await executor.stop(drain_timeout_sec=1.0)

        # Verify command was processed
        assert executor.metrics.commands_processed == 1
        kalshi_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, command_executor):
        """Test that executor metrics are properly tracked."""
        metrics = command_executor.metrics

        assert metrics.commands_processed == 0
        assert metrics.commands_failed == 0
        assert metrics.lock_timeouts == 0
        assert metrics.submit_successes == 0
        assert metrics.submit_failures == 0
        assert metrics.cancel_successes == 0
        assert metrics.cancel_failures == 0
        assert metrics.flatten_successes == 0
        assert metrics.flatten_failures == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.max_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_sequence_counter(self, lock_manager, event_emitter, submit_order_cmd):
        """Test that sequence counter increments for each event."""
        queue = CommandQueue(max_size=10)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Configure venue client
        kalshi_submit = AsyncMock(return_value=(True, "VENUE-123", None))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())

        # Process command
        await executor._handle_submit(submit_order_cmd)

        # Check events have sequential sequence numbers
        assert event_emitter.call_count == 2
        sent_event = event_emitter.call_args_list[0][0][0]
        acked_event = event_emitter.call_args_list[1][0][0]

        assert sent_event.sequence_num == 1
        assert acked_event.sequence_num == 2


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_command_executor(self, lock_manager, event_emitter):
        """Test create_command_executor factory."""
        queue, executor = create_command_executor(
            lock_manager=lock_manager,
            event_emitter=event_emitter,
            queue_size=50,
            drop_policy=DropPolicy.DROP_NEWEST,
            stale_ms=500,
        )

        assert isinstance(queue, CommandQueue)
        assert isinstance(executor, CommandExecutor)
        assert queue.max_size == 50
        assert queue.drop_policy == DropPolicy.DROP_NEWEST
        assert queue.stale_ms == 500
        assert executor.command_queue == queue
        assert executor.lock_manager == lock_manager
        assert executor.event_emitter == event_emitter
        assert executor.stale_ms == 500


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for command queue and executor."""

    @pytest.mark.asyncio
    async def test_end_to_end_submit_flow(self, lock_manager, event_emitter):
        """Test complete flow: enqueue -> process -> emit events."""
        queue, executor = create_command_executor(
            lock_manager=lock_manager,
            event_emitter=event_emitter,
        )

        # Configure venue client
        kalshi_submit = AsyncMock(return_value=(True, "VENUE-123", None))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())

        # Create command
        cmd = SubmitOrderCommand(
            attempt_id="A-123",
            venue="kalshi",
            market_id="TEST-MKT",
            side="buy",
            quantity=10,
            price=0.50,
            client_order_id="A-123-KALSHI-YES",
            order_id="ORD-001",
        )

        # Start executor
        await executor.start()

        # Enqueue command
        success = await queue.put(cmd)
        assert success

        # Wait for processing
        await asyncio.sleep(0.2)

        # Stop executor
        await executor.stop()

        # Verify processing
        assert executor.metrics.commands_processed == 1
        assert executor.metrics.submit_successes == 1
        kalshi_submit.assert_called_once_with(cmd)

        # Verify events
        assert event_emitter.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_commands_different_venues(self, lock_manager, event_emitter):
        """Test processing commands for different venues in parallel."""
        queue, executor = create_command_executor(
            lock_manager=lock_manager,
            event_emitter=event_emitter,
        )

        # Configure both venue clients
        kalshi_submit = AsyncMock(return_value=(True, "KALSHI-123", None))
        poly_submit = AsyncMock(return_value=(True, "POLY-456", None))
        executor.set_kalshi_client(kalshi_submit, AsyncMock())
        executor.set_poly_client(poly_submit, AsyncMock())

        # Create commands for different venues
        cmd_kalshi = SubmitOrderCommand(
            attempt_id="A-1", venue="kalshi", market_id="M1",
            side="buy", quantity=10, price=0.50,
            client_order_id="C1", order_id="O1",
        )
        cmd_poly = SubmitOrderCommand(
            attempt_id="A-2", venue="poly", market_id="M2",
            side="sell", quantity=5, price=0.60,
            client_order_id="C2", order_id="O2",
        )

        # Start executor
        await executor.start()

        # Enqueue both commands
        await queue.put(cmd_kalshi)
        await queue.put(cmd_poly)

        # Wait for processing
        await asyncio.sleep(0.3)

        # Stop executor
        await executor.stop()

        # Verify both processed
        assert executor.metrics.commands_processed == 2
        assert executor.metrics.submit_successes == 2
        kalshi_submit.assert_called_once()
        poly_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_queue_overflow_with_drop_policy(self, lock_manager, event_emitter):
        """Test queue behavior under overflow with drop policy."""
        queue = CommandQueue(max_size=3, drop_policy=DropPolicy.DROP_OLDEST)
        executor = CommandExecutor(queue, lock_manager, event_emitter)

        # Configure slow venue client to create backlog
        async def slow_submit(cmd):
            await asyncio.sleep(0.5)
            return (True, "VENUE-123", None)

        executor.set_kalshi_client(slow_submit, AsyncMock())

        # Start executor
        await executor.start()

        # Enqueue many commands quickly
        for i in range(6):
            cmd = SubmitOrderCommand(
                attempt_id=f"A-{i}",
                venue="kalshi",
                market_id=f"M{i}",
                side="buy",
                quantity=1,
                price=0.5,
                client_order_id=f"C{i}",
                order_id=f"O{i}",
            )
            await queue.put(cmd)

        # Should have dropped some
        assert queue.metrics.commands_dropped > 0

        # Stop executor
        await executor.stop(drain_timeout_sec=3.0)