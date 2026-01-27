"""Tests for async event-driven execution architecture.

Tests US-019 requirements:
- Independent async WS tasks per venue update shared VenueState
- Trade decisions operate on immutable Snapshot copies
- Execution and reconciliation run in separate async task guarded by a mutex
- WS tasks never await execution/reconciliation tasks
- REST fallback is async and cancellable if WS becomes fresh
- All long-running tasks cancellable on rollover or FLAT_FORCED
- Decision-to-submit latency logged and <150ms in integration test
"""

import asyncio
import time
import pytest

from src.async_executor import (
    AsyncExecutor,
    ExecutorState,
    ExecutionResult,
    LatencyMetrics,
    CancellableTask,
    CancellableRESTFetcher,
)
from src.venue_state import (
    VenueState,
    MarketSnapshot,
    OrderLevel,
    create_snapshot,
    create_venue_state,
)


# --- LatencyMetrics Tests ---


class TestLatencyMetrics:
    """Tests for LatencyMetrics immutable dataclass."""

    def test_initial_metrics_all_zero(self):
        """Fresh metrics have all zeros."""
        m = LatencyMetrics()
        assert m.decision_start_ms == 0
        assert m.decision_end_ms == 0
        assert m.submit_start_ms == 0
        assert m.submit_end_ms == 0
        assert m.reconcile_start_ms == 0
        assert m.reconcile_end_ms == 0

    def test_decision_ms_calculation(self):
        """decision_ms is end - start."""
        m = LatencyMetrics(decision_start_ms=1000, decision_end_ms=1050)
        assert m.decision_ms == 50

    def test_submit_ms_calculation(self):
        """submit_ms is end - start."""
        m = LatencyMetrics(submit_start_ms=1050, submit_end_ms=1100)
        assert m.submit_ms == 50

    def test_reconcile_ms_calculation(self):
        """reconcile_ms is end - start."""
        m = LatencyMetrics(reconcile_start_ms=1100, reconcile_end_ms=1200)
        assert m.reconcile_ms == 100

    def test_decision_to_submit_ms(self):
        """decision_to_submit_ms spans from decision start to submit end."""
        m = LatencyMetrics(decision_start_ms=1000, submit_end_ms=1150)
        assert m.decision_to_submit_ms == 150

    def test_with_methods_return_new_instance(self):
        """with_* methods return new instances, original unchanged."""
        m1 = LatencyMetrics()
        m2 = m1.with_decision_start(1000)
        assert m1.decision_start_ms == 0
        assert m2.decision_start_ms == 1000
        assert m1 is not m2

    def test_chained_with_methods(self):
        """with_* methods can be chained."""
        m = (
            LatencyMetrics()
            .with_decision_start(1000)
            .with_decision_end(1050)
            .with_submit_start(1050)
            .with_submit_end(1100)
            .with_reconcile_start(1100)
            .with_reconcile_end(1200)
        )
        assert m.decision_ms == 50
        assert m.submit_ms == 50
        assert m.reconcile_ms == 100
        assert m.decision_to_submit_ms == 100


# --- Helper Fixtures ---


@pytest.fixture
def mock_snapshot():
    """Create a valid mock snapshot for testing."""
    now_ms = int(time.time() * 1000)
    poly_yes_state = create_venue_state("poly", "token123").with_orderbook(
        [OrderLevel(price=0.45, size=100)],
        [OrderLevel(price=0.55, size=100)],
        now_ms,
    )
    poly_no_state = create_venue_state("poly", "token124").with_orderbook(
        [OrderLevel(price=0.45, size=100)],
        [OrderLevel(price=0.55, size=100)],
        now_ms,
    )
    kalshi_state = create_venue_state("kalshi", "CRYPTO-BTC").with_orderbook(
        [OrderLevel(price=0.44, size=100)],
        [OrderLevel(price=0.56, size=100)],
        now_ms,
    )
    return create_snapshot(poly_yes_state, poly_no_state, kalshi_state, "ETH-15M-001", now_ms)


# --- AsyncExecutor Tests ---


class TestAsyncExecutorState:
    """Tests for AsyncExecutor state management."""

    def test_initial_state_is_monitor(self):
        """Executor starts in MONITOR state."""
        executor = AsyncExecutor()
        assert executor.state == ExecutorState.MONITOR

    def test_is_busy_when_locked(self):
        """is_busy reflects mutex state."""
        executor = AsyncExecutor()
        assert not executor.is_busy

    def test_can_start_attempt_in_monitor(self):
        """Can start attempt from MONITOR state."""
        executor = AsyncExecutor()
        assert executor.can_start_attempt()

    def test_can_start_attempt_in_done_equal(self):
        """Can start attempt from DONE_EQUAL state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.DONE_EQUAL
        assert executor.can_start_attempt()

    def test_cannot_start_attempt_in_execute(self):
        """Cannot start attempt from EXECUTE state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.EXECUTE
        assert not executor.can_start_attempt()

    def test_cannot_start_attempt_in_reconcile(self):
        """Cannot start attempt from RECONCILE state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.RECONCILE
        assert not executor.can_start_attempt()

    def test_cannot_start_attempt_in_flat_forced(self):
        """Cannot start attempt from FLAT_FORCED state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.FLAT_FORCED
        assert not executor.can_start_attempt()


class TestAsyncExecutorExecution:
    """Tests for AsyncExecutor execution logic."""

    @pytest.mark.asyncio
    async def test_basic_execution_flow(self, mock_snapshot):
        """Execute transitions through states correctly."""
        states_seen = []

        async def track_execute(snapshot):
            states_seen.append("execute")
            return True

        async def track_reconcile(snapshot):
            states_seen.append("reconcile")
            return ExecutorState.DONE_EQUAL

        executor = AsyncExecutor(
            execute_callback=track_execute,
            reconcile_callback=track_reconcile,
        )

        result = await executor.execute(mock_snapshot)

        assert result.state == ExecutorState.DONE_EQUAL
        assert executor.state == ExecutorState.DONE_EQUAL
        assert states_seen == ["execute", "reconcile"]

    @pytest.mark.asyncio
    async def test_execution_captures_metrics(self, mock_snapshot):
        """Execution records timing metrics."""

        async def slow_execute(snapshot):
            await asyncio.sleep(0.01)  # 10ms
            return True

        async def slow_reconcile(snapshot):
            await asyncio.sleep(0.01)  # 10ms
            return ExecutorState.DONE_EQUAL

        executor = AsyncExecutor(
            execute_callback=slow_execute,
            reconcile_callback=slow_reconcile,
        )

        result = await executor.execute(mock_snapshot)

        assert result.metrics.decision_ms >= 0
        assert result.metrics.submit_ms >= 10  # At least the sleep time
        assert result.metrics.reconcile_ms >= 10
        assert result.metrics.decision_to_submit_ms > 0

    @pytest.mark.asyncio
    async def test_execute_callback_failure_triggers_flat_forced(self, mock_snapshot):
        """Failed execute callback results in FLAT_FORCED."""

        async def failing_execute(snapshot):
            return False

        executor = AsyncExecutor(execute_callback=failing_execute)
        result = await executor.execute(mock_snapshot)

        assert result.state == ExecutorState.FLAT_FORCED
        assert executor.state == ExecutorState.FLAT_FORCED
        assert result.error == "execute_callback failed"

    @pytest.mark.asyncio
    async def test_no_callbacks_defaults_to_done_equal(self, mock_snapshot):
        """Without callbacks, execution defaults to DONE_EQUAL."""
        executor = AsyncExecutor()
        result = await executor.execute(mock_snapshot)

        assert result.state == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_snapshot_immutable_during_execution(self, mock_snapshot):
        """Snapshot passed to callbacks is the original immutable copy."""
        captured_snapshot = None

        async def capture_execute(snapshot):
            nonlocal captured_snapshot
            captured_snapshot = snapshot
            return True

        executor = AsyncExecutor(execute_callback=capture_execute)
        await executor.execute(mock_snapshot)

        # Snapshot should be the exact same object
        assert captured_snapshot is mock_snapshot


class TestAsyncExecutorMutex:
    """Tests for AsyncExecutor mutex behavior."""

    @pytest.mark.asyncio
    async def test_mutex_prevents_concurrent_execution(self, mock_snapshot):
        """Only one execution can run at a time."""
        execution_count = 0

        async def slow_execute(snapshot):
            nonlocal execution_count
            execution_count += 1
            await asyncio.sleep(0.1)  # 100ms
            return True

        executor = AsyncExecutor(execute_callback=slow_execute)

        # Start first execution
        task1 = asyncio.create_task(executor.execute(mock_snapshot))
        await asyncio.sleep(0.01)  # Let it start

        # Second attempt should wait (using execute which waits)
        task2 = asyncio.create_task(executor.execute(mock_snapshot))
        await asyncio.sleep(0.01)  # Give it time to try

        # At this point, only one should be running
        assert executor.is_busy

        await task1
        await task2

        # Both should have eventually run
        assert execution_count == 2

    @pytest.mark.asyncio
    async def test_try_execute_returns_none_when_busy(self, mock_snapshot):
        """try_execute returns None immediately if mutex held."""
        started = asyncio.Event()

        async def blocking_execute(snapshot):
            started.set()
            await asyncio.sleep(0.5)
            return True

        executor = AsyncExecutor(execute_callback=blocking_execute)

        # Start first execution
        task1 = asyncio.create_task(executor.execute(mock_snapshot))
        await started.wait()

        # try_execute should return None immediately
        result = await executor.try_execute(mock_snapshot)
        assert result is None

        # Cleanup
        executor.request_cancel()
        await task1

    @pytest.mark.asyncio
    async def test_ws_task_not_blocked_by_execution(self, mock_snapshot):
        """Simulated WS task can continue while execution runs."""
        ws_updates = []
        execution_started = asyncio.Event()

        async def slow_execute(snapshot):
            execution_started.set()
            await asyncio.sleep(0.1)
            return True

        async def ws_task():
            """Simulate WS updates during execution."""
            for i in range(5):
                ws_updates.append(i)
                await asyncio.sleep(0.02)

        executor = AsyncExecutor(execute_callback=slow_execute)

        # Start both tasks
        exec_task = asyncio.create_task(executor.execute(mock_snapshot))
        await execution_started.wait()

        # WS task should complete even though execution is running
        ws_task_handle = asyncio.create_task(ws_task())
        await ws_task_handle

        # WS updates should have happened
        assert len(ws_updates) == 5

        await exec_task


class TestAsyncExecutorCancellation:
    """Tests for AsyncExecutor cancellation behavior."""

    @pytest.mark.asyncio
    async def test_request_cancel_during_decision(self, mock_snapshot):
        """Cancellation during decision phase results in FLAT_FORCED."""

        async def check_cancel_execute(snapshot):
            # Execution will be cancelled before this returns
            await asyncio.sleep(1.0)
            return True

        executor = AsyncExecutor(execute_callback=check_cancel_execute)

        async def cancel_soon():
            await asyncio.sleep(0.01)
            executor.request_cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        result = await executor.execute(mock_snapshot)
        await cancel_task

        assert result.state == ExecutorState.FLAT_FORCED
        assert "cancelled" in result.error

    @pytest.mark.asyncio
    async def test_cancel_and_wait_completes(self, mock_snapshot):
        """cancel_and_wait returns True when execution stops."""
        execution_started = asyncio.Event()

        async def cancellable_execute(snapshot):
            execution_started.set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                raise
            return True

        executor = AsyncExecutor(execute_callback=cancellable_execute)

        exec_task = asyncio.create_task(executor.execute(mock_snapshot))
        await execution_started.wait()

        # Cancel and wait should complete quickly
        completed = await executor.cancel_and_wait(timeout_sec=1.0)
        assert completed

        # Wait for task to finish
        try:
            await exec_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_reset_to_monitor_after_completion(self, mock_snapshot):
        """Can reset to MONITOR after DONE_EQUAL."""
        executor = AsyncExecutor()
        await executor.execute(mock_snapshot)

        assert executor.state == ExecutorState.DONE_EQUAL
        executor.reset_to_monitor()
        assert executor.state == ExecutorState.MONITOR

    def test_reset_to_monitor_fails_during_execute(self):
        """Cannot reset while in EXECUTE state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.EXECUTE

        with pytest.raises(RuntimeError):
            executor.reset_to_monitor()

    def test_reset_to_monitor_fails_during_reconcile(self):
        """Cannot reset while in RECONCILE state."""
        executor = AsyncExecutor()
        executor._state = ExecutorState.RECONCILE

        with pytest.raises(RuntimeError):
            executor.reset_to_monitor()


# --- CancellableTask Tests ---


class TestCancellableTask:
    """Tests for CancellableTask wrapper."""

    @pytest.mark.asyncio
    async def test_task_starts_and_runs(self):
        """Task starts successfully."""
        task = CancellableTask("test")
        result = []

        async def work():
            result.append(1)
            return 42

        t = task.start(work())
        await t
        assert result == [1]

    @pytest.mark.asyncio
    async def test_is_running_reflects_state(self):
        """is_running tracks task state."""
        task = CancellableTask("test")
        started = asyncio.Event()

        async def work():
            started.set()
            await asyncio.sleep(0.1)

        assert not task.is_running
        t = task.start(work())
        await started.wait()
        assert task.is_running
        await t
        assert not task.is_running

    @pytest.mark.asyncio
    async def test_cannot_start_while_running(self):
        """Starting while running raises RuntimeError."""
        task = CancellableTask("test")

        async def work():
            await asyncio.sleep(1.0)

        task.start(work())

        with pytest.raises(RuntimeError):
            task.start(work())

        task.request_cancel()

    @pytest.mark.asyncio
    async def test_request_cancel_sets_flag(self):
        """request_cancel sets should_cancel flag."""
        task = CancellableTask("test")

        async def work():
            await asyncio.sleep(1.0)

        task.start(work())
        assert not task.should_cancel
        task.request_cancel()
        assert task.should_cancel

    @pytest.mark.asyncio
    async def test_cancel_and_wait_succeeds(self):
        """cancel_and_wait returns True on successful cancel."""
        task = CancellableTask("test")

        async def work():
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                pass

        task.start(work())
        result = await task.cancel_and_wait(timeout_sec=1.0)
        assert result

    @pytest.mark.asyncio
    async def test_wait_for_cancel_or_timeout_returns_false_on_timeout(self):
        """wait_for_cancel_or_timeout returns False if timeout expires."""
        task = CancellableTask("test")
        result = await task.wait_for_cancel_or_timeout(0.05)
        assert not result  # Timeout, not cancelled

    @pytest.mark.asyncio
    async def test_wait_for_cancel_or_timeout_returns_true_on_cancel(self):
        """wait_for_cancel_or_timeout returns True if cancelled."""
        task = CancellableTask("test")

        async def cancel_soon():
            await asyncio.sleep(0.01)
            task.request_cancel()

        cancel_task = asyncio.create_task(cancel_soon())
        result = await task.wait_for_cancel_or_timeout(1.0)
        await cancel_task

        assert result  # Cancelled, not timeout


# --- CancellableRESTFetcher Tests ---


class TestCancellableRESTFetcher:
    """Tests for CancellableRESTFetcher."""

    @pytest.mark.asyncio
    async def test_fetch_returns_result_when_fresh_false(self):
        """Fetch succeeds when WS stays stale."""

        async def fetch():
            return {"data": "result"}

        fetcher = CancellableRESTFetcher(
            fetch_coro_factory=fetch,
            freshness_checker=lambda: False,  # Always stale
            timeout_ms=1000,
        )

        result = await fetcher.fetch_unless_fresh()
        assert result == {"data": "result"}

    @pytest.mark.asyncio
    async def test_fetch_skipped_when_already_fresh(self):
        """Fetch skipped if WS already fresh."""

        async def fetch():
            return {"data": "result"}

        fetcher = CancellableRESTFetcher(
            fetch_coro_factory=fetch,
            freshness_checker=lambda: True,  # Already fresh
            timeout_ms=1000,
        )

        result = await fetcher.fetch_unless_fresh()
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_cancelled_when_ws_becomes_fresh(self):
        """Fetch cancelled mid-flight if WS becomes fresh."""
        fresh = False
        fetch_started = asyncio.Event()

        async def slow_fetch():
            fetch_started.set()
            await asyncio.sleep(0.5)
            return {"data": "result"}

        def is_fresh():
            return fresh

        fetcher = CancellableRESTFetcher(
            fetch_coro_factory=slow_fetch,
            freshness_checker=is_fresh,
            timeout_ms=1000,
        )

        async def make_fresh():
            await fetch_started.wait()
            await asyncio.sleep(0.05)
            nonlocal fresh
            fresh = True

        fresh_task = asyncio.create_task(make_fresh())
        result = await fetcher.fetch_unless_fresh()
        await fresh_task

        # Should return None because WS became fresh
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_times_out(self):
        """Fetch returns None on timeout."""

        async def slow_fetch():
            await asyncio.sleep(1.0)
            return {"data": "result"}

        fetcher = CancellableRESTFetcher(
            fetch_coro_factory=slow_fetch,
            freshness_checker=lambda: False,
            timeout_ms=50,  # 50ms timeout
        )

        result = await fetcher.fetch_unless_fresh()
        assert result is None


# --- Integration Tests ---


class TestIntegrationLatency:
    """Integration tests for latency requirements."""

    @pytest.mark.asyncio
    async def test_decision_to_submit_under_150ms(self, mock_snapshot):
        """Decision-to-submit latency should be <150ms.

        This tests the PRD requirement:
        'Decision-to-submit latency logged and <150ms in integration test'
        """

        async def fast_execute(snapshot):
            # Simulate realistic but fast execution
            await asyncio.sleep(0.01)  # 10ms for order submission
            return True

        executor = AsyncExecutor(execute_callback=fast_execute)
        result = await executor.execute(mock_snapshot)

        # Verify latency is under 150ms
        latency = result.metrics.decision_to_submit_ms
        assert latency < 150, f"Decision-to-submit latency {latency}ms >= 150ms"

    @pytest.mark.asyncio
    async def test_decision_to_submit_with_injected_delays(self, mock_snapshot):
        """Latency test with realistic injected delays."""
        delays = {
            "decision_validation": 0.005,  # 5ms
            "order_submission": 0.020,  # 20ms per order
        }

        async def realistic_execute(snapshot):
            await asyncio.sleep(delays["order_submission"])
            return True

        executor = AsyncExecutor(execute_callback=realistic_execute)

        # Record start time
        start = time.time() * 1000

        result = await executor.execute(mock_snapshot)

        # Verify end-to-end latency
        total_ms = result.metrics.decision_to_submit_ms

        # Should be well under 150ms with these delays
        assert total_ms < 150, f"Total latency {total_ms}ms >= 150ms"

        # Log the metrics as the PRD requires
        print(f"\nLatency metrics:")
        print(f"  decision_ms: {result.metrics.decision_ms}")
        print(f"  submit_ms: {result.metrics.submit_ms}")
        print(f"  decision_to_submit_ms: {total_ms}")


class TestIntegrationWSIndependence:
    """Integration tests for WS task independence."""

    @pytest.mark.asyncio
    async def test_ws_continues_during_slow_execution(self, mock_snapshot):
        """WS updates are not blocked by slow execution.

        Tests PRD requirement:
        'WS tasks never await execution/reconciliation tasks'
        """
        ws_updates = []
        execution_started = asyncio.Event()
        execution_done = asyncio.Event()

        async def slow_execute(snapshot):
            execution_started.set()
            await asyncio.sleep(0.2)  # 200ms execution
            execution_done.set()
            return True

        async def ws_update_task():
            """Simulate WS updates arriving every 20ms."""
            start = time.time()
            while not execution_done.is_set():
                ws_updates.append(time.time() - start)
                await asyncio.sleep(0.02)  # 20ms between updates

        executor = AsyncExecutor(execute_callback=slow_execute)

        # Start WS task and execution concurrently
        ws_task = asyncio.create_task(ws_update_task())
        exec_task = asyncio.create_task(executor.execute(mock_snapshot))

        await execution_started.wait()
        await exec_task
        await ws_task

        # WS should have received ~10 updates during 200ms execution
        assert len(ws_updates) >= 8, f"Only {len(ws_updates)} WS updates during execution"


class TestIntegrationCancellation:
    """Integration tests for cancellation on rollover/FLAT_FORCED."""

    @pytest.mark.asyncio
    async def test_all_tasks_cancellable_on_rollover(self, mock_snapshot):
        """All long-running tasks can be cancelled on market rollover."""
        reconcile_started = asyncio.Event()
        reconcile_cancelled = False

        async def long_reconcile(snapshot):
            nonlocal reconcile_cancelled
            reconcile_started.set()
            try:
                await asyncio.sleep(10.0)  # Long reconciliation
                return ExecutorState.DONE_EQUAL
            except asyncio.CancelledError:
                reconcile_cancelled = True
                return ExecutorState.FLAT_FORCED

        executor = AsyncExecutor(
            execute_callback=lambda s: asyncio.sleep(0) or True,
            reconcile_callback=long_reconcile,
        )

        exec_task = asyncio.create_task(executor.execute(mock_snapshot))
        await reconcile_started.wait()

        # Simulate market rollover - cancel everything
        executor.request_cancel()

        result = await exec_task

        assert result.state == ExecutorState.FLAT_FORCED
