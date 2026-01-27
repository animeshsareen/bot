"""Async event-driven execution architecture for cross-venue arbitrage.

Provides non-blocking concurrency so market data ingestion remains continuous
while execution and reconciliation run in separate guarded tasks.

Key design principles:
- WS tasks NEVER await execution/reconciliation tasks
- Trade decisions operate on immutable Snapshot copies
- Execution/reconciliation guarded by mutex (single active attempt)
- All long-running tasks cancellable on rollover or FLAT_FORCED
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Any, Awaitable

from .venue_state import MarketSnapshot, create_snapshot

logger = logging.getLogger(__name__)


class ExecutorState(Enum):
    """State machine states for execution lifecycle."""

    MONITOR = "MONITOR"
    EXECUTE = "EXECUTE"
    RECONCILE = "RECONCILE"
    DONE_EQUAL = "DONE_EQUAL"
    FLAT_FORCED = "FLAT_FORCED"
    PREFLIGHT_REJECTED = "PREFLIGHT_REJECTED"  # Preflight check failed, no orders submitted


@dataclass(frozen=True)
class LatencyMetrics:
    """Timing metrics for a single execution attempt.

    Attributes:
        decision_start_ms: When opportunity was detected
        decision_end_ms: When decision to trade was finalized
        submit_start_ms: When order submission began
        submit_end_ms: When both orders were submitted
        reconcile_start_ms: When reconciliation began (if needed)
        reconcile_end_ms: When reconciliation completed
    """

    decision_start_ms: int = 0
    decision_end_ms: int = 0
    submit_start_ms: int = 0
    submit_end_ms: int = 0
    reconcile_start_ms: int = 0
    reconcile_end_ms: int = 0

    @property
    def decision_ms(self) -> int:
        """Time from decision start to end."""
        if self.decision_start_ms and self.decision_end_ms:
            return self.decision_end_ms - self.decision_start_ms
        return 0

    @property
    def submit_ms(self) -> int:
        """Time from submit start to end."""
        if self.submit_start_ms and self.submit_end_ms:
            return self.submit_end_ms - self.submit_start_ms
        return 0

    @property
    def reconcile_ms(self) -> int:
        """Time from reconcile start to end."""
        if self.reconcile_start_ms and self.reconcile_end_ms:
            return self.reconcile_end_ms - self.reconcile_start_ms
        return 0

    @property
    def decision_to_submit_ms(self) -> int:
        """Total time from decision start to submit end."""
        if self.decision_start_ms and self.submit_end_ms:
            return self.submit_end_ms - self.decision_start_ms
        return 0

    def with_decision_start(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with decision start time."""
        return LatencyMetrics(
            decision_start_ms=ms,
            decision_end_ms=self.decision_end_ms,
            submit_start_ms=self.submit_start_ms,
            submit_end_ms=self.submit_end_ms,
            reconcile_start_ms=self.reconcile_start_ms,
            reconcile_end_ms=self.reconcile_end_ms,
        )

    def with_decision_end(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with decision end time."""
        return LatencyMetrics(
            decision_start_ms=self.decision_start_ms,
            decision_end_ms=ms,
            submit_start_ms=self.submit_start_ms,
            submit_end_ms=self.submit_end_ms,
            reconcile_start_ms=self.reconcile_start_ms,
            reconcile_end_ms=self.reconcile_end_ms,
        )

    def with_submit_start(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with submit start time."""
        return LatencyMetrics(
            decision_start_ms=self.decision_start_ms,
            decision_end_ms=self.decision_end_ms,
            submit_start_ms=ms,
            submit_end_ms=self.submit_end_ms,
            reconcile_start_ms=self.reconcile_start_ms,
            reconcile_end_ms=self.reconcile_end_ms,
        )

    def with_submit_end(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with submit end time."""
        return LatencyMetrics(
            decision_start_ms=self.decision_start_ms,
            decision_end_ms=self.decision_end_ms,
            submit_start_ms=self.submit_start_ms,
            submit_end_ms=ms,
            reconcile_start_ms=self.reconcile_start_ms,
            reconcile_end_ms=self.reconcile_end_ms,
        )

    def with_reconcile_start(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with reconcile start time."""
        return LatencyMetrics(
            decision_start_ms=self.decision_start_ms,
            decision_end_ms=self.decision_end_ms,
            submit_start_ms=self.submit_start_ms,
            submit_end_ms=self.submit_end_ms,
            reconcile_start_ms=ms,
            reconcile_end_ms=self.reconcile_end_ms,
        )

    def with_reconcile_end(self, ms: int) -> "LatencyMetrics":
        """Return new metrics with reconcile end time."""
        return LatencyMetrics(
            decision_start_ms=self.decision_start_ms,
            decision_end_ms=self.decision_end_ms,
            submit_start_ms=self.submit_start_ms,
            submit_end_ms=self.submit_end_ms,
            reconcile_start_ms=self.reconcile_start_ms,
            reconcile_end_ms=ms,
        )


@dataclass
class ExecutionResult:
    """Result of an execution attempt.

    Attributes:
        snapshot: The immutable snapshot used for this attempt
        state: Final executor state
        metrics: Latency metrics for this attempt
        error: Error message if execution failed
    """

    snapshot: MarketSnapshot
    state: ExecutorState
    metrics: LatencyMetrics
    error: Optional[str] = None


# Type alias for callback functions
ExecuteCallback = Callable[[MarketSnapshot], Awaitable[bool]]
ReconcileCallback = Callable[[MarketSnapshot], Awaitable[ExecutorState]]


class AsyncExecutor:
    """Async executor with mutex-guarded execution and reconciliation.

    Ensures:
    - Single active execution attempt (mutex)
    - WS tasks never blocked
    - All operations cancellable
    - Latency tracking

    The executor operates on immutable snapshot copies, so WS tasks can
    continue updating VenueState without blocking.
    """

    def __init__(
        self,
        execute_callback: Optional[ExecuteCallback] = None,
        reconcile_callback: Optional[ReconcileCallback] = None,
        flatten_callback: Optional[Callable[[MarketSnapshot], Awaitable[None]]] = None,
    ):
        """Initialize executor.

        Args:
            execute_callback: Async function to execute trade (returns success bool)
            reconcile_callback: Async function to reconcile fills (returns final state)
            flatten_callback: Async function to flatten positions (on FORCE_FLAT)
        """
        self._execute_callback = execute_callback
        self._reconcile_callback = reconcile_callback
        self._flatten_callback = flatten_callback
        self._execution_mutex = asyncio.Lock()
        self._state = ExecutorState.MONITOR
        self._current_task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()
        self._metrics = LatencyMetrics()

    @property
    def state(self) -> ExecutorState:
        """Current executor state."""
        return self._state

    @property
    def is_busy(self) -> bool:
        """True if execution or reconciliation is in progress."""
        return self._execution_mutex.locked()

    @property
    def metrics(self) -> LatencyMetrics:
        """Current latency metrics."""
        return self._metrics

    def can_start_attempt(self) -> bool:
        """Check if a new execution attempt can start.

        Returns:
            False if already in EXECUTE, RECONCILE, or FLAT_FORCED
        """
        return self._state in (
            ExecutorState.MONITOR,
            ExecutorState.DONE_EQUAL,
            ExecutorState.PREFLIGHT_REJECTED,  # No orders submitted, can retry
        )

    async def try_execute(self, snapshot: MarketSnapshot) -> Optional[ExecutionResult]:
        """Attempt to execute a trade if not already busy.

        This method is non-blocking from the caller's perspective.
        If the mutex is already held, returns None immediately.

        Args:
            snapshot: Immutable market snapshot for this attempt

        Returns:
            ExecutionResult if execution completed, None if busy or cancelled
        """
        # Non-blocking check - don't wait for mutex
        if self._execution_mutex.locked():
            logger.debug("Execution skipped: already in progress")
            return None

        if not self.can_start_attempt():
            logger.debug("Execution skipped: state is %s", self._state)
            return None

        async with self._execution_mutex:
            return await self._execute_internal(snapshot)

    async def execute(self, snapshot: MarketSnapshot) -> ExecutionResult:
        """Execute a trade, waiting for mutex if needed.

        This method WILL wait for any ongoing execution to complete.
        Use try_execute() for non-blocking behavior.

        Args:
            snapshot: Immutable market snapshot for this attempt

        Returns:
            ExecutionResult with final state and metrics
        """
        async with self._execution_mutex:
            return await self._execute_internal(snapshot)

    async def _execute_internal(self, snapshot: MarketSnapshot) -> ExecutionResult:
        """Internal execution logic (must hold mutex)."""
        self._current_task = asyncio.current_task()
        self._cancel_event.clear()
        now_ms = self._now_ms()
        self._metrics = LatencyMetrics().with_decision_start(now_ms)

        try:
            # EXECUTE phase
            self._state = ExecutorState.EXECUTE
            logger.info(
                "Execution started for %s at %d",
                snapshot.aligned_market_id,
                now_ms,
            )

            # Decision validation (would include profit gates in real impl)
            self._metrics = self._metrics.with_decision_end(self._now_ms())
            decision_ms = self._metrics.decision_ms
            logger.info("Decision completed in %dms", decision_ms)

            if self._cancel_event.is_set():
                logger.info("Execution cancelled during decision")
                await self._invoke_flatten(snapshot)
                self._state = ExecutorState.FLAT_FORCED
                return ExecutionResult(
                    snapshot=snapshot,
                    state=self._state,
                    metrics=self._metrics,
                    error="cancelled",
                )

            # Submit orders
            self._metrics = self._metrics.with_submit_start(self._now_ms())

            if self._execute_callback:
                # execute_callback returns:
                #   True  = success, continue to reconciliation
                #   False = execution failed, needs flatten
                #   None  = preflight rejected, no orders submitted, no flatten needed
                result = await self._execute_callback(snapshot)
                self._metrics = self._metrics.with_submit_end(self._now_ms())

                if result is None:
                    # Preflight rejection - no orders were submitted, nothing to flatten
                    self._state = ExecutorState.PREFLIGHT_REJECTED
                    return ExecutionResult(
                        snapshot=snapshot,
                        state=self._state,
                        metrics=self._metrics,
                        error="preflight_rejected",
                    )

                if result is False:
                    # Execution failed after orders may have been submitted - need flatten
                    await self._invoke_flatten(snapshot)
                    self._state = ExecutorState.FLAT_FORCED
                    return ExecutionResult(
                        snapshot=snapshot,
                        state=self._state,
                        metrics=self._metrics,
                        error="execute_callback failed",
                    )

            self._metrics = self._metrics.with_submit_end(self._now_ms())
            submit_ms = self._metrics.submit_ms
            decision_to_submit = self._metrics.decision_to_submit_ms
            logger.info(
                "Orders submitted in %dms (decision-to-submit: %dms)",
                submit_ms,
                decision_to_submit,
            )

            if self._cancel_event.is_set():
                logger.info("Execution cancelled after submit")
                await self._invoke_flatten(snapshot)
                self._state = ExecutorState.FLAT_FORCED
                return ExecutionResult(
                    snapshot=snapshot,
                    state=self._state,
                    metrics=self._metrics,
                    error="cancelled after submit",
                )

            # RECONCILE phase
            self._state = ExecutorState.RECONCILE
            self._metrics = self._metrics.with_reconcile_start(self._now_ms())

            if self._reconcile_callback:
                final_state = await self._reconcile_callback(snapshot)
                self._state = final_state
            else:
                # Default: assume equal fills
                self._state = ExecutorState.DONE_EQUAL

            self._metrics = self._metrics.with_reconcile_end(self._now_ms())
            reconcile_ms = self._metrics.reconcile_ms
            logger.info(
                "Reconciliation completed in %dms, final state: %s",
                reconcile_ms,
                self._state.value,
            )

            return ExecutionResult(
                snapshot=snapshot,
                state=self._state,
                metrics=self._metrics,
            )

        except asyncio.CancelledError:
            logger.info("Execution task cancelled")
            await self._invoke_flatten(snapshot)
            self._state = ExecutorState.FLAT_FORCED
            return ExecutionResult(
                snapshot=snapshot,
                state=self._state,
                metrics=self._metrics,
                error="task cancelled",
            )
        finally:
            self._current_task = None
            
    async def _invoke_flatten(self, snapshot: MarketSnapshot) -> None:
        """Invoke flatten callback if configured."""
        if self._flatten_callback:
            try:
                logger.info("Invoking flatten callback due to error/cancel")
                await self._flatten_callback(snapshot)
            except Exception as e:
                logger.error("Flatten callback failed: %s", e)

    def request_cancel(self) -> None:
        """Request cancellation of current execution.

        This sets a cancel event that execution checks between phases.
        The execution will transition to FLAT_FORCED.
        """
        self._cancel_event.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        logger.info("Cancellation requested")

    async def cancel_and_wait(self, timeout_sec: float = 5.0) -> bool:
        """Request cancellation and wait for execution to complete.

        Args:
            timeout_sec: Maximum time to wait for completion

        Returns:
            True if execution completed within timeout
        """
        self.request_cancel()

        if not self._execution_mutex.locked():
            return True

        try:
            # Wait for mutex to be released
            async with asyncio.timeout(timeout_sec):
                async with self._execution_mutex:
                    pass
            return True
        except asyncio.TimeoutError:
            logger.warning("Cancel timeout after %.1fs", timeout_sec)
            return False

    def reset_to_monitor(self) -> None:
        """Reset state to MONITOR for new market.

        Only valid when not in EXECUTE or RECONCILE.
        """
        if self._state in (ExecutorState.EXECUTE, ExecutorState.RECONCILE):
            raise RuntimeError(f"Cannot reset from state {self._state}")
        self._state = ExecutorState.MONITOR
        self._metrics = LatencyMetrics()
        logger.info("Executor reset to MONITOR")

    def _now_ms(self) -> int:
        """Current time in milliseconds."""
        return int(time.time() * 1000)


class CancellableTask:
    """Wrapper for long-running async tasks with cancellation support.

    Provides clean cancellation and status tracking for tasks like
    REST fallback or reconciliation loops.
    """

    def __init__(self, name: str):
        self.name = name
        self._task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        """True if task is currently running."""
        return self._task is not None and not self._task.done()

    @property
    def should_cancel(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_event.is_set()

    def start(self, coro: Awaitable[Any]) -> asyncio.Task:
        """Start a new task.

        Args:
            coro: Coroutine to run

        Returns:
            The started Task

        Raises:
            RuntimeError: If a task is already running
        """
        if self.is_running:
            raise RuntimeError(f"Task {self.name} is already running")

        self._cancel_event.clear()
        self._task = asyncio.create_task(coro, name=self.name)
        return self._task

    def request_cancel(self) -> None:
        """Request task cancellation.

        The task should check should_cancel periodically.
        """
        self._cancel_event.set()
        if self._task and not self._task.done():
            self._task.cancel()

    async def cancel_and_wait(self, timeout_sec: float = 5.0) -> bool:
        """Cancel task and wait for completion.

        Args:
            timeout_sec: Maximum time to wait

        Returns:
            True if task completed within timeout
        """
        self.request_cancel()

        if self._task is None or self._task.done():
            return True

        try:
            await asyncio.wait_for(
                asyncio.shield(self._task),
                timeout=timeout_sec,
            )
            return True
        except asyncio.CancelledError:
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for_cancel_or_timeout(
        self, timeout_sec: float
    ) -> bool:
        """Wait until cancel requested or timeout.

        Useful for interruptible sleeps in reconciliation loops.

        Args:
            timeout_sec: Maximum time to wait

        Returns:
            True if cancelled, False if timeout
        """
        try:
            await asyncio.wait_for(
                self._cancel_event.wait(),
                timeout=timeout_sec,
            )
            return True
        except asyncio.TimeoutError:
            return False


class CancellableRESTFetcher:
    """REST fetcher that can be cancelled if WS becomes fresh.

    Implements the PRD requirement:
    "REST fallback is async and cancellable if WS becomes fresh"
    """

    def __init__(
        self,
        fetch_coro_factory: Callable[[], Awaitable[Any]],
        freshness_checker: Callable[[], bool],
        timeout_ms: int = 300,
    ):
        """Initialize fetcher.

        Args:
            fetch_coro_factory: Factory that creates the fetch coroutine
            freshness_checker: Callable that returns True if WS is now fresh
            timeout_ms: Fetch timeout in milliseconds
        """
        self._fetch_factory = fetch_coro_factory
        self._is_fresh = freshness_checker
        self._timeout_ms = timeout_ms
        self._task = CancellableTask("rest_fetch")

    async def fetch_unless_fresh(self) -> Optional[Any]:
        """Fetch via REST unless WS becomes fresh.

        Cancels REST fetch if WS data becomes fresh during fetch.

        Returns:
            Fetch result or None if cancelled/timeout/fresh
        """
        if self._is_fresh():
            logger.debug("Skipping REST fetch: WS already fresh")
            return None

        fetch_coro = self._fetch_factory()

        async def fetch_with_freshness_check():
            task = asyncio.current_task()
            check_interval = 0.05  # 50ms

            # Start the actual fetch
            fetch_task = asyncio.create_task(fetch_coro)

            while not fetch_task.done():
                # Check if WS became fresh
                if self._is_fresh():
                    logger.info("WS became fresh, cancelling REST fetch")
                    fetch_task.cancel()
                    try:
                        await fetch_task
                    except asyncio.CancelledError:
                        pass
                    return None

                # Wait a bit before checking again
                try:
                    await asyncio.wait_for(
                        asyncio.shield(fetch_task),
                        timeout=check_interval,
                    )
                    return fetch_task.result()
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    fetch_task.cancel()
                    raise

            return fetch_task.result()

        try:
            timeout_sec = self._timeout_ms / 1000.0
            return await asyncio.wait_for(
                fetch_with_freshness_check(),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("REST fetch exceeded %dms timeout", self._timeout_ms)
            return None
        except asyncio.CancelledError:
            logger.debug("REST fetch cancelled")
            return None
