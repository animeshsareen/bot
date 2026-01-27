"""Lock manager for preventing race conditions in order submission.

Provides:
- VenueMarketLock: Per-(venue, market_id) lock to prevent 409 conflicts
- AttemptLock: Ensures only one attempt progresses at a time
- LockManager: Central registry for all locks

Key design principles:
- asyncio-based for async compatibility
- Timeout with warning threshold
- Lock per (venue, market) to avoid broad contention
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_LOCK_TIMEOUT_SEC = 5.0
DEFAULT_WARN_THRESHOLD_SEC = 2.0


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LockError(Exception):
    """Base exception for lock errors."""
    pass


class LockTimeoutError(LockError):
    """Raised when lock acquisition times out."""
    pass


class LockNotHeldError(LockError):
    """Raised when trying to release a lock not held."""
    pass


# =============================================================================
# LOCK METRICS
# =============================================================================

@dataclass
class LockMetrics:
    """Metrics for a single lock.

    Attributes:
        acquisitions: Total successful acquisitions
        timeouts: Total timeout failures
        contentions: Times lock was already held on acquire attempt
        total_hold_time_ms: Cumulative hold time
        max_hold_time_ms: Longest single hold duration
    """
    acquisitions: int = 0
    timeouts: int = 0
    contentions: int = 0
    total_hold_time_ms: float = 0.0
    max_hold_time_ms: float = 0.0


@dataclass
class LockManagerMetrics:
    """Aggregate metrics for LockManager.

    Attributes:
        order_lock_acquisitions: Total order lock acquisitions
        order_lock_timeouts: Total order lock timeouts
        attempt_lock_acquisitions: Total attempt lock acquisitions
        attempt_lock_timeouts: Total attempt lock timeouts
        warnings_emitted: Times warning threshold was exceeded
    """
    order_lock_acquisitions: int = 0
    order_lock_timeouts: int = 0
    attempt_lock_acquisitions: int = 0
    attempt_lock_timeouts: int = 0
    warnings_emitted: int = 0


# =============================================================================
# VENUE MARKET LOCK
# =============================================================================

class VenueMarketLock:
    """Lock for a specific (venue, market_id) combination.

    Prevents concurrent order submissions to the same venue+market,
    avoiding 409 "order already exists" conflicts.

    Usage:
        lock = VenueMarketLock("kalshi", "TEST-MKT")
        async with lock.acquire(timeout=5.0):
            # Submit order, wait for terminal status
            pass
    """

    def __init__(
        self,
        venue: str,
        market_id: str,
        timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
        warn_threshold_sec: float = DEFAULT_WARN_THRESHOLD_SEC,
    ):
        """Initialize venue-market lock.

        Args:
            venue: Venue name ("kalshi" or "poly")
            market_id: Market identifier
            timeout_sec: Default acquisition timeout
            warn_threshold_sec: Threshold for emitting warning
        """
        self.venue = venue
        self.market_id = market_id
        self.timeout_sec = timeout_sec
        self.warn_threshold_sec = warn_threshold_sec

        self._lock = asyncio.Lock()
        self._holder: Optional[str] = None  # For debugging
        self._acquired_at: Optional[float] = None
        self.metrics = LockMetrics()

    @property
    def key(self) -> tuple[str, str]:
        """Lock key as (venue, market_id) tuple."""
        return (self.venue, self.market_id)

    @property
    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self._lock.locked()

    @property
    def hold_time_sec(self) -> Optional[float]:
        """Current hold time in seconds, or None if not held."""
        if self._acquired_at is None:
            return None
        return time.monotonic() - self._acquired_at

    @asynccontextmanager
    async def acquire(
        self,
        timeout_sec: Optional[float] = None,
        holder: Optional[str] = None,
    ):
        """Acquire the lock with timeout.

        Args:
            timeout_sec: Override default timeout
            holder: Optional identifier for debugging

        Yields:
            None when lock is acquired

        Raises:
            LockTimeoutError: If acquisition times out
        """
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        start = time.monotonic()

        # Track contention
        if self._lock.locked():
            self.metrics.contentions += 1
            logger.debug(
                "Lock contention on %s/%s (held by %s)",
                self.venue, self.market_id, self._holder
            )

        try:
            acquired = await asyncio.wait_for(
                self._lock.acquire(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.metrics.timeouts += 1
            wait_time = time.monotonic() - start
            logger.error(
                "Lock timeout on %s/%s after %.2fs (held by %s)",
                self.venue, self.market_id, wait_time, self._holder
            )
            raise LockTimeoutError(
                f"Timeout acquiring lock for {self.venue}/{self.market_id} "
                f"after {wait_time:.2f}s"
            )

        # Lock acquired
        wait_time = time.monotonic() - start
        self._holder = holder
        self._acquired_at = time.monotonic()
        self.metrics.acquisitions += 1

        # Warn if waited too long
        warned = False
        if wait_time > self.warn_threshold_sec:
            warned = True
            logger.warning(
                "Slow lock acquisition on %s/%s: %.2fs wait",
                self.venue, self.market_id, wait_time
            )

        try:
            yield warned
        finally:
            # Calculate hold time
            hold_time_ms = (time.monotonic() - self._acquired_at) * 1000
            self.metrics.total_hold_time_ms += hold_time_ms
            self.metrics.max_hold_time_ms = max(
                self.metrics.max_hold_time_ms, hold_time_ms
            )

            # Warn if held too long
            if hold_time_ms > self.warn_threshold_sec * 1000:
                logger.warning(
                    "Long lock hold on %s/%s: %.2fms",
                    self.venue, self.market_id, hold_time_ms
                )

            # Release
            self._holder = None
            self._acquired_at = None
            self._lock.release()

            logger.debug(
                "Released lock on %s/%s (held %.2fms)",
                self.venue, self.market_id, hold_time_ms
            )


# =============================================================================
# ATTEMPT LOCK
# =============================================================================

class AttemptLock:
    """Lock ensuring only one attempt progresses at a time.

    Held from attempt start until terminal state (DONE_*, FAILED).

    Usage:
        lock = AttemptLock()
        async with lock.acquire("attempt_123"):
            # Execute arb attempt
            pass
    """

    def __init__(
        self,
        timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
        warn_threshold_sec: float = DEFAULT_WARN_THRESHOLD_SEC,
    ):
        """Initialize attempt lock.

        Args:
            timeout_sec: Default acquisition timeout
            warn_threshold_sec: Threshold for emitting warning
        """
        self.timeout_sec = timeout_sec
        self.warn_threshold_sec = warn_threshold_sec

        self._lock = asyncio.Lock()
        self._current_attempt_id: Optional[str] = None
        self._acquired_at: Optional[float] = None
        self.metrics = LockMetrics()

    @property
    def is_locked(self) -> bool:
        """Check if lock is currently held."""
        return self._lock.locked()

    @property
    def current_attempt_id(self) -> Optional[str]:
        """ID of attempt currently holding the lock."""
        return self._current_attempt_id

    @property
    def hold_time_sec(self) -> Optional[float]:
        """Current hold time in seconds, or None if not held."""
        if self._acquired_at is None:
            return None
        return time.monotonic() - self._acquired_at

    @asynccontextmanager
    async def acquire(
        self,
        attempt_id: str,
        timeout_sec: Optional[float] = None,
    ):
        """Acquire the attempt lock.

        Args:
            attempt_id: ID of attempt acquiring the lock
            timeout_sec: Override default timeout

        Yields:
            None when lock is acquired

        Raises:
            LockTimeoutError: If acquisition times out
        """
        timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        start = time.monotonic()

        # Track contention
        if self._lock.locked():
            self.metrics.contentions += 1
            logger.debug(
                "Attempt lock contention: %s waiting for %s",
                attempt_id, self._current_attempt_id
            )

        try:
            await asyncio.wait_for(
                self._lock.acquire(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.metrics.timeouts += 1
            wait_time = time.monotonic() - start
            logger.error(
                "Attempt lock timeout for %s after %.2fs (held by %s)",
                attempt_id, wait_time, self._current_attempt_id
            )
            raise LockTimeoutError(
                f"Timeout acquiring attempt lock for {attempt_id} "
                f"after {wait_time:.2f}s (current: {self._current_attempt_id})"
            )

        # Lock acquired
        wait_time = time.monotonic() - start
        self._current_attempt_id = attempt_id
        self._acquired_at = time.monotonic()
        self.metrics.acquisitions += 1

        # Warn if waited too long
        warned = False
        if wait_time > self.warn_threshold_sec:
            warned = True
            logger.warning(
                "Slow attempt lock acquisition for %s: %.2fs wait",
                attempt_id, wait_time
            )

        logger.debug("Acquired attempt lock for %s", attempt_id)

        try:
            yield warned
        finally:
            # Calculate hold time
            hold_time_ms = (time.monotonic() - self._acquired_at) * 1000
            self.metrics.total_hold_time_ms += hold_time_ms
            self.metrics.max_hold_time_ms = max(
                self.metrics.max_hold_time_ms, hold_time_ms
            )

            # Warn if held too long
            if hold_time_ms > self.warn_threshold_sec * 1000:
                logger.warning(
                    "Long attempt lock hold for %s: %.2fms",
                    attempt_id, hold_time_ms
                )

            # Release
            released_id = self._current_attempt_id
            self._current_attempt_id = None
            self._acquired_at = None
            self._lock.release()

            logger.debug(
                "Released attempt lock for %s (held %.2fms)",
                released_id, hold_time_ms
            )


# =============================================================================
# LOCK MANAGER
# =============================================================================

class LockManager:
    """Central registry for all locks.

    Provides:
    - Per-(venue, market) order locks
    - Single attempt lock
    - Metrics aggregation

    Usage:
        manager = LockManager()

        # For order submission
        async with manager.acquire_order_lock("kalshi", "TEST-MKT"):
            # Submit order
            pass

        # For attempt lifecycle
        async with manager.acquire_attempt_lock("attempt_123"):
            # Execute attempt
            pass
    """

    def __init__(
        self,
        lock_timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
        warn_threshold_sec: float = DEFAULT_WARN_THRESHOLD_SEC,
    ):
        """Initialize lock manager.

        Args:
            lock_timeout_sec: Default timeout for all locks
            warn_threshold_sec: Warning threshold for all locks
        """
        self.lock_timeout_sec = lock_timeout_sec
        self.warn_threshold_sec = warn_threshold_sec

        # Order locks: (venue, market_id) -> VenueMarketLock
        self._order_locks: dict[tuple[str, str], VenueMarketLock] = {}
        self._order_locks_lock = asyncio.Lock()  # Protects _order_locks dict

        # Single attempt lock
        self._attempt_lock = AttemptLock(
            timeout_sec=lock_timeout_sec,
            warn_threshold_sec=warn_threshold_sec,
        )

        self.metrics = LockManagerMetrics()

    def _get_or_create_order_lock(
        self,
        venue: str,
        market_id: str,
    ) -> VenueMarketLock:
        """Get existing lock or create new one for (venue, market).

        Note: Caller must hold _order_locks_lock.
        """
        key = (venue, market_id)
        if key not in self._order_locks:
            self._order_locks[key] = VenueMarketLock(
                venue=venue,
                market_id=market_id,
                timeout_sec=self.lock_timeout_sec,
                warn_threshold_sec=self.warn_threshold_sec,
            )
        return self._order_locks[key]

    @asynccontextmanager
    async def acquire_order_lock(
        self,
        venue: str,
        market_id: str,
        timeout_sec: Optional[float] = None,
        holder: Optional[str] = None,
    ):
        """Acquire lock for order submission to venue+market.

        Args:
            venue: Target venue ("kalshi" or "poly")
            market_id: Target market ID
            timeout_sec: Override default timeout
            holder: Optional identifier for debugging

        Yields:
            bool: True if acquisition triggered a warning

        Raises:
            LockTimeoutError: If acquisition times out
        """
        # Get or create the lock (brief sync access to dict)
        async with self._order_locks_lock:
            lock = self._get_or_create_order_lock(venue, market_id)

        # Acquire the venue-market lock
        try:
            async with lock.acquire(timeout_sec=timeout_sec, holder=holder) as warned:
                self.metrics.order_lock_acquisitions += 1
                if warned:
                    self.metrics.warnings_emitted += 1
                yield warned
        except LockTimeoutError:
            self.metrics.order_lock_timeouts += 1
            raise

    @asynccontextmanager
    async def acquire_attempt_lock(
        self,
        attempt_id: str,
        timeout_sec: Optional[float] = None,
    ):
        """Acquire lock for attempt lifecycle.

        Args:
            attempt_id: Attempt being started
            timeout_sec: Override default timeout

        Yields:
            bool: True if acquisition triggered a warning

        Raises:
            LockTimeoutError: If acquisition times out
        """
        try:
            async with self._attempt_lock.acquire(
                attempt_id=attempt_id,
                timeout_sec=timeout_sec,
            ) as warned:
                self.metrics.attempt_lock_acquisitions += 1
                if warned:
                    self.metrics.warnings_emitted += 1
                yield warned
        except LockTimeoutError:
            self.metrics.attempt_lock_timeouts += 1
            raise

    def get_order_lock(self, venue: str, market_id: str) -> Optional[VenueMarketLock]:
        """Get order lock if it exists (for inspection/testing)."""
        return self._order_locks.get((venue, market_id))

    @property
    def attempt_lock(self) -> AttemptLock:
        """Get the attempt lock (for inspection/testing)."""
        return self._attempt_lock

    @property
    def active_order_locks(self) -> list[tuple[str, str]]:
        """List of (venue, market) pairs with currently held locks."""
        return [
            key for key, lock in self._order_locks.items()
            if lock.is_locked
        ]

    @property
    def current_attempt_id(self) -> Optional[str]:
        """ID of attempt currently holding the lock, or None."""
        return self._attempt_lock.current_attempt_id

    def get_aggregate_metrics(self) -> LockManagerMetrics:
        """Get aggregate metrics across all locks."""
        return self.metrics


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_lock_manager(
    lock_timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC,
    warn_threshold_sec: float = DEFAULT_WARN_THRESHOLD_SEC,
) -> LockManager:
    """Factory to create LockManager with configuration.

    Args:
        lock_timeout_sec: Timeout for lock acquisition
        warn_threshold_sec: Threshold for warning logs

    Returns:
        Configured LockManager
    """
    return LockManager(
        lock_timeout_sec=lock_timeout_sec,
        warn_threshold_sec=warn_threshold_sec,
    )
