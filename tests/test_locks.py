"""Comprehensive tests for Lock Manager components.

Tests cover:
1. VenueMarketLock: basic acquire/release, timeout, contention
2. AttemptLock: single-attempt enforcement, timeout
3. LockManager: order locks, attempt locks, metrics
"""

import asyncio
import pytest
import time

from src.locks import (
    VenueMarketLock, AttemptLock, LockManager,
    LockTimeoutError, LockError,
    create_lock_manager,
    DEFAULT_LOCK_TIMEOUT_SEC, DEFAULT_WARN_THRESHOLD_SEC,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def venue_market_lock():
    """Create VenueMarketLock for testing."""
    return VenueMarketLock("kalshi", "TEST-MKT", timeout_sec=1.0, warn_threshold_sec=0.2)


@pytest.fixture
def attempt_lock():
    """Create AttemptLock for testing."""
    return AttemptLock(timeout_sec=1.0, warn_threshold_sec=0.2)


@pytest.fixture
def lock_manager():
    """Create LockManager for testing."""
    return create_lock_manager(lock_timeout_sec=1.0, warn_threshold_sec=0.2)


# =============================================================================
# VENUE MARKET LOCK TESTS
# =============================================================================

class TestVenueMarketLock:
    """Tests for VenueMarketLock."""

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self, venue_market_lock):
        """Lock can be acquired and released."""
        assert not venue_market_lock.is_locked

        async with venue_market_lock.acquire():
            assert venue_market_lock.is_locked
            assert venue_market_lock.hold_time_sec is not None

        assert not venue_market_lock.is_locked
        assert venue_market_lock.hold_time_sec is None

    @pytest.mark.asyncio
    async def test_key_property(self, venue_market_lock):
        """Lock key returns (venue, market_id) tuple."""
        assert venue_market_lock.key == ("kalshi", "TEST-MKT")

    @pytest.mark.asyncio
    async def test_holder_tracking(self, venue_market_lock):
        """Holder is tracked when provided."""
        async with venue_market_lock.acquire(holder="test_holder"):
            assert venue_market_lock._holder == "test_holder"

        assert venue_market_lock._holder is None

    @pytest.mark.asyncio
    async def test_metrics_acquisition(self, venue_market_lock):
        """Acquisition increments metrics."""
        assert venue_market_lock.metrics.acquisitions == 0

        async with venue_market_lock.acquire():
            pass

        assert venue_market_lock.metrics.acquisitions == 1

    @pytest.mark.asyncio
    async def test_metrics_hold_time(self, venue_market_lock):
        """Hold time is tracked in metrics."""
        async with venue_market_lock.acquire():
            await asyncio.sleep(0.05)  # 50ms

        assert venue_market_lock.metrics.total_hold_time_ms >= 50
        assert venue_market_lock.metrics.max_hold_time_ms >= 50

    @pytest.mark.asyncio
    async def test_contention_tracking(self, venue_market_lock):
        """Contention is tracked when lock already held."""
        results = []

        async def holder():
            async with venue_market_lock.acquire():
                await asyncio.sleep(0.1)
                results.append("holder_done")

        async def waiter():
            await asyncio.sleep(0.01)  # Ensure holder goes first
            async with venue_market_lock.acquire():
                results.append("waiter_done")

        await asyncio.gather(holder(), waiter())

        assert results == ["holder_done", "waiter_done"]
        assert venue_market_lock.metrics.contentions == 1
        assert venue_market_lock.metrics.acquisitions == 2

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Lock times out when held too long."""
        lock = VenueMarketLock("kalshi", "TEST", timeout_sec=0.1)

        async def holder():
            async with lock.acquire():
                await asyncio.sleep(0.5)  # Hold longer than timeout

        async def waiter():
            await asyncio.sleep(0.01)  # Ensure holder goes first
            with pytest.raises(LockTimeoutError):
                async with lock.acquire():
                    pass

        await asyncio.gather(holder(), waiter(), return_exceptions=True)
        assert lock.metrics.timeouts == 1

    @pytest.mark.asyncio
    async def test_timeout_override(self):
        """Timeout can be overridden per-call."""
        lock = VenueMarketLock("kalshi", "TEST", timeout_sec=5.0)

        async def holder():
            async with lock.acquire():
                await asyncio.sleep(0.2)

        async def waiter():
            await asyncio.sleep(0.01)
            # Override to short timeout
            with pytest.raises(LockTimeoutError):
                async with lock.acquire(timeout_sec=0.05):
                    pass

        await asyncio.gather(holder(), waiter(), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_warns_on_slow_acquisition(self):
        """Warning yielded when acquisition exceeds threshold."""
        lock = VenueMarketLock("kalshi", "TEST", timeout_sec=1.0, warn_threshold_sec=0.05)

        async def holder():
            async with lock.acquire():
                await asyncio.sleep(0.1)

        async def waiter():
            await asyncio.sleep(0.01)
            async with lock.acquire() as warned:
                assert warned is True  # Should have triggered warning

        await asyncio.gather(holder(), waiter())

    @pytest.mark.asyncio
    async def test_reentrant_not_allowed(self, venue_market_lock):
        """Same coroutine cannot acquire lock twice (would deadlock)."""
        # This tests that we'd timeout if trying to reacquire
        lock = VenueMarketLock("kalshi", "TEST", timeout_sec=0.1)

        async with lock.acquire():
            with pytest.raises(LockTimeoutError):
                async with lock.acquire():
                    pass


# =============================================================================
# ATTEMPT LOCK TESTS
# =============================================================================

class TestAttemptLock:
    """Tests for AttemptLock."""

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self, attempt_lock):
        """Lock can be acquired and released."""
        assert not attempt_lock.is_locked
        assert attempt_lock.current_attempt_id is None

        async with attempt_lock.acquire("attempt_1"):
            assert attempt_lock.is_locked
            assert attempt_lock.current_attempt_id == "attempt_1"

        assert not attempt_lock.is_locked
        assert attempt_lock.current_attempt_id is None

    @pytest.mark.asyncio
    async def test_hold_time_tracking(self, attempt_lock):
        """Hold time is tracked."""
        async with attempt_lock.acquire("attempt_1"):
            await asyncio.sleep(0.05)
            assert attempt_lock.hold_time_sec is not None
            assert attempt_lock.hold_time_sec >= 0.05

        assert attempt_lock.hold_time_sec is None

    @pytest.mark.asyncio
    async def test_metrics(self, attempt_lock):
        """Metrics are tracked correctly."""
        async with attempt_lock.acquire("attempt_1"):
            await asyncio.sleep(0.01)

        assert attempt_lock.metrics.acquisitions == 1
        assert attempt_lock.metrics.total_hold_time_ms >= 10

    @pytest.mark.asyncio
    async def test_single_attempt_enforcement(self, attempt_lock):
        """Only one attempt can hold the lock at a time."""
        results = []

        async def attempt_one():
            async with attempt_lock.acquire("attempt_1"):
                results.append("attempt_1_start")
                await asyncio.sleep(0.1)
                results.append("attempt_1_end")

        async def attempt_two():
            await asyncio.sleep(0.01)  # Ensure attempt_1 starts first
            async with attempt_lock.acquire("attempt_2"):
                results.append("attempt_2_start")

        await asyncio.gather(attempt_one(), attempt_two())

        # attempt_2 should only start after attempt_1 ends
        assert results.index("attempt_1_end") < results.index("attempt_2_start")
        assert attempt_lock.metrics.contentions == 1

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Lock times out when another attempt holds too long."""
        lock = AttemptLock(timeout_sec=0.1)

        async def attempt_one():
            async with lock.acquire("attempt_1"):
                await asyncio.sleep(0.5)

        async def attempt_two():
            await asyncio.sleep(0.01)
            with pytest.raises(LockTimeoutError) as exc_info:
                async with lock.acquire("attempt_2"):
                    pass
            # Error message should mention current holder
            assert "attempt_1" in str(exc_info.value)

        await asyncio.gather(attempt_one(), attempt_two(), return_exceptions=True)
        assert lock.metrics.timeouts == 1

    @pytest.mark.asyncio
    async def test_warns_on_slow_acquisition(self):
        """Warning yielded when acquisition exceeds threshold."""
        lock = AttemptLock(timeout_sec=1.0, warn_threshold_sec=0.05)

        async def attempt_one():
            async with lock.acquire("attempt_1"):
                await asyncio.sleep(0.1)

        async def attempt_two():
            await asyncio.sleep(0.01)
            async with lock.acquire("attempt_2") as warned:
                assert warned is True

        await asyncio.gather(attempt_one(), attempt_two())


# =============================================================================
# LOCK MANAGER TESTS
# =============================================================================

class TestLockManager:
    """Tests for LockManager."""

    @pytest.mark.asyncio
    async def test_order_lock_acquisition(self, lock_manager):
        """Can acquire order lock for venue+market."""
        async with lock_manager.acquire_order_lock("kalshi", "TEST-MKT"):
            locks = lock_manager.active_order_locks
            assert ("kalshi", "TEST-MKT") in locks

        # Lock should be released
        assert ("kalshi", "TEST-MKT") not in lock_manager.active_order_locks

    @pytest.mark.asyncio
    async def test_order_lock_reuse(self, lock_manager):
        """Same lock is reused for same venue+market."""
        async with lock_manager.acquire_order_lock("kalshi", "TEST-MKT"):
            pass

        lock1 = lock_manager.get_order_lock("kalshi", "TEST-MKT")

        async with lock_manager.acquire_order_lock("kalshi", "TEST-MKT"):
            pass

        lock2 = lock_manager.get_order_lock("kalshi", "TEST-MKT")

        assert lock1 is lock2

    @pytest.mark.asyncio
    async def test_different_markets_independent(self, lock_manager):
        """Different markets have independent locks."""
        held = []

        async def acquire_mkt1():
            async with lock_manager.acquire_order_lock("kalshi", "MKT-1"):
                held.append("MKT-1")
                await asyncio.sleep(0.05)

        async def acquire_mkt2():
            async with lock_manager.acquire_order_lock("kalshi", "MKT-2"):
                held.append("MKT-2")
                await asyncio.sleep(0.05)

        await asyncio.gather(acquire_mkt1(), acquire_mkt2())

        # Both should be held simultaneously at some point
        # Since they're independent, order is non-deterministic
        assert len(held) == 2

    @pytest.mark.asyncio
    async def test_same_market_serialized(self, lock_manager):
        """Same market serializes acquisitions."""
        order = []

        async def first():
            async with lock_manager.acquire_order_lock("kalshi", "TEST"):
                order.append("first_start")
                await asyncio.sleep(0.05)
                order.append("first_end")

        async def second():
            await asyncio.sleep(0.01)  # Ensure first starts first
            async with lock_manager.acquire_order_lock("kalshi", "TEST"):
                order.append("second_start")

        await asyncio.gather(first(), second())

        assert order.index("first_end") < order.index("second_start")

    @pytest.mark.asyncio
    async def test_attempt_lock_acquisition(self, lock_manager):
        """Can acquire attempt lock."""
        assert lock_manager.current_attempt_id is None

        async with lock_manager.acquire_attempt_lock("attempt_1"):
            assert lock_manager.current_attempt_id == "attempt_1"

        assert lock_manager.current_attempt_id is None

    @pytest.mark.asyncio
    async def test_order_lock_metrics(self, lock_manager):
        """Order lock metrics are tracked."""
        async with lock_manager.acquire_order_lock("kalshi", "TEST"):
            pass

        assert lock_manager.metrics.order_lock_acquisitions == 1

    @pytest.mark.asyncio
    async def test_attempt_lock_metrics(self, lock_manager):
        """Attempt lock metrics are tracked."""
        async with lock_manager.acquire_attempt_lock("attempt_1"):
            pass

        assert lock_manager.metrics.attempt_lock_acquisitions == 1

    @pytest.mark.asyncio
    async def test_order_lock_timeout_metrics(self):
        """Order lock timeout increments metrics."""
        manager = create_lock_manager(lock_timeout_sec=0.1)

        async def holder():
            async with manager.acquire_order_lock("kalshi", "TEST"):
                await asyncio.sleep(0.5)

        async def waiter():
            await asyncio.sleep(0.01)
            with pytest.raises(LockTimeoutError):
                async with manager.acquire_order_lock("kalshi", "TEST"):
                    pass

        await asyncio.gather(holder(), waiter(), return_exceptions=True)
        assert manager.metrics.order_lock_timeouts == 1

    @pytest.mark.asyncio
    async def test_attempt_lock_timeout_metrics(self):
        """Attempt lock timeout increments metrics."""
        manager = create_lock_manager(lock_timeout_sec=0.1)

        async def holder():
            async with manager.acquire_attempt_lock("attempt_1"):
                await asyncio.sleep(0.5)

        async def waiter():
            await asyncio.sleep(0.01)
            with pytest.raises(LockTimeoutError):
                async with manager.acquire_attempt_lock("attempt_2"):
                    pass

        await asyncio.gather(holder(), waiter(), return_exceptions=True)
        assert manager.metrics.attempt_lock_timeouts == 1

    @pytest.mark.asyncio
    async def test_warning_metrics(self):
        """Warning emission is tracked in metrics."""
        manager = create_lock_manager(lock_timeout_sec=1.0, warn_threshold_sec=0.02)

        async def holder():
            async with manager.acquire_order_lock("kalshi", "TEST"):
                await asyncio.sleep(0.05)

        async def waiter():
            await asyncio.sleep(0.01)
            async with manager.acquire_order_lock("kalshi", "TEST") as warned:
                assert warned is True

        await asyncio.gather(holder(), waiter())
        assert manager.metrics.warnings_emitted >= 1

    @pytest.mark.asyncio
    async def test_holder_parameter(self, lock_manager):
        """Holder parameter is passed through."""
        async with lock_manager.acquire_order_lock(
            "kalshi", "TEST", holder="order_123"
        ):
            lock = lock_manager.get_order_lock("kalshi", "TEST")
            assert lock._holder == "order_123"

    @pytest.mark.asyncio
    async def test_timeout_override(self):
        """Timeout can be overridden per-call."""
        manager = create_lock_manager(lock_timeout_sec=5.0)

        async def holder():
            async with manager.acquire_order_lock("kalshi", "TEST"):
                await asyncio.sleep(0.2)

        async def waiter():
            await asyncio.sleep(0.01)
            with pytest.raises(LockTimeoutError):
                async with manager.acquire_order_lock(
                    "kalshi", "TEST", timeout_sec=0.05
                ):
                    pass

        await asyncio.gather(holder(), waiter(), return_exceptions=True)


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================

class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_many_concurrent_different_markets(self):
        """Many coroutines can acquire locks on different markets."""
        manager = create_lock_manager()
        results = []

        async def acquire_market(n):
            async with manager.acquire_order_lock("kalshi", f"MKT-{n}"):
                results.append(n)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[acquire_market(i) for i in range(10)])

        assert len(results) == 10
        # All 10 locks were created
        assert len(manager._order_locks) == 10

    @pytest.mark.asyncio
    async def test_mixed_venue_markets(self):
        """Locks are independent across venues."""
        manager = create_lock_manager()

        held_simultaneously = []

        async def kalshi_order():
            async with manager.acquire_order_lock("kalshi", "TEST"):
                held_simultaneously.append("kalshi")
                await asyncio.sleep(0.05)

        async def poly_order():
            async with manager.acquire_order_lock("poly", "TEST"):
                held_simultaneously.append("poly")
                await asyncio.sleep(0.05)

        await asyncio.gather(kalshi_order(), poly_order())

        # Both should have been held (order may vary)
        assert "kalshi" in held_simultaneously
        assert "poly" in held_simultaneously

    @pytest.mark.asyncio
    async def test_order_and_attempt_locks_independent(self):
        """Order locks and attempt lock are independent."""
        manager = create_lock_manager()

        async def order_op():
            async with manager.acquire_order_lock("kalshi", "TEST"):
                await asyncio.sleep(0.05)
                return "order"

        async def attempt_op():
            async with manager.acquire_attempt_lock("attempt_1"):
                await asyncio.sleep(0.05)
                return "attempt"

        results = await asyncio.gather(order_op(), attempt_op())

        assert set(results) == {"order", "attempt"}


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactory:
    """Tests for factory functions."""

    def test_create_lock_manager_defaults(self):
        """Factory uses default values."""
        manager = create_lock_manager()
        assert manager.lock_timeout_sec == DEFAULT_LOCK_TIMEOUT_SEC
        assert manager.warn_threshold_sec == DEFAULT_WARN_THRESHOLD_SEC

    def test_create_lock_manager_custom_values(self):
        """Factory accepts custom values."""
        manager = create_lock_manager(
            lock_timeout_sec=10.0,
            warn_threshold_sec=3.0
        )
        assert manager.lock_timeout_sec == 10.0
        assert manager.warn_threshold_sec == 3.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
