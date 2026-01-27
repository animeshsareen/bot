"""Tests for deterministic flatten safety valve (US-012)."""

import pytest
import asyncio
from src.flatten import (
    Flattener,
    FlattenResult,
    FlattenStatus,
    FlattenOrder,
    create_flatten_order,
    DEFAULT_FLATTEN_TIMEOUT_SEC,
    DEFAULT_FLATTEN_PRICE_BUFFER,
)
from src.venue_execution import OrderResult, OrderSide, OrderType
from src.position import Outcome


# =============================================================================
# Test Fixtures
# =============================================================================


class MockFlattenSubmitter:
    """Mock order submitter for flatten testing."""

    def __init__(
        self,
        fill_pct: float = 1.0,
        latency_ms: float = 10.0,
        fail: bool = False,
    ):
        self.fill_pct = fill_pct
        self.latency_ms = latency_ms
        self.fail = fail
        self.submitted_orders = []
        self.order_counter = 0

    async def submit(self, order) -> OrderResult:
        self.order_counter += 1
        self.submitted_orders.append(order)
        await asyncio.sleep(self.latency_ms / 1000)

        if self.fail:
            return OrderResult(
                venue=order.venue,
                success=False,
                filled_qty=0,
                avg_fill_price=None,
                order_id=None,
                latency_ms=self.latency_ms,
                error="Simulated failure",
            )

        filled = int(order.quantity * self.fill_pct)
        return OrderResult(
            venue=order.venue,
            success=True,
            filled_qty=filled,
            avg_fill_price=order.price if filled > 0 else None,
            order_id=f"flatten_{self.order_counter}",
            latency_ms=self.latency_ms,
        )


class MockCanceller:
    """Mock order canceller."""

    def __init__(self):
        self.cancelled_orders = []

    async def cancel(self, order_id: str) -> bool:
        self.cancelled_orders.append(order_id)
        return True


# =============================================================================
# Tests for Flattener
# =============================================================================


class TestFlattener:
    """Tests for flatten operation."""

    @pytest.mark.asyncio
    async def test_flatten_both_venues(self):
        """Flattens both venues to zero."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter(fill_pct=1.0)

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=8,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.status == FlattenStatus.COMPLETED
        assert result.outcome == Outcome.FLAT_FORCED
        assert result.final_kalshi_qty == 0
        assert result.final_poly_qty == 0
        assert result.is_flat

    @pytest.mark.asyncio
    async def test_flatten_kalshi_only(self):
        """Flattens only Kalshi when Poly is zero."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter(fill_pct=1.0)

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=0,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.status == FlattenStatus.COMPLETED
        assert result.outcome == Outcome.FLAT_FORCED
        assert result.final_kalshi_qty == 0
        assert result.kalshi_order is not None
        assert result.poly_order is None

    @pytest.mark.asyncio
    async def test_flatten_poly_only(self):
        """Flattens only Poly when Kalshi is zero."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter(fill_pct=1.0)

        result = await flattener.flatten(
            kalshi_qty=0,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.status == FlattenStatus.COMPLETED
        assert result.outcome == Outcome.FLAT_FORCED
        assert result.final_poly_qty == 0
        assert result.poly_order is not None
        assert result.kalshi_order is None

    @pytest.mark.asyncio
    async def test_flatten_nothing_when_already_flat(self):
        """No action when already flat."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()

        result = await flattener.flatten(
            kalshi_qty=0,
            poly_qty=0,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.status == FlattenStatus.COMPLETED
        assert result.outcome == Outcome.FLAT_FORCED
        assert len(submitter.submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_uses_taker_orders(self):
        """Uses IOC/FAK taker orders."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()

        await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        kalshi_order = next(o for o in submitter.submitted_orders if o.venue == "kalshi")
        poly_order = next(o for o in submitter.submitted_orders if o.venue == "poly")

        assert kalshi_order.order_type == OrderType.IOC
        assert poly_order.order_type == OrderType.FAK
        assert kalshi_order.side == OrderSide.SELL
        assert poly_order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Completes within 5s timeout."""
        flattener = Flattener(timeout_sec=5.0)
        submitter = MockFlattenSubmitter(latency_ms=100)

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.total_ms < 5000
        assert result.status == FlattenStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_timeout_when_too_slow(self):
        """Times out when operations too slow."""
        flattener = Flattener(timeout_sec=0.05)  # 50ms timeout
        submitter = MockFlattenSubmitter(latency_ms=100)  # 100ms per order

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.status == FlattenStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_cancellable(self):
        """Operations can be cancelled."""
        flattener = Flattener()

        async def slow_submit(order):
            await asyncio.sleep(0.05)
            flattener.request_cancel()
            await asyncio.sleep(0.05)
            return OrderResult(
                venue=order.venue,
                success=True,
                filled_qty=order.quantity,
                avg_fill_price=order.price,
                order_id="test",
                latency_ms=100,
            )

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=slow_submit,
        )

        assert result.status == FlattenStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_idempotent_multiple_calls(self):
        """Can be called multiple times safely."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()

        # First call
        result1 = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        # Second call on already flat position
        result2 = await flattener.flatten(
            kalshi_qty=0,
            poly_qty=0,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result1.status == FlattenStatus.COMPLETED
        assert result2.status == FlattenStatus.COMPLETED
        # Only first call should submit orders
        assert len(submitter.submitted_orders) == 2

    @pytest.mark.asyncio
    async def test_outcome_flat_forced(self):
        """Outcome is FLAT_FORCED on success."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.outcome == Outcome.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_cancels_outstanding_orders(self):
        """Cancels outstanding orders after flatten."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()
        canceller = MockCanceller()

        await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
            cancel_order=canceller.cancel,
        )

        assert len(canceller.cancelled_orders) > 0


# =============================================================================
# Tests for FlattenResult
# =============================================================================


class TestFlattenResult:
    """Tests for FlattenResult dataclass."""

    def test_is_flat_when_both_zero(self):
        """is_flat True when both venues zero."""
        result = FlattenResult(
            status=FlattenStatus.COMPLETED,
            outcome=Outcome.FLAT_FORCED,
            kalshi_order=None,
            poly_order=None,
            total_ms=50.0,
            final_kalshi_qty=0,
            final_poly_qty=0,
        )
        assert result.is_flat

    def test_not_flat_when_kalshi_remaining(self):
        """is_flat False when Kalshi has position."""
        result = FlattenResult(
            status=FlattenStatus.COMPLETED,
            outcome=None,
            kalshi_order=None,
            poly_order=None,
            total_ms=50.0,
            final_kalshi_qty=5,
            final_poly_qty=0,
        )
        assert not result.is_flat

    def test_result_is_frozen(self):
        """FlattenResult is immutable."""
        result = FlattenResult(
            status=FlattenStatus.COMPLETED,
            outcome=Outcome.FLAT_FORCED,
            kalshi_order=None,
            poly_order=None,
            total_ms=50.0,
            final_kalshi_qty=0,
            final_poly_qty=0,
        )
        with pytest.raises(AttributeError):
            result.status = FlattenStatus.FAILED

    def test_log_details(self, caplog):
        """Log details works correctly."""
        import logging
        caplog.set_level(logging.INFO)

        result = FlattenResult(
            status=FlattenStatus.COMPLETED,
            outcome=Outcome.FLAT_FORCED,
            kalshi_order=None,
            poly_order=None,
            total_ms=100.0,
            final_kalshi_qty=0,
            final_poly_qty=0,
        )
        result.log_details()

        assert "FLATTEN" in caplog.text
        assert "FLAT_FORCED" in caplog.text


# =============================================================================
# Tests for create_flatten_order
# =============================================================================


class TestCreateFlattenOrder:
    """Tests for flatten order creation."""

    def test_creates_kalshi_ioc(self):
        """Creates Kalshi IOC order."""
        order = create_flatten_order("kalshi", 10, 0.50)

        assert order.venue == "kalshi"
        assert order.order_type == OrderType.IOC
        assert order.side == OrderSide.SELL
        assert order.quantity == 10

    def test_creates_poly_fak(self):
        """Creates Poly FAK order."""
        order = create_flatten_order("poly", 10, 0.48)

        assert order.venue == "poly"
        assert order.order_type == OrderType.FAK
        assert order.side == OrderSide.SELL

    def test_price_below_bid(self):
        """Price is below bid for aggressive fill."""
        order = create_flatten_order("kalshi", 10, 0.50, price_buffer=0.05)

        assert order.price == 0.45  # 0.50 - 0.05

    def test_price_minimum_floor(self):
        """Price has minimum floor of 0.01."""
        order = create_flatten_order("kalshi", 10, 0.03, price_buffer=0.05)

        assert order.price == 0.01  # max(0.01, 0.03-0.05)


# =============================================================================
# Tests for FlattenOrder
# =============================================================================


class TestFlattenOrder:
    """Tests for FlattenOrder dataclass."""

    def test_order_is_frozen(self):
        """FlattenOrder is immutable."""
        order = FlattenOrder(
            venue="kalshi",
            quantity=10,
            price=0.45,
            filled_qty=10,
            order_id="test",
            latency_ms=25.0,
        )
        with pytest.raises(AttributeError):
            order.filled_qty = 5


# =============================================================================
# Tests for default config
# =============================================================================


class TestDefaultConfig:
    """Tests for default configuration."""

    def test_default_timeout(self):
        """Default timeout is 5 seconds."""
        assert DEFAULT_FLATTEN_TIMEOUT_SEC == 5.0

    def test_default_price_buffer(self):
        """Default price buffer is 5 cents."""
        assert DEFAULT_FLATTEN_PRICE_BUFFER == 0.05


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_partial_flatten(self):
        """Handles partial flatten."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter(fill_pct=0.5)

        result = await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        assert result.final_kalshi_qty == 5
        assert result.final_poly_qty == 5
        assert not result.is_flat

    @pytest.mark.asyncio
    async def test_reset_between_calls(self):
        """Flattener resets state between calls."""
        flattener = Flattener()
        submitter = MockFlattenSubmitter()

        # First call
        await flattener.flatten(
            kalshi_qty=10,
            poly_qty=10,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        # Request cancel after first call
        flattener.request_cancel()

        # Second call should reset
        result = await flattener.flatten(
            kalshi_qty=5,
            poly_qty=5,
            kalshi_bid=0.50,
            poly_bid=0.48,
            submit_order=submitter.submit,
        )

        # Should complete normally despite previous cancel request
        assert result.status == FlattenStatus.COMPLETED
