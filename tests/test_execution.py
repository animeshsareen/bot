"""Tests for fill-biased execution sequencing module."""

import pytest
import asyncio
from src.venue_execution import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    ExecutionPlan,
    ExecutionResult,
    ExecutionSequencer,
    MockOrderSubmitter,
    PriceEscalationManager,
    create_execution_plan,
    DEFAULT_KALSHI_WAIT_MS,
    DEFAULT_HIGH_LATENCY_THRESHOLD_MS,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_kalshi_order(
    price: float = 0.50,
    quantity: int = 10,
) -> OrderRequest:
    """Create a Kalshi order request."""
    return OrderRequest(
        venue="kalshi",
        market_id="kalshi_ticker",
        side=OrderSide.BUY,
        price=price,
        quantity=quantity,
        order_type=OrderType.IOC,
    )


def make_poly_order(
    price: float = 0.48,
    quantity: int = 10,
) -> OrderRequest:
    """Create a Polymarket order request."""
    return OrderRequest(
        venue="poly",
        market_id="poly_token",
        side=OrderSide.BUY,
        price=price,
        quantity=quantity,
        order_type=OrderType.FAK,
    )


def make_execution_plan(
    target_qty: int = 10,
    kalshi_price: float = 0.50,
    poly_price: float = 0.48,
    strategy_path: int = 1,
) -> ExecutionPlan:
    """Create an execution plan."""
    return ExecutionPlan(
        kalshi_order=make_kalshi_order(kalshi_price, target_qty),
        poly_order=make_poly_order(poly_price, target_qty),
        target_qty=target_qty,
        walked_kalshi_price=kalshi_price,
        walked_poly_price=poly_price,
        strategy_path=strategy_path,
        kalshi_outcome="YES",
        poly_outcome="YES",
    )


# =============================================================================
# Tests for PriceEscalationManager
# =============================================================================


class TestPriceEscalationManager:
    """Tests for price escalation on retry."""

    def test_price_escalation_on_zero_fill(self):
        """Verify price escalates on zero fill retry."""
        manager = PriceEscalationManager(initial_price=0.48, max_price=0.50)

        price_1 = manager.get_price_for_attempt(attempt=0)
        assert price_1 == 0.48, f"Attempt 0 should return initial price 0.48, got {price_1}"

        price_2 = manager.get_price_for_attempt(attempt=1)
        assert price_2 == 0.49, f"Attempt 1 should escalate by 1 cent to 0.49, got {price_2}"

        price_3 = manager.get_price_for_attempt(attempt=2)
        assert price_3 == 0.50, f"Attempt 2 should escalate by 2 cents to 0.50, got {price_3}"

        price_4 = manager.get_price_for_attempt(attempt=3)
        assert price_4 == 0.50, f"Attempt 3+ should cap at max 0.50, got {price_4}"

    def test_price_escalation_respects_profit_boundary(self):
        """Verify escalated prices don't exceed profit boundary."""
        manager = PriceEscalationManager(
            initial_price=0.48,
            max_price=0.52,
            profit_boundary_price=0.50
        )

        price_3 = manager.get_price_for_attempt(attempt=2)
        assert price_3 <= 0.50, f"Should not exceed profit boundary 0.50, got {price_3}"

    def test_no_escalation_on_timeout(self):
        """Verify price doesn't escalate on timeout (only on zero fill)."""
        manager = PriceEscalationManager(initial_price=0.48)

        price_timeout = manager.get_price_for_attempt(attempt=1, reason="timeout")
        assert price_timeout == 0.48, f"Timeout should use initial price, not escalate"

        price_exception = manager.get_price_for_attempt(attempt=1, reason="exception")
        assert price_exception == 0.48, f"Exception should use initial price, not escalate"

        price_zero_fill = manager.get_price_for_attempt(attempt=1, reason="zero_fill")
        assert price_zero_fill == 0.49, f"Zero fill should escalate to 0.49"

    def test_capping_at_max_price(self):
        """Verify price caps at max_price."""
        manager = PriceEscalationManager(initial_price=0.49, max_price=0.50)

        price_1 = manager.get_price_for_attempt(attempt=1)
        assert price_1 == 0.50, "Should cap at max_price 0.50"

    def test_escalation_with_no_boundary(self):
        """Verify escalation works when profit_boundary_price is None."""
        manager = PriceEscalationManager(initial_price=0.48, max_price=1.00)

        price_2 = manager.get_price_for_attempt(attempt=1)
        assert price_2 == 0.49

        price_3 = manager.get_price_for_attempt(attempt=2)
        assert price_3 == 0.50


# =============================================================================
# Tests for OrderRequest
# =============================================================================


class TestOrderRequest:
    """Tests for OrderRequest dataclass."""

    def test_create_kalshi_order(self):
        """Create Kalshi IOC order."""
        order = make_kalshi_order()
        assert order.venue == "kalshi"
        assert order.order_type == OrderType.IOC

    def test_create_poly_order(self):
        """Create Polymarket FAK order."""
        order = make_poly_order()
        assert order.venue == "poly"
        assert order.order_type == OrderType.FAK

    def test_order_is_frozen(self):
        """OrderRequest is immutable."""
        order = make_kalshi_order()
        with pytest.raises(AttributeError):
            order.price = 0.55


# =============================================================================
# Tests for OrderResult
# =============================================================================


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_successful_fill(self):
        """Successful order with fill."""
        result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=10,
            avg_fill_price=0.50,
            order_id="order_123",
            latency_ms=25.0,
        )
        assert result.success
        assert result.filled_qty == 10
        assert result.error is None

    def test_partial_fill(self):
        """Order with partial fill."""
        result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=5,
            avg_fill_price=0.50,
            order_id="order_123",
            latency_ms=25.0,
        )
        assert result.success
        assert result.filled_qty == 5

    def test_zero_fill(self):
        """Order with zero fill."""
        result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=0,
            avg_fill_price=None,
            order_id="order_123",
            latency_ms=25.0,
        )
        assert result.success
        assert result.filled_qty == 0
        assert result.avg_fill_price is None

    def test_failed_order(self):
        """Failed order."""
        result = OrderResult(
            venue="kalshi",
            success=False,
            filled_qty=0,
            avg_fill_price=None,
            order_id=None,
            latency_ms=25.0,
            error="Connection timeout",
        )
        assert not result.success
        assert result.error == "Connection timeout"

    def test_result_is_frozen(self):
        """OrderResult is immutable."""
        result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=10,
            avg_fill_price=0.50,
            order_id="order_123",
            latency_ms=25.0,
        )
        with pytest.raises(AttributeError):
            result.filled_qty = 5


# =============================================================================
# Tests for ExecutionSequencer
# =============================================================================


class TestExecutionSequencer:
    """Tests for fill-biased execution sequencing."""

    @pytest.mark.asyncio
    async def test_normal_sequence_both_fill(self):
        """Normal sequence: Kalshi fills, then Poly fills."""
        sequencer = ExecutionSequencer()
        plan = make_execution_plan()

        kalshi_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=10)
        poly_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=10)

        result = await sequencer.execute(
            plan,
            kalshi_submitter.submit,
            poly_submitter.submit,
        )

        assert result.kalshi_result.filled_qty == 10
        assert result.poly_submitted
        assert result.poly_result.filled_qty == 10
        assert result.reason == "kalshi_filled"

    @pytest.mark.asyncio
    async def test_kalshi_zero_fill_skips_poly(self):
        """Poly skipped when Kalshi fills zero after retries."""
        # Use execute_with_retry which has explicit retry logic
        from src.venue_execution import ExecutionConfig
        config = ExecutionConfig(kalshi_wait_sec=0.5, kalshi_max_retries=1, kalshi_retry_wait_sec=0.1)
        sequencer = ExecutionSequencer(config=config)
        plan = make_execution_plan()

        kalshi_submitter = MockOrderSubmitter(fill_pct=0.0, latency_ms=10)
        poly_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=10)

        result = await sequencer.execute_with_retry(
            plan,
            kalshi_submitter.submit,
            poly_submitter.submit,
        )

        assert result.kalshi_result.filled_qty == 0
        assert not result.poly_submitted
        assert result.poly_result is None
        assert result.reason == "kalshi_zero_after_retries"
        # Poly should not have been called
        assert len(poly_submitter.submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_kalshi_partial_fill_submits_poly(self):
        """Poly submitted even on Kalshi partial fill."""
        sequencer = ExecutionSequencer(kalshi_wait_ms=1000)  # Increase wait time
        plan = make_execution_plan()

        kalshi_submitter = MockOrderSubmitter(fill_pct=0.5, latency_ms=10)
        poly_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=10)

        result = await sequencer.execute(
            plan,
            kalshi_submitter.submit,
            poly_submitter.submit,
        )

        assert result.kalshi_result.filled_qty == 5
        assert result.poly_submitted
        # Poly is submitted with matched quantity (5, not 10)
        assert result.poly_result.filled_qty == 5
        assert result.reason == "kalshi_filled"

    @pytest.mark.asyncio
    async def test_kalshi_submitted_first(self):
        """Kalshi is always submitted before Poly."""
        sequencer = ExecutionSequencer()
        plan = make_execution_plan()

        submission_order = []

        async def track_kalshi(order):
            submission_order.append("kalshi")
            return OrderResult(
                venue="kalshi",
                success=True,
                filled_qty=10,
                avg_fill_price=0.50,
                order_id="k1",
                latency_ms=5,
            )

        async def track_poly(order):
            submission_order.append("poly")
            return OrderResult(
                venue="poly",
                success=True,
                filled_qty=10,
                avg_fill_price=0.48,
                order_id="p1",
                latency_ms=5,
            )

        await sequencer.execute(plan, track_kalshi, track_poly)

        assert submission_order == ["kalshi", "poly"]


# =============================================================================
# Tests for create_execution_plan
# =============================================================================


class TestCreateExecutionPlan:
    """Tests for execution plan creation."""

    def test_creates_both_orders(self):
        """Creates orders for both venues."""
        plan = create_execution_plan(
            kalshi_market_id="ticker_123",
            poly_market_id="token_456",
            target_qty=10,
            walked_kalshi_price=0.50,
            walked_poly_price=0.48,
        )

        assert plan.kalshi_order.venue == "kalshi"
        assert plan.kalshi_order.market_id == "ticker_123"
        assert plan.kalshi_order.order_type == OrderType.IOC

        assert plan.poly_order.venue == "poly"
        assert plan.poly_order.market_id == "token_456"
        assert plan.poly_order.order_type == OrderType.FAK

    def test_prices_set_correctly(self):
        """Walked prices are used for orders."""
        plan = create_execution_plan(
            kalshi_market_id="ticker",
            poly_market_id="token",
            target_qty=10,
            walked_kalshi_price=0.52,
            walked_poly_price=0.46,
        )

        assert plan.kalshi_order.price == 0.52
        assert plan.poly_order.price == 0.46

    def test_strategy_path_1_sides(self):
        """Strategy path 1 uses SELL Kalshi NO, BUY Poly YES."""
        plan = create_execution_plan(
            kalshi_market_id="ticker",
            poly_market_id="token",
            target_qty=10,
            walked_kalshi_price=0.50,
            walked_poly_price=0.48,
            strategy_path=1,
        )

        assert plan.kalshi_order.side == OrderSide.SELL
        assert plan.poly_order.side == OrderSide.BUY
        assert plan.kalshi_outcome == "NO"
        assert plan.poly_outcome == "YES"

    def test_strategy_path_2_sides(self):
        """Strategy path 2 uses BUY Kalshi, BUY Poly NO."""
        plan = create_execution_plan(
            kalshi_market_id="ticker",
            poly_market_id="token",
            target_qty=10,
            walked_kalshi_price=0.50,
            walked_poly_price=0.48,
            strategy_path=2,
        )

        assert plan.kalshi_order.side == OrderSide.BUY
        assert plan.poly_order.side == OrderSide.BUY
        assert plan.kalshi_outcome == "YES"
        assert plan.poly_outcome == "NO"


# =============================================================================
# Tests for MockOrderSubmitter
# =============================================================================


class TestMockOrderSubmitter:
    """Tests for mock order submitter."""

    @pytest.mark.asyncio
    async def test_full_fill(self):
        """Full fill returns all quantity."""
        submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=5)
        order = make_kalshi_order(quantity=10)
        result = await submitter.submit(order)

        assert result.success
        assert result.filled_qty == 10

    @pytest.mark.asyncio
    async def test_partial_fill(self):
        """Partial fill returns fraction."""
        submitter = MockOrderSubmitter(fill_pct=0.5, latency_ms=5)
        order = make_kalshi_order(quantity=10)
        result = await submitter.submit(order)

        assert result.success
        assert result.filled_qty == 5

    @pytest.mark.asyncio
    async def test_zero_fill(self):
        """Zero fill returns 0."""
        submitter = MockOrderSubmitter(fill_pct=0.0, latency_ms=5)
        order = make_kalshi_order(quantity=10)
        result = await submitter.submit(order)

        assert result.success
        assert result.filled_qty == 0
        assert result.avg_fill_price is None

    @pytest.mark.asyncio
    async def test_simulated_failure(self):
        """Failure returns error."""
        submitter = MockOrderSubmitter(fail=True, latency_ms=5)
        order = make_kalshi_order()
        result = await submitter.submit(order)

        assert not result.success
        assert result.error == "Simulated failure"

    @pytest.mark.asyncio
    async def test_tracks_submitted_orders(self):
        """Tracks all submitted orders."""
        submitter = MockOrderSubmitter(latency_ms=1)
        order1 = make_kalshi_order()
        order2 = make_poly_order()

        await submitter.submit(order1)
        await submitter.submit(order2)

        assert len(submitter.submitted_orders) == 2
        assert submitter.submitted_orders[0] == order1
        assert submitter.submitted_orders[1] == order2


# =============================================================================
# Tests for ExecutionResult
# =============================================================================


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_result_is_frozen(self):
        """ExecutionResult is immutable."""
        kalshi_result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=10,
            avg_fill_price=0.50,
            order_id="k1",
            latency_ms=25,
        )
        result = ExecutionResult(
            kalshi_result=kalshi_result,
            poly_result=None,
            poly_submitted=False,
            execution_ms=30.0,
            reason="test",
        )
        with pytest.raises(AttributeError):
            result.poly_submitted = True

    def test_log_details(self, caplog):
        """Log details works correctly."""
        import logging
        caplog.set_level(logging.INFO)

        kalshi_result = OrderResult(
            venue="kalshi",
            success=True,
            filled_qty=10,
            avg_fill_price=0.50,
            order_id="k1",
            latency_ms=25,
        )
        poly_result = OrderResult(
            venue="poly",
            success=True,
            filled_qty=10,
            avg_fill_price=0.48,
            order_id="p1",
            latency_ms=15,
        )
        result = ExecutionResult(
            kalshi_result=kalshi_result,
            poly_result=poly_result,
            poly_submitted=True,
            execution_ms=50.0,
            reason="normal_sequence",
        )
        result.log_details()

        assert "EXECUTION" in caplog.text
        assert "kalshi_fill=10" in caplog.text


# =============================================================================
# Tests for default config values
# =============================================================================


class TestDefaultConfig:
    """Tests for default configuration values."""

    def test_default_kalshi_wait_ms(self):
        """Default Kalshi wait time (loosened from 150ms)."""
        assert DEFAULT_KALSHI_WAIT_MS == 500

    def test_default_high_latency_threshold(self):
        """Default high latency threshold (loosened from 100ms)."""
        assert DEFAULT_HIGH_LATENCY_THRESHOLD_MS == 200


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_execution_timing_recorded(self):
        """Execution time is recorded."""
        sequencer = ExecutionSequencer()
        plan = make_execution_plan()

        kalshi_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=5)
        poly_submitter = MockOrderSubmitter(fill_pct=1.0, latency_ms=5)

        result = await sequencer.execute(
            plan,
            kalshi_submitter.submit,
            poly_submitter.submit,
        )

        assert result.execution_ms > 0

    def test_execution_plan_is_frozen(self):
        """ExecutionPlan is immutable."""
        plan = make_execution_plan()
        with pytest.raises(AttributeError):
            plan.target_qty = 20
