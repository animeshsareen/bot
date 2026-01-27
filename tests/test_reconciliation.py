"""Tests for chase-based reconciliation module (US-011)."""

import pytest
import asyncio
from src.reconciliation import (
    ChaseReconciler,
    ChaseStep,
    ReconciliationResult,
    ReconcileOutcome,
    ShortLeg,
    SimpleReconciler,
    identify_short_leg,
    compute_chase_price,
    compute_common_qty,
    DEFAULT_MAX_CHASE_STEPS,
    DEFAULT_MAX_HEDGE_RECONCILE_SEC,
    DEFAULT_CHASE_STEP_CENTS_KALSHI,
    DEFAULT_CHASE_STEP_DOLLARS_POLY,
)
from src.venue_execution import (
    DualFillRecord,
    KalshiFillRecord,
    PolyFillRecord,
    FlattenTrigger,
)
from src.venue_execution import OrderResult
from src.async_executor import ExecutorState


# =============================================================================
# Test Fixtures
# =============================================================================


def make_kalshi_fill(filled_qty: int, price_cents: int = 50) -> KalshiFillRecord:
    """Create a Kalshi fill record."""
    return KalshiFillRecord(
        filled_qty=filled_qty,
        avg_fill_price_cents=price_cents if filled_qty > 0 else None,
        order_id=f"k_{filled_qty}",
        success=True,
    )


def make_poly_fill(filled_qty: int, price: float = 0.48) -> PolyFillRecord:
    """Create a Poly fill record."""
    return PolyFillRecord(
        filled_qty=filled_qty,
        avg_fill_price=price if filled_qty > 0 else None,
        order_id=f"p_{filled_qty}",
        success=True,
    )


def make_dual_fill(
    kalshi_qty: int,
    poly_qty: int,
) -> DualFillRecord:
    """Create a dual fill record."""
    return DualFillRecord(
        kalshi_fill=make_kalshi_fill(kalshi_qty),
        poly_fill=make_poly_fill(poly_qty),
        poly_submitted_first=False,
        flatten_trigger=FlattenTrigger.NONE,
    )


class MockChaseSubmitter:
    """Mock order submitter for chase testing."""

    def __init__(
        self,
        fill_pct: float = 1.0,
        latency_ms: float = 5.0,
    ):
        self.fill_pct = fill_pct
        self.latency_ms = latency_ms
        self.submitted_orders = []
        self.order_counter = 0

    async def submit(self, order) -> OrderResult:
        self.order_counter += 1
        self.submitted_orders.append(order)
        await asyncio.sleep(self.latency_ms / 1000)

        filled = int(order.quantity * self.fill_pct)
        return OrderResult(
            venue=order.venue,
            success=True,
            filled_qty=filled,
            avg_fill_price=order.price if filled > 0 else None,
            order_id=f"chase_{self.order_counter}",
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
# Tests for compute_common_qty
# =============================================================================


class TestComputeCommonQty:
    """Tests for common quantity computation."""

    def test_equal_fills(self):
        """Common qty equals both when equal."""
        assert compute_common_qty(10, 10) == 10

    def test_kalshi_less(self):
        """Common qty is Kalshi when less."""
        assert compute_common_qty(7, 10) == 7

    def test_poly_less(self):
        """Common qty is Poly when less."""
        assert compute_common_qty(10, 5) == 5

    def test_both_zero(self):
        """Common qty is 0 when both zero."""
        assert compute_common_qty(0, 0) == 0


# =============================================================================
# Tests for identify_short_leg
# =============================================================================


class TestIdentifyShortLeg:
    """Tests for short leg identification."""

    def test_balanced_no_short(self):
        """No short leg when balanced."""
        leg, deficit = identify_short_leg(10, 10)
        assert leg == ShortLeg.NONE
        assert deficit == 0

    def test_kalshi_short(self):
        """Kalshi is short when fewer fills."""
        leg, deficit = identify_short_leg(7, 10)
        assert leg == ShortLeg.KALSHI
        assert deficit == 3

    def test_poly_short(self):
        """Poly is short when fewer fills."""
        leg, deficit = identify_short_leg(10, 6)
        assert leg == ShortLeg.POLY
        assert deficit == 4


# =============================================================================
# Tests for compute_chase_price
# =============================================================================


class TestComputeChasePrice:
    """Tests for chase price computation."""

    def test_kalshi_price_walk(self):
        """Kalshi walks by cents per step."""
        # Step 1: 0.50 + 0.01 = 0.51
        price = compute_chase_price(0.50, 1, "kalshi", chase_step_cents_kalshi=1)
        assert abs(price - 0.51) < 0.0001

    def test_kalshi_price_walk_step_2(self):
        """Kalshi step 2 walks further."""
        # Step 2: 0.50 + 0.02 = 0.52
        price = compute_chase_price(0.50, 2, "kalshi", chase_step_cents_kalshi=1)
        assert abs(price - 0.52) < 0.0001

    def test_poly_price_walk(self):
        """Poly walks by dollars per step."""
        # Step 1: 0.48 + 0.01 = 0.49
        price = compute_chase_price(0.48, 1, "poly", chase_step_dollars_poly=0.01)
        assert abs(price - 0.49) < 0.0001

    def test_poly_price_walk_step_3(self):
        """Poly step 3 walks further."""
        # Step 3: 0.48 + 0.03 = 0.51
        price = compute_chase_price(0.48, 3, "poly", chase_step_dollars_poly=0.01)
        assert abs(price - 0.51) < 0.0001


# =============================================================================
# Tests for ChaseReconciler
# =============================================================================


class TestChaseReconciler:
    """Tests for chase reconciliation."""

    @pytest.mark.asyncio
    async def test_no_chase_when_balanced(self):
        """No chase needed when fills balanced."""
        reconciler = ChaseReconciler()
        dual_fill = make_dual_fill(10, 10)
        submitter = MockChaseSubmitter()

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.outcome == ReconcileOutcome.NO_CHASE_NEEDED
        assert result.chase_steps_taken == 0
        assert len(submitter.submitted_orders) == 0

    @pytest.mark.asyncio
    async def test_chase_kalshi_short_leg(self):
        """Chases Kalshi when it's the short leg."""
        reconciler = ChaseReconciler(max_chase_steps=5)
        dual_fill = make_dual_fill(7, 10)  # Kalshi short by 3
        submitter = MockChaseSubmitter(fill_pct=1.0)

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.short_leg == ShortLeg.KALSHI
        assert result.initial_common_qty == 7
        assert result.final_common_qty == 10
        assert result.outcome == ReconcileOutcome.EQUALIZED
        assert all(o.venue == "kalshi" for o in submitter.submitted_orders)

    @pytest.mark.asyncio
    async def test_chase_poly_short_leg(self):
        """Chases Poly when it's the short leg."""
        reconciler = ChaseReconciler(max_chase_steps=5)
        dual_fill = make_dual_fill(10, 6)  # Poly short by 4
        submitter = MockChaseSubmitter(fill_pct=1.0)

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.short_leg == ShortLeg.POLY
        assert result.initial_common_qty == 6
        assert result.final_common_qty == 10
        assert result.outcome == ReconcileOutcome.EQUALIZED
        assert all(o.venue == "poly" for o in submitter.submitted_orders)

    @pytest.mark.asyncio
    async def test_rewalks_price_each_step(self):
        """Price increases each step."""
        reconciler = ChaseReconciler(
            max_chase_steps=5,
            chase_step_cents_kalshi=2,
        )
        dual_fill = make_dual_fill(5, 10)  # Kalshi short by 5
        submitter = MockChaseSubmitter(fill_pct=0.5)  # Partial fills

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        # Verify prices increase each step
        prices = [step.price for step in result.chase_steps]
        for i in range(1, len(prices)):
            assert prices[i] > prices[i - 1]

    @pytest.mark.asyncio
    async def test_stops_after_max_steps(self):
        """Stops after max_chase_steps."""
        reconciler = ChaseReconciler(max_chase_steps=3)
        dual_fill = make_dual_fill(0, 100)  # Kalshi very short
        submitter = MockChaseSubmitter(fill_pct=0.1)  # Very partial

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.chase_steps_taken == 3
        assert result.outcome == ReconcileOutcome.MAX_STEPS_REACHED

    @pytest.mark.asyncio
    async def test_stops_after_timeout(self):
        """Stops after max_reconcile_sec."""
        reconciler = ChaseReconciler(
            max_chase_steps=100,
            max_reconcile_sec=0.05,  # 50ms timeout
        )
        dual_fill = make_dual_fill(0, 100)
        submitter = MockChaseSubmitter(fill_pct=0.01, latency_ms=20)  # Slow

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.outcome == ReconcileOutcome.TIMEOUT

    @pytest.mark.asyncio
    async def test_cancels_outstanding_orders(self):
        """Cancels outstanding orders before returning."""
        reconciler = ChaseReconciler(max_chase_steps=3)
        dual_fill = make_dual_fill(5, 10)
        submitter = MockChaseSubmitter(fill_pct=0.5)
        canceller = MockCanceller()

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
            cancel_order=canceller.cancel,
        )

        assert result.orders_cancelled > 0
        assert len(canceller.cancelled_orders) > 0

    @pytest.mark.asyncio
    async def test_logs_chase_steps_and_final_qty(self):
        """Records chase steps and final common qty."""
        reconciler = ChaseReconciler(max_chase_steps=5)
        dual_fill = make_dual_fill(7, 10)
        submitter = MockChaseSubmitter(fill_pct=1.0)

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=submitter.submit,
        )

        assert result.chase_steps_taken >= 1
        assert len(result.chase_steps) == result.chase_steps_taken
        assert result.final_common_qty >= result.initial_common_qty


# =============================================================================
# Tests for ReconciliationResult
# =============================================================================


class TestReconciliationResult:
    """Tests for ReconciliationResult dataclass."""

    def test_result_is_frozen(self):
        """ReconciliationResult is immutable."""
        result = ReconciliationResult(
            outcome=ReconcileOutcome.EQUALIZED,
            initial_common_qty=7,
            final_common_qty=10,
            short_leg=ShortLeg.KALSHI,
            chase_steps_taken=1,
            chase_steps=(),
            total_ms=50.0,
        )
        with pytest.raises(AttributeError):
            result.outcome = ReconcileOutcome.PARTIAL

    def test_log_details(self, caplog):
        """Log details works correctly."""
        import logging
        caplog.set_level(logging.INFO)

        result = ReconciliationResult(
            outcome=ReconcileOutcome.EQUALIZED,
            initial_common_qty=7,
            final_common_qty=10,
            short_leg=ShortLeg.KALSHI,
            chase_steps_taken=2,
            chase_steps=(),
            total_ms=100.0,
        )
        result.log_details()

        assert "RECONCILE" in caplog.text
        assert "equalized" in caplog.text
        assert "steps=2" in caplog.text


# =============================================================================
# Tests for ChaseStep
# =============================================================================


class TestChaseStep:
    """Tests for ChaseStep dataclass."""

    def test_step_is_frozen(self):
        """ChaseStep is immutable."""
        step = ChaseStep(
            step_number=1,
            venue="kalshi",
            price=0.51,
            quantity_requested=3,
            quantity_filled=3,
            cumulative_filled=3,
            latency_ms=10.0,
        )
        with pytest.raises(AttributeError):
            step.quantity_filled = 5


# =============================================================================
# Tests for default config values
# =============================================================================


class TestDefaultConfig:
    """Tests for default configuration values."""

    def test_default_max_chase_steps(self):
        """Default max chase steps matches PRD."""
        assert DEFAULT_MAX_CHASE_STEPS == 5

    def test_default_max_reconcile_sec(self):
        """Default max reconcile time matches PRD."""
        assert DEFAULT_MAX_HEDGE_RECONCILE_SEC == 10

    def test_default_kalshi_step(self):
        """Default Kalshi price step matches PRD."""
        assert DEFAULT_CHASE_STEP_CENTS_KALSHI == 1

    def test_default_poly_step(self):
        """Default Poly price step matches PRD."""
        assert DEFAULT_CHASE_STEP_DOLLARS_POLY == 0.01


# =============================================================================
# Tests for cancellation
# =============================================================================


class TestCancellation:
    """Tests for reconciliation cancellation."""

    @pytest.mark.asyncio
    async def test_request_cancel_stops_chase(self):
        """Cancellation request stops chasing."""
        reconciler = ChaseReconciler(max_chase_steps=10)
        dual_fill = make_dual_fill(0, 100)

        call_count = 0

        async def slow_submit(order):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                reconciler.request_cancel()
            await asyncio.sleep(0.01)
            return OrderResult(
                venue=order.venue,
                success=True,
                filled_qty=1,
                avg_fill_price=order.price,
                order_id=f"o_{call_count}",
                latency_ms=10,
            )

        result = await reconciler.reconcile(
            dual_fill,
            current_kalshi_ask=0.50,
            current_poly_ask=0.48,
            submit_order=slow_submit,
        )

        assert result.outcome == ReconcileOutcome.CANCELLED
        assert result.chase_steps_taken <= 3


# =============================================================================
# Tests for SimpleReconciler
# =============================================================================


class TestSimpleReconciler:
    """Tests for simple post-execution reconciliation."""

    # =========================================================================
    # Case 1: Balanced fills (kalshi == poly > 0)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_case1_balanced_equal_fills(self):
        """Case 1: Equal balanced fills return DONE_EQUAL."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(10, 10)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case1_balanced_single_contract(self):
        """Case 1: Single contract each returns DONE_EQUAL."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(1, 1)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case1_balanced_large_fills(self):
        """Case 1: Very large balanced fills return DONE_EQUAL."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(10000, 10000)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case1_logs_info_for_balanced(self, caplog):
        """Case 1: Logs at INFO level for balanced fills."""
        import logging
        caplog.set_level(logging.INFO)

        reconciler = SimpleReconciler()
        await reconciler.reconcile_fills(5, 5)

        assert "fills balanced" in caplog.text
        assert "DONE_EQUAL" in caplog.text
        assert "kalshi=5" in caplog.text
        assert "poly=5" in caplog.text

    # =========================================================================
    # Case 2: Imbalanced with Poly having some (kalshi > poly > 0)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_case2_imbalanced_poly_short(self):
        """Case 2: Kalshi > Poly but Poly > 0 returns DONE_EQUAL with warning."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(15, 10)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case2_imbalanced_small_difference(self):
        """Case 2: Small imbalance (1 contract) returns DONE_EQUAL."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(11, 10)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case2_imbalanced_large_difference(self):
        """Case 2: Large imbalance but Poly > 0 returns DONE_EQUAL."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(100, 50)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_case2_logs_warning_for_imbalance(self, caplog):
        """Case 2: Logs WARNING for imbalanced fills."""
        import logging
        caplog.set_level(logging.WARNING)

        reconciler = SimpleReconciler()
        await reconciler.reconcile_fills(20, 15)

        assert "imbalanced fills" in caplog.text
        # Check log level in records
        assert any(record.levelname == "WARNING" for record in caplog.records)
        assert "DONE_EQUAL" in caplog.text
        assert "kalshi=20" in caplog.text
        assert "poly=15" in caplog.text
        assert "diff=5" in caplog.text

    # =========================================================================
    # Case 3: Unhedged position (kalshi > 0, poly == 0)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_case3_unhedged_position_kalshi_only(self):
        """Case 3: Kalshi filled but Poly zero returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(10, 0)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case3_unhedged_single_contract(self):
        """Case 3: Single contract Kalshi unhedged returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(1, 0)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case3_unhedged_large_position(self):
        """Case 3: Very large unhedged position returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(5000, 0)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case3_logs_error_for_unhedged(self, caplog):
        """Case 3: Logs ERROR for unhedged position."""
        import logging
        caplog.set_level(logging.ERROR)

        reconciler = SimpleReconciler()
        await reconciler.reconcile_fills(25, 0)

        assert "unhedged position" in caplog.text
        # Check log level in records
        assert any(record.levelname == "ERROR" for record in caplog.records)
        assert "FLAT_FORCED" in caplog.text
        assert "kalshi=25" in caplog.text
        assert "forcing_flatten" in caplog.text

    # =========================================================================
    # Case 4: Impossible state (poly > kalshi)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_case4_poly_exceeds_kalshi(self):
        """Case 4: Poly > Kalshi (impossible) returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(5, 15)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case4_poly_much_larger(self):
        """Case 4: Poly much larger than Kalshi returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(10, 100)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case4_kalshi_zero_poly_nonzero(self):
        """Case 4: Kalshi zero but Poly nonzero returns FLAT_FORCED."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(0, 5)
        assert result == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_case4_logs_error_for_impossible_state(self, caplog):
        """Case 4: Logs ERROR for impossible state."""
        import logging
        caplog.set_level(logging.ERROR)

        reconciler = SimpleReconciler()
        await reconciler.reconcile_fills(8, 20)

        assert "impossible state" in caplog.text
        # Check log level in records
        assert any(record.levelname == "ERROR" for record in caplog.records)
        assert "FLAT_FORCED" in caplog.text
        assert "poly_exceeds_kalshi" in caplog.text
        assert "poly=20" in caplog.text
        assert "kalshi=8" in caplog.text

    # =========================================================================
    # Edge cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_edge_case_both_zero(self):
        """Edge case: Both zero returns DONE_EQUAL with warning."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(0, 0)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_edge_case_both_zero_logs_warning(self, caplog):
        """Edge case: Both zero logs WARNING."""
        import logging
        caplog.set_level(logging.WARNING)

        reconciler = SimpleReconciler()
        await reconciler.reconcile_fills(0, 0)

        assert "both venues zero" in caplog.text
        assert "DONE_EQUAL" in caplog.text

    @pytest.mark.asyncio
    async def test_multiple_calls_independent(self):
        """Reconciler can be called multiple times."""
        reconciler = SimpleReconciler()

        result1 = await reconciler.reconcile_fills(10, 10)
        result2 = await reconciler.reconcile_fills(20, 0)
        result3 = await reconciler.reconcile_fills(5, 10)

        assert result1 == ExecutorState.DONE_EQUAL
        assert result2 == ExecutorState.FLAT_FORCED
        assert result3 == ExecutorState.FLAT_FORCED

    @pytest.mark.asyncio
    async def test_very_large_balanced_fills(self):
        """Test with very large numbers."""
        reconciler = SimpleReconciler()
        result = await reconciler.reconcile_fills(999999, 999999)
        assert result == ExecutorState.DONE_EQUAL

    @pytest.mark.asyncio
    async def test_no_state_mutation(self):
        """Reconciler doesn't mutate or maintain state."""
        reconciler = SimpleReconciler()

        # Call twice with different inputs
        result1 = await reconciler.reconcile_fills(10, 10)
        result2 = await reconciler.reconcile_fills(5, 0)

        # Same reconciler can be reused safely
        assert result1 == ExecutorState.DONE_EQUAL
        assert result2 == ExecutorState.FLAT_FORCED
