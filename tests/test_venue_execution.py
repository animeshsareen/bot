"""Tests for venue-specific execution module (US-009, US-010)."""

import pytest
from src.venue_execution import (
    KalshiFillRecord,
    PolyFillRecord,
    DualFillRecord,
    FlattenTrigger,
    KalshiExecutor,
    PolyExecutor,
    create_kalshi_ioc_order,
    create_poly_fak_order,
    process_dual_execution,
    OrderResult,
    OrderSide,
    OrderType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_kalshi_result(
    filled_qty: int = 10,
    avg_price: float = 0.50,
    success: bool = True,
    error: str = None,
) -> OrderResult:
    """Create a Kalshi order result."""
    return OrderResult(
        venue="kalshi",
        success=success,
        filled_qty=filled_qty,
        avg_fill_price=avg_price if filled_qty > 0 else None,
        order_id="kalshi_123" if success else None,
        latency_ms=25.0,
        error=error,
    )


def make_poly_result(
    filled_qty: int = 10,
    avg_price: float = 0.48,
    success: bool = True,
    error: str = None,
) -> OrderResult:
    """Create a Polymarket order result."""
    return OrderResult(
        venue="poly",
        success=success,
        filled_qty=filled_qty,
        avg_fill_price=avg_price if filled_qty > 0 else None,
        order_id="poly_456" if success else None,
        latency_ms=20.0,
        error=error,
    )


# =============================================================================
# Tests for KalshiFillRecord (US-009)
# =============================================================================


class TestKalshiFillRecord:
    """Tests for Kalshi fill recording."""

    def test_record_filled_qty(self):
        """Records filled quantity."""
        record = KalshiFillRecord(
            filled_qty=10,
            avg_fill_price_cents=50,
            order_id="k123",
            success=True,
        )
        assert record.filled_qty == 10

    def test_record_avg_fill_price_cents(self):
        """Records average fill price in cents."""
        record = KalshiFillRecord(
            filled_qty=10,
            avg_fill_price_cents=52,
            order_id="k123",
            success=True,
        )
        assert record.avg_fill_price_cents == 52

    def test_price_conversion_to_dollars(self):
        """Converts cents to dollars correctly."""
        record = KalshiFillRecord(
            filled_qty=10,
            avg_fill_price_cents=50,
            order_id="k123",
            success=True,
        )
        assert record.avg_fill_price_dollars == 0.50

    def test_price_conversion_none_when_no_fill(self):
        """Price is None when no fills."""
        record = KalshiFillRecord(
            filled_qty=0,
            avg_fill_price_cents=None,
            order_id="k123",
            success=True,
        )
        assert record.avg_fill_price_dollars is None

    def test_record_is_frozen(self):
        """KalshiFillRecord is immutable."""
        record = KalshiFillRecord(
            filled_qty=10,
            avg_fill_price_cents=50,
            order_id="k123",
            success=True,
        )
        with pytest.raises(AttributeError):
            record.filled_qty = 5

    def test_log_details(self, caplog):
        """Log details works correctly."""
        import logging
        caplog.set_level(logging.INFO)

        record = KalshiFillRecord(
            filled_qty=10,
            avg_fill_price_cents=50,
            order_id="k123",
            success=True,
        )
        record.log_details()

        assert "KALSHI_FILL" in caplog.text
        assert "qty=10" in caplog.text
        assert "price_cents=50" in caplog.text


# =============================================================================
# Tests for KalshiExecutor (US-009)
# =============================================================================


class TestKalshiExecutor:
    """Tests for Kalshi IOC execution."""

    def test_process_successful_fill(self):
        """Processes successful fill."""
        executor = KalshiExecutor()
        result = make_kalshi_result(filled_qty=10, avg_price=0.52)
        fill_record, trigger = executor.process_result(result)

        assert fill_record.filled_qty == 10
        assert fill_record.avg_fill_price_cents == 52
        assert fill_record.success
        assert trigger == FlattenTrigger.NONE

    def test_no_retries_on_partial_fill(self):
        """No retries - records partial fill as-is."""
        executor = KalshiExecutor()
        result = make_kalshi_result(filled_qty=5, avg_price=0.50)
        fill_record, trigger = executor.process_result(result)

        # No retry, just record the partial
        assert fill_record.filled_qty == 5
        assert trigger == FlattenTrigger.NONE

    def test_flatten_trigger_when_poly_placed_and_zero_fill(self):
        """Triggers flatten if Poly placed and Kalshi fills zero."""
        executor = KalshiExecutor()
        result = make_kalshi_result(filled_qty=0, avg_price=None)
        fill_record, trigger = executor.process_result(
            result, poly_already_placed=True
        )

        assert fill_record.filled_qty == 0
        assert trigger == FlattenTrigger.KALSHI_ZERO_AFTER_POLY

    def test_no_flatten_when_poly_not_placed(self):
        """No flatten trigger if Poly not yet placed."""
        executor = KalshiExecutor()
        result = make_kalshi_result(filled_qty=0, avg_price=None)
        fill_record, trigger = executor.process_result(
            result, poly_already_placed=False
        )

        assert fill_record.filled_qty == 0
        assert trigger == FlattenTrigger.NONE

    def test_records_error_on_failure(self):
        """Records error message on failure."""
        executor = KalshiExecutor()
        result = make_kalshi_result(
            filled_qty=0, success=False, error="Connection timeout"
        )
        fill_record, _ = executor.process_result(result)

        assert not fill_record.success
        assert fill_record.error == "Connection timeout"


# =============================================================================
# Tests for PolyFillRecord (US-010)
# =============================================================================


class TestPolyFillRecord:
    """Tests for Polymarket fill recording."""

    def test_record_filled_qty(self):
        """Records filled quantity."""
        record = PolyFillRecord(
            filled_qty=10,
            avg_fill_price=0.48,
            order_id="p456",
            success=True,
        )
        assert record.filled_qty == 10

    def test_record_avg_fill_price(self):
        """Records average fill price in dollars."""
        record = PolyFillRecord(
            filled_qty=10,
            avg_fill_price=0.485,
            order_id="p456",
            success=True,
        )
        assert record.avg_fill_price == 0.485

    def test_no_resting_order_default(self):
        """Default has no resting order."""
        record = PolyFillRecord(
            filled_qty=10,
            avg_fill_price=0.48,
            order_id="p456",
            success=True,
        )
        assert not record.has_resting_order

    def test_record_is_frozen(self):
        """PolyFillRecord is immutable."""
        record = PolyFillRecord(
            filled_qty=10,
            avg_fill_price=0.48,
            order_id="p456",
            success=True,
        )
        with pytest.raises(AttributeError):
            record.filled_qty = 5

    def test_log_details(self, caplog):
        """Log details works correctly."""
        import logging
        caplog.set_level(logging.INFO)

        record = PolyFillRecord(
            filled_qty=10,
            avg_fill_price=0.48,
            order_id="p456",
            success=True,
        )
        record.log_details()

        assert "POLY_FILL" in caplog.text
        assert "qty=10" in caplog.text


# =============================================================================
# Tests for PolyExecutor (US-010)
# =============================================================================


class TestPolyExecutor:
    """Tests for Polymarket FAK execution."""

    def test_process_successful_fill(self):
        """Processes successful fill."""
        executor = PolyExecutor()
        result = make_poly_result(filled_qty=10, avg_price=0.48)
        fill_record = executor.process_result(result)

        assert fill_record.filled_qty == 10
        assert fill_record.avg_fill_price == 0.48
        assert fill_record.success

    def test_no_resting_order_for_fak(self):
        """FAK orders leave no resting orders."""
        executor = PolyExecutor()
        result = make_poly_result(filled_qty=5, avg_price=0.48)
        fill_record = executor.process_result(result)

        assert not fill_record.has_resting_order

    def test_zero_fill_no_resting(self):
        """Zero fill still has no resting order."""
        executor = PolyExecutor()
        result = make_poly_result(filled_qty=0, avg_price=None)
        fill_record = executor.process_result(result)

        assert fill_record.filled_qty == 0
        assert not fill_record.has_resting_order


# =============================================================================
# Tests for DualFillRecord
# =============================================================================


class TestDualFillRecord:
    """Tests for combined fill records."""

    def test_common_qty_both_fill(self):
        """Common qty is minimum of both fills."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        poly_fill = PolyFillRecord(
            filled_qty=8, avg_fill_price=0.48, order_id="p1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=poly_fill,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert record.common_qty == 8

    def test_common_qty_no_poly(self):
        """Common qty is 0 if Poly not submitted."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=None,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert record.common_qty == 0

    def test_kalshi_excess(self):
        """Kalshi excess when Kalshi fills more."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        poly_fill = PolyFillRecord(
            filled_qty=7, avg_fill_price=0.48, order_id="p1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=poly_fill,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert record.kalshi_excess == 3
        assert record.poly_excess == 0

    def test_poly_excess(self):
        """Poly excess when Poly fills more."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=5, avg_fill_price_cents=50, order_id="k1", success=True
        )
        poly_fill = PolyFillRecord(
            filled_qty=10, avg_fill_price=0.48, order_id="p1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=poly_fill,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert record.kalshi_excess == 0
        assert record.poly_excess == 5

    def test_needs_reconciliation_when_imbalanced(self):
        """Needs reconciliation when fills imbalanced."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        poly_fill = PolyFillRecord(
            filled_qty=7, avg_fill_price=0.48, order_id="p1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=poly_fill,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert record.needs_reconciliation

    def test_no_reconciliation_when_balanced(self):
        """No reconciliation when fills balanced."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        poly_fill = PolyFillRecord(
            filled_qty=10, avg_fill_price=0.48, order_id="p1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=poly_fill,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )

        assert not record.needs_reconciliation


# =============================================================================
# Tests for process_dual_execution
# =============================================================================


class TestProcessDualExecution:
    """Tests for processing dual execution results."""

    def test_balanced_fills_no_flatten(self):
        """Balanced fills don't trigger flatten."""
        kalshi_result = make_kalshi_result(filled_qty=10, avg_price=0.50)
        poly_result = make_poly_result(filled_qty=10, avg_price=0.48)

        record = process_dual_execution(kalshi_result, poly_result)

        assert record.flatten_trigger == FlattenTrigger.NONE
        assert not record.needs_reconciliation

    def test_imbalanced_fills_trigger_reconcile(self):
        """Imbalanced fills trigger reconciliation."""
        kalshi_result = make_kalshi_result(filled_qty=10, avg_price=0.50)
        poly_result = make_poly_result(filled_qty=7, avg_price=0.48)

        record = process_dual_execution(kalshi_result, poly_result)

        assert record.flatten_trigger == FlattenTrigger.IMBALANCED_FILLS
        assert record.needs_reconciliation

    def test_kalshi_zero_after_poly_triggers_flatten(self):
        """Kalshi zero fill after Poly triggers flatten."""
        kalshi_result = make_kalshi_result(filled_qty=0, avg_price=None)
        poly_result = make_poly_result(filled_qty=10, avg_price=0.48)

        record = process_dual_execution(
            kalshi_result, poly_result, poly_submitted_first=True
        )

        assert record.flatten_trigger == FlattenTrigger.KALSHI_ZERO_AFTER_POLY

    def test_no_poly_no_flatten(self):
        """No flatten if Poly not submitted."""
        kalshi_result = make_kalshi_result(filled_qty=0, avg_price=None)

        record = process_dual_execution(kalshi_result, None)

        assert record.flatten_trigger == FlattenTrigger.NONE
        assert record.poly_fill is None


# =============================================================================
# Tests for order creation helpers
# =============================================================================


class TestOrderCreation:
    """Tests for order creation helpers."""

    def test_create_kalshi_ioc_order(self):
        """Creates Kalshi IOC order with correct params."""
        order = create_kalshi_ioc_order(
            market_id="ticker_123",
            side=OrderSide.BUY,
            price_cents=50,
            quantity=10,
        )

        assert order.venue == "kalshi"
        assert order.market_id == "ticker_123"
        assert order.order_type == OrderType.IOC
        assert order.price == 0.50  # Converted to dollars
        assert order.quantity == 10

    def test_create_poly_fak_order(self):
        """Creates Polymarket FAK order with correct params."""
        order = create_poly_fak_order(
            market_id="token_456",
            side=OrderSide.BUY,
            price=0.48,
            quantity=10,
        )

        assert order.venue == "poly"
        assert order.market_id == "token_456"
        assert order.order_type == OrderType.FAK
        assert order.price == 0.48
        assert order.quantity == 10


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_both_zero_fills(self):
        """Both venues with zero fills."""
        kalshi_result = make_kalshi_result(filled_qty=0, avg_price=None)
        poly_result = make_poly_result(filled_qty=0, avg_price=None)

        record = process_dual_execution(kalshi_result, poly_result)

        assert record.common_qty == 0
        assert not record.needs_reconciliation  # Both zero is "balanced"

    def test_kalshi_fill_record_cents_precision(self):
        """Cents conversion handles precision."""
        executor = KalshiExecutor()
        # 0.525 -> 52 cents (truncated)
        result = make_kalshi_result(filled_qty=10, avg_price=0.525)
        fill_record, _ = executor.process_result(result)

        assert fill_record.avg_fill_price_cents == 52

    def test_dual_fill_record_frozen(self):
        """DualFillRecord is immutable."""
        kalshi_fill = KalshiFillRecord(
            filled_qty=10, avg_fill_price_cents=50, order_id="k1", success=True
        )
        record = DualFillRecord(
            kalshi_fill=kalshi_fill,
            poly_fill=None,
            poly_submitted_first=False,
            flatten_trigger=FlattenTrigger.NONE,
        )
        with pytest.raises(AttributeError):
            record.poly_submitted_first = True
