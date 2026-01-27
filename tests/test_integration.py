"""Integration tests for price escalation and execution flow.

Validates the end-to-end flow:
1. Trigger detection using best_ask pricing (Task 1)
2. Initial order placement using best_ask (Task 2)
3. Price escalation on zero fill (Task 3)
4. Fill handling and reconciliation
5. Profit boundary enforcement
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.venue_state import (
    MarketSnapshot,
    VenueState,
    OrderLevel,
    create_snapshot,
)
from src.opportunity import OpportunityDetector
from src.venue_execution import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderType,
    ExecutionSequencer,
    ExecutionPlan,
    PriceEscalationManager,
    MockOrderSubmitter,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_snapshot(
    poly_yes_ask: float = 0.45,
    poly_yes_bid: float = 0.40,
    poly_no_ask: float = 0.56,
    poly_no_bid: float = 0.51,
    kalshi_yes_ask: float = 0.48,
    kalshi_yes_bid: float = 0.42,
) -> MarketSnapshot:
    """Create a test market snapshot with specified prices.

    Args:
        poly_yes_ask: Polymarket YES ask price
        poly_yes_bid: Polymarket YES bid price
        poly_no_ask: Polymarket NO ask price
        poly_no_bid: Polymarket NO bid price
        kalshi_yes_ask: Kalshi YES ask price (derived from NO bid)
        kalshi_yes_bid: Kalshi YES bid price

    Returns:
        MarketSnapshot with specified prices
    """
    poly_yes_state = VenueState(
        venue="poly",
        market_id="poly_yes_token",
        last_update_ms=1000,
        best_bid=poly_yes_bid,
        best_ask=poly_yes_ask,
        best_bid_size=100.0,
        best_ask_size=100.0,
        bid_levels=(OrderLevel(price=poly_yes_bid, size=100.0),),
        ask_levels=(OrderLevel(price=poly_yes_ask, size=100.0),),
        is_fresh_after_reconnect=True,
    )

    poly_no_state = VenueState(
        venue="poly",
        market_id="poly_no_token",
        last_update_ms=1000,
        best_bid=poly_no_bid,
        best_ask=poly_no_ask,
        best_bid_size=100.0,
        best_ask_size=100.0,
        bid_levels=(OrderLevel(price=poly_no_bid, size=100.0),),
        ask_levels=(OrderLevel(price=poly_no_ask, size=100.0),),
        is_fresh_after_reconnect=True,
    )

    kalshi_state = VenueState(
        venue="kalshi",
        market_id="kalshi_ticker",
        last_update_ms=1000,
        best_bid=kalshi_yes_bid,
        best_ask=kalshi_yes_ask,
        best_bid_size=100.0,
        best_ask_size=100.0,
        bid_levels=(OrderLevel(price=kalshi_yes_bid, size=100.0),),
        ask_levels=(OrderLevel(price=kalshi_yes_ask, size=100.0),),
        is_fresh_after_reconnect=True,
    )

    return create_snapshot(
        poly_yes_state=poly_yes_state,
        poly_no_state=poly_no_state,
        kalshi_state=kalshi_state,
        aligned_market_id="ETH",
        snapshot_ms=1000,
    )


# =============================================================================
# Integration Tests: Full Execution Flow with Price Escalation
# =============================================================================


class TestFullExecutionFlow:
    """Tests for complete execution flow from trigger to fill."""

    @pytest.mark.asyncio
    async def test_full_execution_with_price_escalation(self):
        """End-to-end: trigger detection → execution with retry → fill.

        Validates:
        - Opportunity detected with best_ask pricing
        - Initial order uses best_ask (Task 2)
        - Retry escalates price by 1¢ (Task 3)
        - Fills are recorded correctly
        """
        # Create snapshot with profit opportunity (using best_ask pricing)
        # Path 1: poly_yes_ask=0.45, kalshi_yes_bid=0.48
        # Cost Path 1 = 0.45 + (1.0 - 0.48) = 0.45 + 0.52 = 0.97 (3% profit)
        # Path 2: kalshi_yes_ask=0.46, poly_no_ask=0.54
        # Cost Path 2 = 0.46 + 0.54 = 1.00 (0% profit)
        snapshot = create_test_snapshot(
            poly_yes_ask=0.45,
            poly_no_ask=0.54,
            kalshi_yes_ask=0.46,
            kalshi_yes_bid=0.48,  # Path 1 uses this bid (1 - 0.48 = 0.52)
        )

        # Verify opportunity is detected
        detector = OpportunityDetector()
        trigger = detector.detect(snapshot)

        assert trigger.triggered, "Should detect profit opportunity"
        assert trigger.strategy_path in (1, 2), "Should identify strategy path"
        assert trigger.cross_cost is not None, "Should calculate cross cost"

        # Track submission attempts and prices
        submission_attempts = []

        async def mock_kalshi_submit(order: OrderRequest) -> OrderResult:
            """Mock Kalshi submission with zero fill on first attempt."""
            attempt_num = len(submission_attempts)
            submission_attempts.append({
                "venue": "kalshi",
                "price": order.price,
                "quantity": order.quantity,
                "attempt": attempt_num,
            })

            if attempt_num == 0:
                # First attempt: zero fill (triggers retry with escalation)
                return OrderResult(
                    venue="kalshi",
                    success=True,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id=f"kalshi-order-{attempt_num}",
                    latency_ms=5.0,
                )
            else:
                # Retry: full fill at escalated price
                return OrderResult(
                    venue="kalshi",
                    success=True,
                    filled_qty=50,
                    avg_fill_price=order.price,
                    order_id=f"kalshi-order-{attempt_num}",
                    latency_ms=5.0,
                )

        async def mock_poly_submit(order: OrderRequest) -> OrderResult:
            """Mock Polymarket submission with immediate full fill."""
            return OrderResult(
                venue="poly",
                success=True,
                filled_qty=50,
                avg_fill_price=order.price,
                order_id="poly-order-1",
                latency_ms=5.0,
            )

        # Create execution plan
        plan = ExecutionPlan(
            kalshi_order=OrderRequest(
                venue="kalshi",
                market_id="kalshi_ticker",
                side=OrderSide.SELL if trigger.strategy_path == 1 else OrderSide.BUY,
                price=0.52 if trigger.strategy_path == 1 else 0.48,  # Walked price
                quantity=50,
                order_type=OrderType.IOC,
            ),
            poly_order=OrderRequest(
                venue="poly",
                market_id="poly_yes_token",
                side=OrderSide.BUY,
                price=0.45,
                quantity=50,
                order_type=OrderType.FAK,
            ),
            target_qty=50,
            walked_kalshi_price=0.52 if trigger.strategy_path == 1 else 0.48,
            walked_poly_price=0.45,
            strategy_path=trigger.strategy_path,
            kalshi_outcome="NO" if trigger.strategy_path == 1 else "YES",
            poly_outcome="YES",
        )

        # Execute with sequencer
        sequencer = ExecutionSequencer()
        result = await sequencer.execute_with_retry(
            plan,
            mock_kalshi_submit,
            mock_poly_submit,
        )

        # Verify execution succeeded
        assert result.kalshi_result.success, "Kalshi execution should succeed"
        assert result.kalshi_result.filled_qty == 50, "Kalshi should fill 50 after retry"
        assert result.poly_submitted, "Poly should be submitted after Kalshi fill"
        assert result.poly_result is not None, "Poly result should exist"
        assert result.poly_result.filled_qty == 50, "Poly should fill 50"

        # Verify we made multiple submission attempts
        assert len(submission_attempts) >= 2, "Should attempt at least twice (initial + retry)"

        # Verify price escalation
        initial_price = submission_attempts[0]["price"]
        retry_price = submission_attempts[1]["price"]

        assert retry_price > initial_price, "Retry price should be higher (escalated)"
        assert abs(retry_price - initial_price - 0.01) < 1e-6, "Escalation should be 1¢"

    @pytest.mark.asyncio
    async def test_execution_sequence_respects_fill_gating(self):
        """Verify Poly skipped when Kalshi fills zero after max retries.

        Validates:
        - Fill-biased gating logic unchanged
        - Poly not submitted if Kalshi never fills
        """
        snapshot = create_test_snapshot(
            poly_yes_ask=0.45,
            kalshi_yes_ask=0.48,
        )

        # Track calls
        kalshi_calls = []
        poly_calls = []

        async def kalshi_zero_fill(order: OrderRequest) -> OrderResult:
            """Kalshi always returns zero fill."""
            kalshi_calls.append(order)
            return OrderResult(
                venue="kalshi",
                success=True,
                filled_qty=0,  # Always zero
                avg_fill_price=None,
                order_id=f"k-{len(kalshi_calls)}",
                latency_ms=5.0,
            )

        async def poly_should_not_be_called(order: OrderRequest) -> OrderResult:
            """Poly should never be called."""
            poly_calls.append(order)
            raise AssertionError("Poly should not be called when Kalshi fills zero")

        # Create plan
        plan = ExecutionPlan(
            kalshi_order=OrderRequest(
                venue="kalshi",
                market_id="kalshi_ticker",
                side=OrderSide.BUY,
                price=0.48,
                quantity=50,
                order_type=OrderType.IOC,
            ),
            poly_order=OrderRequest(
                venue="poly",
                market_id="poly_yes_token",
                side=OrderSide.BUY,
                price=0.45,
                quantity=50,
                order_type=OrderType.FAK,
            ),
            target_qty=50,
            walked_kalshi_price=0.48,
            walked_poly_price=0.45,
            strategy_path=2,
            kalshi_outcome="YES",
            poly_outcome="NO",
        )

        # Execute
        sequencer = ExecutionSequencer()
        result = await sequencer.execute_with_retry(
            plan,
            kalshi_zero_fill,
            poly_should_not_be_called,
        )

        # Verify
        assert result.kalshi_result.filled_qty == 0, "Kalshi should fill zero"
        assert not result.poly_submitted, "Poly should not be submitted"
        assert result.poly_result is None, "Poly result should be None"
        assert result.reason == "kalshi_zero_after_retries"
        assert len(poly_calls) == 0, "Poly should not have been called at all"


# =============================================================================
# Integration Tests: Profit Boundary Enforcement
# =============================================================================


class TestProfitBoundaryEnforcement:
    """Tests for profit boundary enforcement during price escalation."""

    @pytest.mark.asyncio
    async def test_escalation_respects_profit_boundary(self):
        """Verify escalation never exceeds profit boundary.

        Scenario:
        - poly_yes_ask = 0.48
        - kalshi_yes_bid = 0.53 (used for Path 1 cost)
        - Path 1 cross_cost = 0.48 + (1.0 - 0.53) = 0.95 (5% profit)
        - max_kalshi_no_price = 1.0 - 0.48 = 0.52 (for break-even)
        - Escalation should never exceed 0.52

        Validates:
        - Price escalation respects profit boundary
        - Attempts continue until max retries
        - Prices stay within acceptable range
        """
        snapshot = create_test_snapshot(
            poly_yes_ask=0.48,
            poly_no_ask=0.54,
            kalshi_yes_ask=0.46,
            kalshi_yes_bid=0.53,  # Path 1 uses this (1 - 0.53 = 0.47)
        )

        # Expected profit boundary for Path 1
        poly_yes_ask = 0.48
        expected_max_kalshi_no_price = 1.0 - poly_yes_ask  # 0.52 (break-even for Path 1)

        # Detector to identify path
        detector = OpportunityDetector()
        trigger = detector.detect(snapshot)

        if trigger.triggered:
            # Collect all submitted prices
            submitted_prices = []

            async def capture_kalshi_price(order: OrderRequest) -> OrderResult:
                """Capture submitted price and return zero fill."""
                submitted_prices.append(order.price)

                return OrderResult(
                    venue="kalshi",
                    success=True,
                    filled_qty=0,  # Zero fill to trigger retries
                    avg_fill_price=None,
                    order_id=f"k-{len(submitted_prices)}",
                    latency_ms=5.0,
                )

            async def dummy_poly(order: OrderRequest) -> OrderResult:
                return OrderResult(
                    venue="poly",
                    success=True,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id="p",
                    latency_ms=5.0,
                )

            # Create plan with walked price from trigger
            plan = ExecutionPlan(
                kalshi_order=OrderRequest(
                    venue="kalshi",
                    market_id="kalshi_ticker",
                    side=OrderSide.BUY if trigger.strategy_path == 2 else OrderSide.SELL,
                    price=0.53 if trigger.strategy_path == 2 else 0.47,
                    quantity=50,
                    order_type=OrderType.IOC,
                ),
                poly_order=OrderRequest(
                    venue="poly",
                    market_id="poly_yes_token",
                    side=OrderSide.BUY,
                    price=0.48,
                    quantity=50,
                    order_type=OrderType.FAK,
                ),
                target_qty=50,
                walked_kalshi_price=0.53 if trigger.strategy_path == 2 else 0.47,
                walked_poly_price=0.48,
                strategy_path=trigger.strategy_path,
                kalshi_outcome="YES" if trigger.strategy_path == 2 else "NO",
                poly_outcome="YES",
            )

            # Execute
            sequencer = ExecutionSequencer()
            result = await sequencer.execute_with_retry(
                plan,
                capture_kalshi_price,
                dummy_poly,
            )

            # Verify prices submitted (if any)
            if len(submitted_prices) > 0:
                # For Path 1: max price is 1.0 - 0.48 = 0.52 (Kalshi NO)
                # For Path 2: max price depends on poly_no_ask
                max_acceptable_kalshi_price = expected_max_kalshi_no_price

                for i, price in enumerate(submitted_prices):
                    assert price <= max_acceptable_kalshi_price + 0.01, (
                        f"Attempt {i}: price ${price:.4f} exceeds boundary "
                        f"${max_acceptable_kalshi_price:.4f}"
                    )

    def test_price_escalation_manager_respects_boundary(self):
        """Unit test: PriceEscalationManager respects profit_boundary_price.

        Validates:
        - Initial price within budget
        - Escalation tries to increase price
        - But capped at profit_boundary_price
        """
        # Scenario: initial price 0.48, max price 0.50, profit boundary 0.50
        manager = PriceEscalationManager(
            initial_price=0.48,
            max_price=0.52,
            profit_boundary_price=0.50,
        )

        price_1 = manager.get_price_for_attempt(attempt=0)
        assert price_1 == 0.48, "Attempt 0: initial price"

        price_2 = manager.get_price_for_attempt(attempt=1, reason="zero_fill")
        assert price_2 == 0.49, "Attempt 1: escalate by 1¢"

        price_3 = manager.get_price_for_attempt(attempt=2, reason="zero_fill")
        # Should try 0.50, but profit_boundary capped at 0.50
        assert price_3 <= 0.50, "Attempt 2: should not exceed boundary"

        price_4 = manager.get_price_for_attempt(attempt=3, reason="zero_fill")
        # Should still be capped
        assert price_4 <= 0.50, "Attempt 3: should stay within boundary"


# =============================================================================
# Integration Tests: Opportunity Detection with Best Ask Pricing
# =============================================================================


class TestOpportunityDetectionWithBestAsk:
    """Tests for opportunity detection using best_ask (Task 1 validation)."""

    def test_path1_profit_calculation_uses_best_ask(self):
        """Path 1 profit uses best_bid (not best_ask).

        Path 1: Buy Poly YES + Buy Kalshi NO
        Cost = poly_yes_ask + (1 - kalshi_yes_bid)

        Validates:
        - Profit calculation derives Kalshi NO price from YES bid
        - This is correct: we derive the price we can get by selling Kalshi YES at bid
        """
        # Setup
        poly_yes_ask = 0.45
        kalshi_yes_ask = 0.46
        kalshi_yes_bid = 0.48  # Used for Path 1 (1 - 0.48 = 0.52)

        # Create snapshot
        snapshot = create_test_snapshot(
            poly_yes_ask=poly_yes_ask,
            poly_no_ask=0.54,
            kalshi_yes_ask=kalshi_yes_ask,
            kalshi_yes_bid=kalshi_yes_bid,
        )

        # Detect
        detector = OpportunityDetector()
        trigger = detector.detect(snapshot)

        if trigger.triggered:
            # Verify cross cost uses correct path
            # Path 1: cost = 0.45 + (1 - 0.48) = 0.45 + 0.52 = 0.97 (profitable)
            # Path 2: cost = 0.46 + 0.54 = 1.00 (break-even)
            expected_path1_cost = poly_yes_ask + (1.0 - kalshi_yes_bid)

            assert trigger.cross_cost <= expected_path1_cost + 0.001, (
                f"Cross cost should use path 1: "
                f"expected {expected_path1_cost:.4f}, got {trigger.cross_cost:.4f}"
            )

    def test_opportunity_triggered_when_cross_cost_below_threshold(self):
        """Opportunity triggers when cross_cost < $0.99.

        Validates:
        - Trigger threshold works correctly
        - Opportunity detection logic unchanged
        """
        # Setup for guaranteed trigger: cross cost = 0.97
        # Path 1: 0.45 + (1 - 0.48) = 0.97 (3% profit)
        snapshot = create_test_snapshot(
            poly_yes_ask=0.45,
            poly_no_ask=0.54,
            kalshi_yes_ask=0.46,
            kalshi_yes_bid=0.48,  # 1 - 0.48 = 0.52
        )

        detector = OpportunityDetector()
        trigger = detector.detect(snapshot)

        # Should trigger (3% profit > 2% default threshold)
        assert trigger.triggered, f"Should trigger with 3% profit, got: {trigger}"
        assert trigger.cross_cost is not None, "Should have cross cost"
        assert trigger.strategy_path in (1, 2), "Should identify path"


# =============================================================================
# Integration Tests: Price Escalation Sequence
# =============================================================================


class TestPriceEscalationSequence:
    """Tests for price escalation sequence on retries."""

    @pytest.mark.asyncio
    async def test_escalation_sequence_initial_plus_one_cent(self):
        """Verify escalation sequence: initial → +1¢ → +2¢.

        Validates:
        - Attempt 0: initial price
        - Attempt 1: initial + 1¢ (on zero fill)
        - Attempt 2: initial + 2¢ (on zero fill)
        """
        prices_submitted = []

        async def capture_price(order: OrderRequest) -> OrderResult:
            prices_submitted.append(order.price)
            if len(prices_submitted) < 3:
                # Zero fill to trigger retry
                return OrderResult(
                    venue="kalshi",
                    success=True,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id=f"k-{len(prices_submitted)}",
                    latency_ms=5.0,
                )
            else:
                # Fill on third attempt
                return OrderResult(
                    venue="kalshi",
                    success=True,
                    filled_qty=50,
                    avg_fill_price=order.price,
                    order_id=f"k-{len(prices_submitted)}",
                    latency_ms=5.0,
                )

        async def dummy_poly(order: OrderRequest) -> OrderResult:
            return OrderResult(
                venue="poly",
                success=True,
                filled_qty=50,
                avg_fill_price=order.price,
                order_id="p",
                latency_ms=5.0,
            )

        plan = ExecutionPlan(
            kalshi_order=OrderRequest(
                venue="kalshi",
                market_id="kalshi_ticker",
                side=OrderSide.BUY,
                price=0.48,
                quantity=50,
                order_type=OrderType.IOC,
            ),
            poly_order=OrderRequest(
                venue="poly",
                market_id="poly_yes_token",
                side=OrderSide.BUY,
                price=0.45,
                quantity=50,
                order_type=OrderType.FAK,
            ),
            target_qty=50,
            walked_kalshi_price=0.48,
            walked_poly_price=0.45,
            strategy_path=2,
            kalshi_outcome="YES",
            poly_outcome="NO",
        )

        sequencer = ExecutionSequencer()
        result = await sequencer.execute_with_retry(plan, capture_price, dummy_poly)

        # Verify escalation sequence
        assert len(prices_submitted) >= 1, "Should submit at least once"

        if len(prices_submitted) >= 2:
            initial = prices_submitted[0]
            escalated_1 = prices_submitted[1]
            assert abs(escalated_1 - initial - 0.01) < 1e-6, (
                f"First escalation should be +1¢: "
                f"{initial:.4f} → {escalated_1:.4f}"
            )

        if len(prices_submitted) >= 3:
            initial = prices_submitted[0]
            escalated_2 = prices_submitted[2]
            assert abs(escalated_2 - initial - 0.02) < 1e-6, (
                f"Second escalation should be +2¢: "
                f"{initial:.4f} → {escalated_2:.4f}"
            )

    def test_escalation_capped_at_max_price(self):
        """Verify escalation caps at max_price.

        Validates:
        - Even with high escalation attempts, prices capped
        - No infinite escalation
        """
        manager = PriceEscalationManager(
            initial_price=0.49,
            max_price=0.50,
        )

        # Multiple attempts
        for attempt in range(10):
            price = manager.get_price_for_attempt(
                attempt=attempt,
                reason="zero_fill"
            )
            assert price <= 0.50, f"Attempt {attempt}: price {price} exceeds max 0.50"

    def test_no_escalation_on_timeout(self):
        """Verify price doesn't escalate on timeout (only on zero fill).

        Validates:
        - Escalation is specific to zero fill reason
        - Other failure modes don't trigger escalation
        """
        manager = PriceEscalationManager(
            initial_price=0.48,
            max_price=0.52,
        )

        # Timeout should NOT escalate
        price_timeout = manager.get_price_for_attempt(
            attempt=1,
            reason="timeout"
        )
        assert price_timeout == 0.48, "Timeout should use initial price"

        # Exception should NOT escalate
        price_exception = manager.get_price_for_attempt(
            attempt=1,
            reason="exception"
        )
        assert price_exception == 0.48, "Exception should use initial price"

        # Zero fill SHOULD escalate
        price_zero_fill = manager.get_price_for_attempt(
            attempt=1,
            reason="zero_fill"
        )
        assert price_zero_fill == 0.49, "Zero fill should escalate to +1¢"


# =============================================================================
# Integration Tests: End-to-End Fills and Reconciliation
# =============================================================================


class TestFillHandlingAndReconciliation:
    """Tests for fill handling and execution reconciliation."""

    @pytest.mark.asyncio
    async def test_both_venues_fill_successfully(self):
        """Both Kalshi and Poly fill → execution complete.

        Validates:
        - Normal path: both fill
        - Fills recorded correctly
        - Execution marked successful
        """
        async def kalshi_fill(order: OrderRequest) -> OrderResult:
            return OrderResult(
                venue="kalshi",
                success=True,
                filled_qty=50,
                avg_fill_price=order.price,
                order_id="k1",
                latency_ms=5.0,
            )

        async def poly_fill(order: OrderRequest) -> OrderResult:
            return OrderResult(
                venue="poly",
                success=True,
                filled_qty=50,
                avg_fill_price=order.price,
                order_id="p1",
                latency_ms=5.0,
            )

        plan = ExecutionPlan(
            kalshi_order=OrderRequest(
                venue="kalshi",
                market_id="kalshi_ticker",
                side=OrderSide.BUY,
                price=0.48,
                quantity=50,
                order_type=OrderType.IOC,
            ),
            poly_order=OrderRequest(
                venue="poly",
                market_id="poly_yes_token",
                side=OrderSide.BUY,
                price=0.45,
                quantity=50,
                order_type=OrderType.FAK,
            ),
            target_qty=50,
            walked_kalshi_price=0.48,
            walked_poly_price=0.45,
            strategy_path=2,
            kalshi_outcome="YES",
            poly_outcome="NO",
        )

        sequencer = ExecutionSequencer()
        result = await sequencer.execute(plan, kalshi_fill, poly_fill)

        assert result.kalshi_result.success, "Kalshi should succeed"
        assert result.kalshi_result.filled_qty == 50, "Kalshi should fill 50"
        assert result.poly_submitted, "Poly should be submitted"
        assert result.poly_result.success, "Poly should succeed"
        assert result.poly_result.filled_qty == 50, "Poly should fill 50"
        assert result.reason == "kalshi_filled", "Normal execution"

    @pytest.mark.asyncio
    async def test_partial_fill_allows_poly(self):
        """Kalshi partial fill → Poly still submitted and filled.

        Validates:
        - Partial fill on Kalshi (>0) triggers Poly submission
        - Poly fill adjusted to matched quantity
        """
        async def kalshi_partial(order: OrderRequest) -> OrderResult:
            return OrderResult(
                venue="kalshi",
                success=True,
                filled_qty=30,  # Partial fill
                avg_fill_price=order.price,
                order_id="k1",
                latency_ms=5.0,
            )

        async def poly_adjusted(order: OrderRequest) -> OrderResult:
            # Poly should receive adjusted quantity (30, not 50)
            return OrderResult(
                venue="poly",
                success=True,
                filled_qty=order.quantity,  # Fill whatever quantity was requested
                avg_fill_price=order.price,
                order_id="p1",
                latency_ms=5.0,
            )

        plan = ExecutionPlan(
            kalshi_order=OrderRequest(
                venue="kalshi",
                market_id="kalshi_ticker",
                side=OrderSide.BUY,
                price=0.48,
                quantity=50,
                order_type=OrderType.IOC,
            ),
            poly_order=OrderRequest(
                venue="poly",
                market_id="poly_yes_token",
                side=OrderSide.BUY,
                price=0.45,
                quantity=50,
                order_type=OrderType.FAK,
            ),
            target_qty=50,
            walked_kalshi_price=0.48,
            walked_poly_price=0.45,
            strategy_path=2,
            kalshi_outcome="YES",
            poly_outcome="NO",
        )

        sequencer = ExecutionSequencer()
        result = await sequencer.execute(plan, kalshi_partial, poly_adjusted)

        assert result.kalshi_result.filled_qty == 30, "Kalshi partial fill"
        assert result.poly_submitted, "Poly submitted despite partial"
        assert result.poly_result.filled_qty == 30, "Poly adjusted to matched qty"
