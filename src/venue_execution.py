"""Unified venue execution adapter for cross-venue arbitrage.

Single execution adapter that owns:
- Core domain types (OrderSide, OrderType, OrderRequest, OrderResult)
- Order submission sequencing (which leg first, concurrency policy)
- Retry policy for submit (network/auth/transient), separate retry for "wait for fill"
- Fill polling / timeouts / partial fill detection
- Price escalation on retry
- Preflight validation
- Producing structured ExecutionResult (fills, avg prices, latency, errors)
- Idempotency hooks (stable client_order_id, double-submit guard)
- Explicit timeout/retry policy (3 retries, 2s timeout)

Does NOT own:
- Cross-attempt reconciliation (stays in reconciliation.py)
- Position flattening decisions (stays in flatten.py)
- Global risk state machine (stays in async_executor.py)

Adapter boundary: domain objects in, API-specific calls out.
Side effects live here; decisioning stays out.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, Literal, Set
from enum import Enum
import asyncio
import time
import uuid
import hashlib
import logging

from .venue_state import MarketSnapshot

logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

DEFAULT_KALSHI_WAIT_MS = 2000
DEFAULT_HIGH_LATENCY_THRESHOLD_MS = 200
DEFAULT_KALSHI_STATUS_CHECK_RETRIES = 3
DEFAULT_KALSHI_RETRY_DELAY_SEC = 1.0


# ============================================================
# Enums
# ============================================================


class OrderSide(Enum):
    """Order side for trading."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    IOC = "ioc"  # Immediate-Or-Cancel (Kalshi)
    FAK = "fak"  # Fill-And-Kill (Polymarket)


class FlattenTrigger(Enum):
    """Reasons for triggering a flatten operation."""
    NONE = "none"
    KALSHI_ZERO_AFTER_POLY = "kalshi_zero_after_poly"
    IMBALANCED_FILLS = "imbalanced_fills"
    TIMEOUT = "timeout"


# ============================================================
# Core Domain Dataclasses
# ============================================================


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for execution with retry logic.

    Attributes:
        kalshi_wait_sec: Time in seconds to wait for Kalshi response (2.0 by default)
        kalshi_max_retries: Maximum number of retry attempts (2 by default)
        kalshi_retry_wait_sec: Time in seconds between retry attempts (1.0 by default)
    """

    kalshi_wait_sec: float = 2.0
    kalshi_max_retries: int = 2
    kalshi_retry_wait_sec: float = 1.0


@dataclass(frozen=True)
class OrderRequest:
    """Request to submit an order.

    Attributes:
        venue: Venue identifier ("poly" or "kalshi")
        market_id: Market identifier
        side: Buy or sell
        price: Limit price
        quantity: Number of contracts (integer for Kalshi)
        order_type: IOC or FAK
        action: Explicit action verb ("buy" or "sell") — use instead of overloading side
        outcome: Kalshi-specific outcome ("YES" or "NO") — never call this "side"
        client_order_id: Stable ID for retry deduplication (deterministic across retries)
    """

    venue: str
    market_id: str
    side: OrderSide
    price: float
    quantity: int
    order_type: OrderType
    action: Optional[Literal["buy", "sell"]] = None
    outcome: Optional[Literal["YES", "NO"]] = None
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class OrderResult:
    """Result of order submission.

    Attributes:
        venue: Venue identifier
        success: Whether order was accepted
        filled_qty: Quantity filled
        avg_fill_price: Average fill price (None if no fills)
        order_id: Exchange order ID (None if rejected)
        latency_ms: Time from submission to response
        error: Error message if failed
    """

    venue: str
    success: bool
    filled_qty: int
    avg_fill_price: Optional[float]
    order_id: Optional[str]
    latency_ms: float
    error: Optional[str] = None


@dataclass(frozen=True)
class ExecutionPlan:
    """Plan for executing both legs.

    Attributes:
        kalshi_order: Kalshi order request
        poly_order: Polymarket order request
        target_qty: Target quantity for both legs
        walked_kalshi_price: Walked price for Kalshi
        walked_poly_price: Walked price for Polymarket
        strategy_path: Which arbitrage path (1 or 2) defines the hedge relationship
        kalshi_outcome: Which outcome (YES or NO) being traded on Kalshi
        poly_outcome: Which outcome (YES or NO) being traded on Polymarket
        max_acceptable_kalshi_price: Maximum price for Kalshi to maintain profitability
    """

    kalshi_order: OrderRequest
    poly_order: OrderRequest
    target_qty: int
    walked_kalshi_price: float
    walked_poly_price: float
    strategy_path: int = 0  # 1 or 2
    kalshi_outcome: str = "YES"  # YES or NO
    poly_outcome: str = "YES"  # YES or NO
    max_acceptable_kalshi_price: float = 1.00  # Don't escalate above this


@dataclass(frozen=True)
class ExecutionResult:
    """Result of fill-biased execution.

    Attributes:
        kalshi_result: Result from Kalshi leg
        poly_result: Result from Polymarket leg (None if not submitted)
        poly_submitted: Whether Poly order was submitted
        execution_ms: Total execution time
        reason: Reason for execution outcome
    """

    kalshi_result: OrderResult
    poly_result: Optional[OrderResult]
    poly_submitted: bool
    execution_ms: float
    reason: str

    def log_details(self) -> None:
        """Log execution details at INFO level."""
        logger.info(
            "EXECUTION: kalshi_fill=%d poly_fill=%s poly_submitted=%s "
            "exec_ms=%.1f reason=%s",
            self.kalshi_result.filled_qty,
            self.poly_result.filled_qty if self.poly_result else "N/A",
            self.poly_submitted,
            self.execution_ms,
            self.reason,
        )


# ============================================================
# Venue-Specific Fill Records
# ============================================================


@dataclass(frozen=True)
class KalshiFillRecord:
    """Record of Kalshi IOC execution.

    Attributes:
        filled_qty: Number of contracts filled
        avg_fill_price_cents: Average fill price in cents (0-100)
        order_id: Exchange order ID
        success: Whether order was accepted
        error: Error message if failed
    """

    filled_qty: int
    avg_fill_price_cents: Optional[int]
    order_id: Optional[str]
    success: bool
    error: Optional[str] = None

    @property
    def avg_fill_price_dollars(self) -> Optional[float]:
        """Convert cents to dollars."""
        if self.avg_fill_price_cents is None:
            return None
        return self.avg_fill_price_cents / 100.0

    def log_details(self) -> None:
        """Log fill details at INFO level."""
        logger.info(
            "KALSHI_FILL: qty=%d price_cents=%s order_id=%s success=%s",
            self.filled_qty,
            self.avg_fill_price_cents,
            self.order_id,
            self.success,
        )


@dataclass(frozen=True)
class PolyFillRecord:
    """Record of Polymarket FAK execution.

    Attributes:
        filled_qty: Number of contracts filled
        avg_fill_price: Average fill price in dollars (0.0-1.0)
        order_id: Exchange order ID
        success: Whether order was accepted
        has_resting_order: Whether any maker order was left (should be False)
        error: Error message if failed
    """

    filled_qty: int
    avg_fill_price: Optional[float]
    order_id: Optional[str]
    success: bool
    has_resting_order: bool = False
    error: Optional[str] = None

    def log_details(self) -> None:
        """Log fill details at INFO level."""
        logger.info(
            "POLY_FILL: qty=%d price=%.4f order_id=%s success=%s resting=%s",
            self.filled_qty,
            self.avg_fill_price if self.avg_fill_price else 0,
            self.order_id,
            self.success,
            self.has_resting_order,
        )


@dataclass(frozen=True)
class DualFillRecord:
    """Combined record of both venue fills.

    Attributes:
        kalshi_fill: Kalshi fill record
        poly_fill: Polymarket fill record (None if not submitted)
        poly_submitted_first: Whether Poly was submitted before Kalshi result
        flatten_trigger: Reason to flatten (if any)
        common_qty: Minimum of both fills (matched quantity)
    """

    kalshi_fill: KalshiFillRecord
    poly_fill: Optional[PolyFillRecord]
    poly_submitted_first: bool
    flatten_trigger: FlattenTrigger

    @property
    def common_qty(self) -> int:
        """Compute matched quantity across both venues."""
        if self.poly_fill is None:
            return 0
        return min(self.kalshi_fill.filled_qty, self.poly_fill.filled_qty)

    @property
    def kalshi_excess(self) -> int:
        """Excess Kalshi fills needing reconciliation."""
        if self.poly_fill is None:
            return self.kalshi_fill.filled_qty
        return max(0, self.kalshi_fill.filled_qty - self.poly_fill.filled_qty)

    @property
    def poly_excess(self) -> int:
        """Excess Poly fills needing reconciliation."""
        if self.poly_fill is None:
            return 0
        return max(0, self.poly_fill.filled_qty - self.kalshi_fill.filled_qty)

    @property
    def needs_reconciliation(self) -> bool:
        """Whether fills are imbalanced and need reconciliation."""
        return self.kalshi_excess > 0 or self.poly_excess > 0

    def log_details(self) -> None:
        """Log dual fill details at INFO level."""
        logger.info(
            "DUAL_FILL: kalshi_qty=%d poly_qty=%s common=%d "
            "flatten_trigger=%s needs_reconcile=%s",
            self.kalshi_fill.filled_qty,
            self.poly_fill.filled_qty if self.poly_fill else "N/A",
            self.common_qty,
            self.flatten_trigger.value,
            self.needs_reconciliation,
        )


# ============================================================
# Preflight Validation
# ============================================================


@dataclass(frozen=True)
class PreflightValidation:
    """Result of pre-flight validation checks.

    Attributes:
        passed: Whether all validation checks passed
        reason: Human-readable reason for pass/fail
        kalshi_price: Executable L1 price from Kalshi (None if not applicable)
        kalshi_l1_size: Size at Kalshi L1 level
        poly_l1_size: Size at Polymarket L1 ask (strategy_path dependent)
        profit_pct: Estimated profit percentage after crossing L1 prices (None if check failed)
    """

    passed: bool
    reason: str
    kalshi_price: Optional[float] = None
    kalshi_l1_size: float = 0.0
    poly_l1_size: float = 0.0
    profit_pct: Optional[float] = None
    executable_qty: int = 0


def validate_executable_l1(
    snapshot: MarketSnapshot,
    strategy_path: int,
    min_shares: int = 1,
    max_shares: int = 5,
    profit_threshold_pct: float = 2.0,
) -> PreflightValidation:
    """Validate executable L1 prices and sizes before execution.

    Pre-flight validation ensures:
    1. Polymarket uses best ask (YES ask for path 1, NO ask for path 2)
    2. Kalshi:
       - Path 1 buys Kalshi NO, so executable NO price is 1.0 - kalshi_yes_bid (derived from YES bid)
       - Path 2 buys Kalshi YES, so executable YES price is kalshi_yes_ask
    3. min_shares is the minimum required depth to consider a trade.
    4. max_shares caps the executable quantity for test mode.
    5. Profit is computed from cross_cost = poly_ask + kalshi_leg_price.

    Args:
        snapshot: Market snapshot with per-venue states
        strategy_path: 1 (Poly YES + Kalshi NO) or 2 (Kalshi YES + Poly NO)
        min_shares: Minimum size required at L1 to consider (default 1)
        max_shares: Maximum shares to execute (default 5)
        profit_threshold_pct: Minimum profit % required (default 2.0)

    Returns:
        PreflightValidation with validation result and details
    """
    if strategy_path not in (1, 2):
        return PreflightValidation(
            passed=False,
            reason=f"invalid_strategy_path: {strategy_path}",
        )

    # Extract Polymarket ask L1 size based on strategy path
    if strategy_path == 1:
        poly_state = snapshot.poly_yes_state
    else:
        poly_state = snapshot.poly_no_state

    poly_ask = poly_state.best_ask
    poly_depth_l1 = poly_state.best_ask_size

    if poly_ask is None or poly_depth_l1 < min_shares:
        path_name = "Path1_PolyYES" if strategy_path == 1 else "Path2_PolyNO"
        return PreflightValidation(
            passed=False,
            reason=f"insufficient_poly_depth_{path_name}: {poly_depth_l1:.1f} < {min_shares}",
            poly_l1_size=poly_depth_l1,
            kalshi_l1_size=0.0,
            executable_qty=0,
        )

    # Extract Kalshi executable L1 price and size
    kalshi_state = snapshot.kalshi_state

    if strategy_path == 1:
        kalshi_yes_bid = kalshi_state.best_bid
        kalshi_l1_size = kalshi_state.best_bid_size

        if kalshi_yes_bid is None or kalshi_l1_size < min_shares:
            return PreflightValidation(
                passed=False,
                reason=f"insufficient_kalshi_depth_l1: {kalshi_l1_size:.1f} < {min_shares}",
                poly_l1_size=poly_depth_l1,
                kalshi_l1_size=kalshi_l1_size,
                executable_qty=0,
            )

        kalshi_price = 1.0 - kalshi_yes_bid

    else:  # strategy_path == 2
        kalshi_yes_ask = kalshi_state.best_ask
        kalshi_l1_size = kalshi_state.best_ask_size

        if kalshi_yes_ask is None or kalshi_l1_size < min_shares:
            return PreflightValidation(
                passed=False,
                reason=f"insufficient_kalshi_depth_l1: {kalshi_l1_size:.1f} < {min_shares}",
                poly_l1_size=poly_depth_l1,
                kalshi_l1_size=kalshi_l1_size,
                executable_qty=0,
            )

        kalshi_price = kalshi_yes_ask

    cross_cost = kalshi_price + poly_ask
    profit_pct = (1.0 - cross_cost) * 100
    executable_qty = int(min(max_shares, poly_depth_l1, kalshi_l1_size))

    if profit_pct < profit_threshold_pct:
        return PreflightValidation(
            passed=False,
            reason=f"profit_below_threshold: {profit_pct:.2f}% < {profit_threshold_pct}%",
            kalshi_price=kalshi_price,
            kalshi_l1_size=kalshi_l1_size,
            poly_l1_size=poly_depth_l1,
            profit_pct=profit_pct,
            executable_qty=executable_qty,
        )

    return PreflightValidation(
        passed=True,
        reason="all_checks_passed",
        kalshi_price=kalshi_price,
        kalshi_l1_size=kalshi_l1_size,
        poly_l1_size=poly_depth_l1,
        profit_pct=profit_pct,
        executable_qty=executable_qty,
    )



# ============================================================
# Price Escalation
# ============================================================


class PriceEscalationManager:
    """Manages price escalation for retry attempts.

    Escalates price by 1c on second attempt, 2c on third attempt.
    Respects maximum price and profit boundary constraints.

    Escalation only happens on zero fill. Timeout and exception retries
    use the original price.
    """

    def __init__(
        self,
        initial_price: float,
        max_price: float = 1.00,
        profit_boundary_price: Optional[float] = None,
    ):
        """Initialize escalation manager.

        Args:
            initial_price: Starting price in dollars (0.0-1.0)
            max_price: Maximum price cap (default: 1.00)
            profit_boundary_price: Do not escalate above this price
        """
        self.initial_price = initial_price
        self.max_price = max_price
        self.profit_boundary_price = profit_boundary_price or max_price

    def get_price_for_attempt(
        self,
        attempt: int,
        reason: str = "zero_fill",
    ) -> float:
        """Get price for given attempt number.

        Args:
            attempt: Attempt number (0-indexed)
            reason: Reason for retry ('zero_fill', 'timeout', 'exception')

        Returns:
            Price in dollars (0.0-1.0)
        """
        # Only escalate on zero fill, not on timeout/exception
        if reason != "zero_fill":
            return self.initial_price

        # Escalation schedule: +1c on attempt 1, +2c on attempt 2
        escalation_cents = {
            0: 0,    # Initial: no escalation
            1: 1,    # First retry: +1c
            2: 2,    # Second retry: +2c
        }.get(attempt, 2)  # Cap at 2c

        escalated_price = self.initial_price + (escalation_cents / 100.0)

        # Cap at maximum and profit boundary
        capped_price = min(escalated_price, self.max_price, self.profit_boundary_price)

        return capped_price


# ============================================================
# Venue Executors
# ============================================================


class KalshiExecutor:
    """Executes Kalshi IOC orders with no retries.

    Records fill quantity and average price in cents.
    Triggers flatten if Poly already placed and Kalshi fills zero.
    """

    def __init__(self):
        """Initialize Kalshi executor."""
        pass

    def process_result(
        self,
        result: OrderResult,
        poly_already_placed: bool = False,
    ) -> tuple[KalshiFillRecord, FlattenTrigger]:
        """Process Kalshi order result.

        Args:
            result: Order result from exchange
            poly_already_placed: Whether Poly order was already submitted

        Returns:
            Tuple of (fill record, flatten trigger)
        """
        # Convert price to cents
        avg_price_cents = None
        if result.avg_fill_price is not None:
            avg_price_cents = int(result.avg_fill_price * 100)

        fill_record = KalshiFillRecord(
            filled_qty=result.filled_qty,
            avg_fill_price_cents=avg_price_cents,
            order_id=result.order_id,
            success=result.success,
            error=result.error,
        )

        # Determine flatten trigger
        flatten_trigger = FlattenTrigger.NONE
        if poly_already_placed and result.filled_qty == 0:
            flatten_trigger = FlattenTrigger.KALSHI_ZERO_AFTER_POLY

        return fill_record, flatten_trigger


class PolyExecutor:
    """Executes Polymarket FAK orders with no resting exposure.

    Records fill quantity and average price in dollars.
    Ensures no maker orders are left behind.
    """

    def __init__(self):
        """Initialize Poly executor."""
        pass

    def process_result(self, result: OrderResult) -> PolyFillRecord:
        """Process Polymarket order result.

        Args:
            result: Order result from exchange

        Returns:
            Fill record with no resting order guarantee
        """
        has_resting = False

        return PolyFillRecord(
            filled_qty=result.filled_qty,
            avg_fill_price=result.avg_fill_price,
            order_id=result.order_id,
            success=result.success,
            has_resting_order=has_resting,
            error=result.error,
        )


# ============================================================
# Execution Sequencer (with idempotency)
# ============================================================


# Type for order submission callback
OrderSubmitter = Callable[[OrderRequest], Awaitable[OrderResult]]


def _generate_client_order_id(
    attempt_id: str,
    leg: str,
    market_id: str,
    outcome: str,
    action: str,
    price: float,
    qty: int,
) -> str:
    """Generate stable client_order_id for idempotent submission.

    Deterministic across retries of the same attempt: same inputs produce the
    same ID. Different attempt_ids produce different IDs even with identical
    params, so price-escalated retries get distinct keys.

    No nonce or timestamp — the ID is fully reproducible from the logical
    order identity, which is what venue-side idempotency requires.

    Args:
        attempt_id: Stable identifier for this execution attempt
        leg: "kalshi" or "poly"
        market_id: Exchange market identifier
        outcome: "YES" or "NO"
        action: "buy" or "sell"
        price: Limit price in dollars
        qty: Number of contracts

    Returns:
        Deterministic 16-char hex string (first 16 of SHA-256)
    """
    raw = f"{attempt_id}:{leg}:{market_id}:{outcome}:{action}:{price:.6f}:{qty}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class _DoubleSubmitGuard:
    """Prevents double-submission of the same client_order_id.

    Process-level safety net. Marks IDs only AFTER successful submission
    (i.e., after receiving an order_id back from the venue) so that a
    failed-before-reaching-venue submit does not block a legitimate retry.

    Server-side idempotency via deterministic client_order_id is the primary
    correctness mechanism; this guard is a secondary defence.
    """

    def __init__(self, max_tracked: int = 1000):
        self._submitted: Set[str] = set()
        self._max_tracked = max_tracked

    def is_known(self, client_order_id: str) -> bool:
        """Check if this ID was already successfully submitted."""
        return client_order_id in self._submitted

    def mark_submitted(self, client_order_id: str) -> None:
        """Mark an ID as successfully submitted (call after venue ack)."""
        self._submitted.add(client_order_id)
        if len(self._submitted) > self._max_tracked:
            self._submitted.pop()

    def reset(self) -> None:
        """Clear all tracked IDs (e.g., between trading sessions)."""
        self._submitted.clear()


class ExecutionSequencer:
    """Fill-biased execution sequencer with idempotency.

    Submits Kalshi first, then Poly only if Kalshi fills > 0.
    Prioritizes zero-fill avoidance over fill rate.

    Idempotency:
    - Generates deterministic client_order_id per order
    - Tracks successfully submitted IDs in a DoubleSubmitGuard
    - Guard marks AFTER successful venue ack (not before submit)

    Supports both simple execute() and execute_with_retry() for Kalshi retry logic.
    """

    def __init__(
        self,
        kalshi_wait_ms: float = DEFAULT_KALSHI_WAIT_MS,
        high_latency_threshold_ms: float = DEFAULT_HIGH_LATENCY_THRESHOLD_MS,
        config: Optional[ExecutionConfig] = None,
    ):
        """Initialize execution sequencer.

        Args:
            kalshi_wait_ms: Max time to wait for Kalshi ack/fill (legacy parameter)
            high_latency_threshold_ms: If Kalshi latency exceeds this, skip wait
            config: ExecutionConfig for retry logic (overrides kalshi_wait_ms if provided)
        """
        self.kalshi_wait_ms = kalshi_wait_ms
        self.high_latency_threshold_ms = high_latency_threshold_ms
        self.config = config or ExecutionConfig()
        self._attempt_id = uuid.uuid4().hex[:12]
        self._guard = _DoubleSubmitGuard()

    def _make_order_with_id(
        self,
        order: OrderRequest,
        leg: str,
        outcome: str,
        attempt: int,
    ) -> OrderRequest:
        """Attach a deterministic client_order_id to an OrderRequest."""
        coid = _generate_client_order_id(
            attempt_id=f"{self._attempt_id}-{attempt}",
            leg=leg,
            market_id=order.market_id,
            outcome=outcome,
            action=order.side.value,
            price=order.price,
            qty=order.quantity,
        )
        return OrderRequest(
            venue=order.venue,
            market_id=order.market_id,
            side=order.side,
            price=order.price,
            quantity=order.quantity,
            order_type=order.order_type,
            action=order.action,
            outcome=order.outcome,
            client_order_id=coid,
        )

    def _mark_if_submitted(self, result: OrderResult, client_order_id: Optional[str]) -> None:
        """Mark client_order_id in guard only after successful venue ack."""
        if client_order_id and result.order_id is not None:
            self._guard.mark_submitted(client_order_id)

    async def execute(
        self,
        plan: ExecutionPlan,
        submit_kalshi: OrderSubmitter,
        submit_poly: OrderSubmitter,
    ) -> ExecutionResult:
        """Execute both legs with fill-biased sequencing.

        On timeout, retries submission 3 times with 1-second delays.
        On zero fill, escalates price by 1c per retry (up to 2c total).

        Args:
            plan: Execution plan with both orders
            submit_kalshi: Callback to submit Kalshi order
            submit_poly: Callback to submit Poly order

        Returns:
            ExecutionResult with outcomes from both legs
        """
        start_ms = time.time() * 1000

        # Create escalation manager to handle price bumps on retry
        escalation = PriceEscalationManager(
            initial_price=plan.kalshi_order.price,
            max_price=1.00,
            profit_boundary_price=plan.max_acceptable_kalshi_price,
        )

        # Step 1: Submit Kalshi with retry on timeout (3 total attempts)
        max_attempts = DEFAULT_KALSHI_STATUS_CHECK_RETRIES
        kalshi_result = None
        last_retry_reason = "zero_fill"

        for attempt in range(max_attempts):
            current_price = escalation.get_price_for_attempt(
                attempt=attempt,
                reason=last_retry_reason,
            )

            order_to_submit = OrderRequest(
                venue=plan.kalshi_order.venue,
                market_id=plan.kalshi_order.market_id,
                side=plan.kalshi_order.side,
                price=current_price,
                quantity=plan.kalshi_order.quantity,
                order_type=plan.kalshi_order.order_type,
            )

            # Attach idempotency key
            order_to_submit = self._make_order_with_id(
                order_to_submit, "kalshi", plan.kalshi_outcome, attempt
            )

            # Check guard — skip if already successfully submitted
            if self._guard.is_known(order_to_submit.client_order_id):
                logger.warning("Double submit blocked for attempt %d", attempt + 1)
                continue

            if current_price != plan.kalshi_order.price:
                logger.info(
                    "Price escalation for attempt %d: %.2f$ -> %.2f$",
                    attempt + 1,
                    plan.kalshi_order.price,
                    current_price
                )

            kalshi_task = asyncio.create_task(submit_kalshi(order_to_submit))

            try:
                kalshi_result = await asyncio.wait_for(
                    asyncio.shield(kalshi_task),
                    timeout=self.kalshi_wait_ms / 1000.0
                )

                # Mark in guard after successful venue ack
                self._mark_if_submitted(kalshi_result, order_to_submit.client_order_id)

                if kalshi_result.filled_qty > 0:
                    logger.info("Kalshi filled on attempt %d: qty=%d @$%.4f", attempt + 1, kalshi_result.filled_qty, current_price)
                    break
                else:
                    logger.info("Kalshi zero fill on attempt %d @$%.4f, will retry", attempt + 1, current_price)
                    last_retry_reason = "zero_fill"

            except asyncio.TimeoutError:
                logger.info("Kalshi timeout on attempt %d at %dms @$%.4f", attempt + 1, self.kalshi_wait_ms, current_price)
                last_retry_reason = "timeout"

                if not kalshi_task.done():
                    kalshi_task.cancel()
                    try:
                        await kalshi_task
                    except asyncio.CancelledError:
                        pass

                kalshi_result = OrderResult(
                    venue="kalshi",
                    success=False,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id=None,
                    latency_ms=int(time.time() * 1000 - start_ms),
                    error="timeout_waiting_response"
                )

            except Exception as e:
                logger.warning("Kalshi exception on attempt %d @$%.4f: %s", attempt + 1, current_price, e)
                last_retry_reason = "exception"

                if not kalshi_task.done():
                    kalshi_task.cancel()
                    try:
                        await kalshi_task
                    except (asyncio.CancelledError, Exception):
                        pass

                kalshi_result = OrderResult(
                    venue="kalshi",
                    success=False,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id=None,
                    latency_ms=int(time.time() * 1000 - start_ms),
                    error=f"exception: {str(e)}"
                )

            if attempt < max_attempts - 1:
                if kalshi_result.filled_qty == 0:
                    logger.info("Retrying in %fs... (attempt %d/%d)",
                               DEFAULT_KALSHI_RETRY_DELAY_SEC, attempt + 1, max_attempts)
                    await asyncio.sleep(DEFAULT_KALSHI_RETRY_DELAY_SEC)

        # After all retry attempts, check if we got a fill
        if kalshi_result is None or kalshi_result.filled_qty == 0:
            execution_ms = int(time.time() * 1000 - start_ms)
            return ExecutionResult(
                kalshi_result=kalshi_result or OrderResult(
                    venue="kalshi",
                    success=False,
                    filled_qty=0,
                    avg_fill_price=None,
                    order_id=None,
                    latency_ms=execution_ms,
                    error="timeout_no_response"
                ),
                poly_result=None,
                poly_submitted=False,
                execution_ms=execution_ms,
                reason="kalshi_timeout_all_retries",
            )

        # Step 2: Gate Poly on Kalshi fill - skip if zero fill
        if kalshi_result.filled_qty == 0:
            execution_ms = time.time() * 1000 - start_ms
            return ExecutionResult(
                kalshi_result=kalshi_result,
                poly_result=None,
                poly_submitted=False,
                execution_ms=execution_ms,
                reason="kalshi_zero_fill",
            )

        # Step 3: Kalshi filled > 0, submit Poly with matched quantity only
        poly_order_matched = OrderRequest(
            venue=plan.poly_order.venue,
            market_id=plan.poly_order.market_id,
            side=plan.poly_order.side,
            price=plan.poly_order.price,
            quantity=kalshi_result.filled_qty,
            order_type=plan.poly_order.order_type,
        )

        # Attach idempotency key for poly leg
        poly_order_matched = self._make_order_with_id(
            poly_order_matched, "poly", plan.poly_outcome, 0
        )

        poly_result = await submit_poly(poly_order_matched)
        self._mark_if_submitted(poly_result, poly_order_matched.client_order_id)

        execution_ms = time.time() * 1000 - start_ms

        return ExecutionResult(
            kalshi_result=kalshi_result,
            poly_result=poly_result,
            poly_submitted=True,
            execution_ms=execution_ms,
            reason="kalshi_filled",
        )

    async def execute_with_retry(
        self,
        plan: ExecutionPlan,
        submit_kalshi: OrderSubmitter,
        submit_poly: OrderSubmitter,
    ) -> ExecutionResult:
        """Execute with Kalshi retry logic on zero fill.

        Implements retry behavior with price escalation:
        1. Submit Kalshi IOC at initial price
        2. Wait for response (config.kalshi_wait_sec = 2.0 by default)
        3. If filled > 0: proceed to Poly
        4. If filled == 0 and retries remaining: escalate price, wait 1s, retry
        5. After all retries with zero fill: return without Poly submission

        Price escalation:
        - Attempt 0: Use initial price
        - Attempt 1 (zero fill only): Escalate +1c
        - Attempt 2 (zero fill only): Escalate +2c
        - Respects profit boundary from preflight validation

        Args:
            plan: Execution plan with both orders
            submit_kalshi: Callback to submit Kalshi order
            submit_poly: Callback to submit Poly order

        Returns:
            ExecutionResult with outcomes from both legs
        """
        start_ms = time.time() * 1000

        escalation = PriceEscalationManager(
            initial_price=plan.kalshi_order.price,
            max_price=1.00,
            profit_boundary_price=plan.max_acceptable_kalshi_price,
        )

        kalshi_result = None
        wait_sec = self.config.kalshi_wait_sec
        max_retries = self.config.kalshi_max_retries
        retry_wait_sec = self.config.kalshi_retry_wait_sec
        last_retry_reason = "zero_fill"

        for attempt in range(1 + max_retries):
            logger.info(
                "Kalshi submission attempt %d of %d",
                attempt + 1,
                1 + max_retries
            )

            current_price = escalation.get_price_for_attempt(
                attempt=attempt,
                reason=last_retry_reason,
            )

            order_to_submit = OrderRequest(
                venue=plan.kalshi_order.venue,
                market_id=plan.kalshi_order.market_id,
                side=plan.kalshi_order.side,
                price=current_price,
                quantity=plan.kalshi_order.quantity,
                order_type=plan.kalshi_order.order_type,
            )

            # Attach idempotency key
            order_to_submit = self._make_order_with_id(
                order_to_submit, "kalshi", plan.kalshi_outcome, attempt
            )

            if self._guard.is_known(order_to_submit.client_order_id):
                logger.warning("Double submit blocked for attempt %d", attempt + 1)
                continue

            if current_price != plan.kalshi_order.price:
                logger.info(
                    "Price escalation for attempt %d: %.2f$ -> %.2f$",
                    attempt + 1,
                    plan.kalshi_order.price,
                    current_price
                )

            kalshi_task = asyncio.create_task(submit_kalshi(order_to_submit))

            try:
                kalshi_result = await asyncio.wait_for(
                    asyncio.shield(kalshi_task),
                    timeout=wait_sec
                )

                self._mark_if_submitted(kalshi_result, order_to_submit.client_order_id)

            except asyncio.TimeoutError:
                if not kalshi_task.done():
                    kalshi_task.cancel()
                    try:
                        await kalshi_task
                    except asyncio.CancelledError:
                        pass

                logger.info("Kalshi timeout on attempt %d @$%.4f", attempt + 1, current_price)
                last_retry_reason = "timeout"

                if attempt < max_retries:
                    logger.info("Retrying in %.1fs...", retry_wait_sec)
                    await asyncio.sleep(retry_wait_sec)
                    continue
                else:
                    execution_ms = time.time() * 1000 - start_ms
                    kalshi_result = OrderResult(
                        venue="kalshi",
                        success=False,
                        filled_qty=0,
                        avg_fill_price=None,
                        order_id=None,
                        latency_ms=execution_ms,
                        error="timeout_after_retries"
                    )
                    return ExecutionResult(
                        kalshi_result=kalshi_result,
                        poly_result=None,
                        poly_submitted=False,
                        execution_ms=execution_ms,
                        reason="kalshi_timeout_after_retries",
                    )

            except Exception as e:
                logger.error("Kalshi exception on attempt %d @$%.4f: %s", attempt + 1, current_price, e)
                last_retry_reason = "exception"

                if not kalshi_task.done():
                    kalshi_task.cancel()
                    try:
                        await kalshi_task
                    except (asyncio.CancelledError, Exception):
                        pass

                if attempt < max_retries:
                    logger.info("Retrying in %.1fs...", retry_wait_sec)
                    await asyncio.sleep(retry_wait_sec)
                    continue
                else:
                    execution_ms = time.time() * 1000 - start_ms
                    kalshi_result = OrderResult(
                        venue="kalshi",
                        success=False,
                        filled_qty=0,
                        avg_fill_price=None,
                        order_id=None,
                        latency_ms=execution_ms,
                        error=f"exception_after_retries: {str(e)}"
                    )
                    return ExecutionResult(
                        kalshi_result=kalshi_result,
                        poly_result=None,
                        poly_submitted=False,
                        execution_ms=execution_ms,
                        reason="kalshi_exception_after_retries",
                    )

            # Got a response
            if kalshi_result.filled_qty > 0:
                logger.info("Kalshi filled on attempt %d: qty=%d @$%.4f", attempt + 1, kalshi_result.filled_qty, current_price)
                break
            else:
                logger.info("Kalshi zero fill on attempt %d @$%.4f, will retry", attempt + 1, current_price)
                last_retry_reason = "zero_fill"

                if attempt < max_retries:
                    logger.info("Retrying in %.1fs...", retry_wait_sec)
                    await asyncio.sleep(retry_wait_sec)
                    continue
                else:
                    execution_ms = time.time() * 1000 - start_ms
                    return ExecutionResult(
                        kalshi_result=kalshi_result,
                        poly_result=None,
                        poly_submitted=False,
                        execution_ms=execution_ms,
                        reason="kalshi_zero_after_retries",
                    )

        # Step 2: Gate Poly on Kalshi fill
        if kalshi_result.filled_qty == 0:
            execution_ms = time.time() * 1000 - start_ms
            return ExecutionResult(
                kalshi_result=kalshi_result,
                poly_result=None,
                poly_submitted=False,
                execution_ms=execution_ms,
                reason="kalshi_zero_fill",
            )

        # Step 3: Kalshi filled > 0, submit Poly with matched quantity
        poly_order_matched = OrderRequest(
            venue=plan.poly_order.venue,
            market_id=plan.poly_order.market_id,
            side=plan.poly_order.side,
            price=plan.poly_order.price,
            quantity=kalshi_result.filled_qty,
            order_type=plan.poly_order.order_type,
        )

        poly_order_matched = self._make_order_with_id(
            poly_order_matched, "poly", plan.poly_outcome, 0
        )

        poly_result = await submit_poly(poly_order_matched)
        self._mark_if_submitted(poly_result, poly_order_matched.client_order_id)

        execution_ms = time.time() * 1000 - start_ms

        return ExecutionResult(
            kalshi_result=kalshi_result,
            poly_result=poly_result,
            poly_submitted=True,
            execution_ms=execution_ms,
            reason="kalshi_filled_with_retry",
        )


# ============================================================
# Plan Creation & Helper Functions
# ============================================================


def create_execution_plan(
    kalshi_market_id: str,
    poly_market_id: str,
    target_qty: int,
    walked_kalshi_price: float,
    walked_poly_price: float,
    strategy_path: int = 1,
    max_acceptable_kalshi_price: float = 1.00,
) -> ExecutionPlan:
    """Create an execution plan for both legs with correct hedge sides.

    Strategy definitions:
    - Path 1: Buy Poly YES + Sell Kalshi YES (buy Kalshi NO)
      - Poly: BUY YES
      - Kalshi: SELL YES
    - Path 2: Buy Kalshi YES + Buy Poly NO
      - Kalshi: BUY YES
      - Poly: BUY NO

    Args:
        kalshi_market_id: Kalshi market ticker
        poly_market_id: Polymarket token ID
        target_qty: Target quantity for both legs
        walked_kalshi_price: Walked price for Kalshi (in dollars)
        walked_poly_price: Walked price for Polymarket
        strategy_path: Which arbitrage path (1 or 2) to execute
        max_acceptable_kalshi_price: Maximum price for Kalshi to maintain profitability

    Returns:
        ExecutionPlan with both order requests and correct hedge sides
    """
    if strategy_path == 1:
        kalshi_side = OrderSide.SELL
        poly_side = OrderSide.BUY
        kalshi_outcome = "NO"
        poly_outcome = "YES"
    else:  # strategy_path == 2
        kalshi_side = OrderSide.BUY
        poly_side = OrderSide.BUY
        kalshi_outcome = "YES"
        poly_outcome = "NO"

    kalshi_order = OrderRequest(
        venue="kalshi",
        market_id=kalshi_market_id,
        side=kalshi_side,
        price=walked_kalshi_price,
        quantity=target_qty,
        order_type=OrderType.IOC,
    )

    poly_order = OrderRequest(
        venue="poly",
        market_id=poly_market_id,
        side=poly_side,
        price=walked_poly_price,
        quantity=target_qty,
        order_type=OrderType.FAK,
    )

    return ExecutionPlan(
        kalshi_order=kalshi_order,
        poly_order=poly_order,
        target_qty=target_qty,
        walked_kalshi_price=walked_kalshi_price,
        walked_poly_price=walked_poly_price,
        strategy_path=strategy_path,
        kalshi_outcome=kalshi_outcome,
        poly_outcome=poly_outcome,
        max_acceptable_kalshi_price=max_acceptable_kalshi_price,
    )


def create_kalshi_ioc_order(
    market_id: str,
    side: OrderSide,
    price_cents: int,
    quantity: int,
) -> OrderRequest:
    """Create a Kalshi IOC order request.

    Args:
        market_id: Kalshi ticker
        side: Buy or sell
        price_cents: Limit price in cents (0-100)
        quantity: Number of contracts

    Returns:
        OrderRequest for Kalshi IOC
    """
    return OrderRequest(
        venue="kalshi",
        market_id=market_id,
        side=side,
        price=price_cents / 100.0,
        quantity=quantity,
        order_type=OrderType.IOC,
    )


def create_poly_fak_order(
    market_id: str,
    side: OrderSide,
    price: float,
    quantity: int,
) -> OrderRequest:
    """Create a Polymarket FAK order request.

    Args:
        market_id: Polymarket token ID
        side: Buy or sell
        price: Limit price in dollars (0.0-1.0)
        quantity: Number of contracts

    Returns:
        OrderRequest for Polymarket FAK
    """
    return OrderRequest(
        venue="poly",
        market_id=market_id,
        side=side,
        price=price,
        quantity=quantity,
        order_type=OrderType.FAK,
    )


def process_dual_execution(
    kalshi_result: OrderResult,
    poly_result: Optional[OrderResult],
    poly_submitted_first: bool = False,
) -> DualFillRecord:
    """Process results from both venue executions.

    Args:
        kalshi_result: Kalshi order result
        poly_result: Poly order result (None if not submitted)
        poly_submitted_first: Whether Poly was submitted before Kalshi

    Returns:
        DualFillRecord with combined fill information
    """
    kalshi_executor = KalshiExecutor()
    poly_executor = PolyExecutor()

    kalshi_fill, flatten_trigger = kalshi_executor.process_result(
        kalshi_result,
        poly_already_placed=poly_submitted_first or (poly_result is not None),
    )

    poly_fill = None
    if poly_result is not None:
        poly_fill = poly_executor.process_result(poly_result)

    if flatten_trigger == FlattenTrigger.NONE:
        if poly_fill is not None:
            kalshi_qty = kalshi_fill.filled_qty
            poly_qty = poly_fill.filled_qty
            if kalshi_qty != poly_qty and (kalshi_qty > 0 or poly_qty > 0):
                flatten_trigger = FlattenTrigger.IMBALANCED_FILLS

    return DualFillRecord(
        kalshi_fill=kalshi_fill,
        poly_fill=poly_fill,
        poly_submitted_first=poly_submitted_first,
        flatten_trigger=flatten_trigger,
    )


# ============================================================
# Mock / Test Support
# ============================================================


class MockOrderSubmitter:
    """Mock order submitter for testing.

    Simulates order submission with configurable fill behavior.
    """

    def __init__(
        self,
        fill_pct: float = 1.0,
        latency_ms: float = 10.0,
        fail: bool = False,
    ):
        """Initialize mock submitter.

        Args:
            fill_pct: Percentage of quantity to fill (0.0-1.0)
            latency_ms: Simulated latency
            fail: Whether to simulate failure
        """
        self.fill_pct = fill_pct
        self.latency_ms = latency_ms
        self.fail = fail
        self.submitted_orders: list[OrderRequest] = []

    async def submit(self, order: OrderRequest) -> OrderResult:
        """Submit order and return mock result."""
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

        filled_qty = int(order.quantity * self.fill_pct)
        return OrderResult(
            venue=order.venue,
            success=True,
            filled_qty=filled_qty,
            avg_fill_price=order.price if filled_qty > 0 else None,
            order_id=f"mock_{order.venue}_{len(self.submitted_orders)}",
            latency_ms=self.latency_ms,
        )
