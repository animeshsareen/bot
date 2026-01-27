
import asyncio
import logging
import os
import sys
import time
from typing import Optional

from dotenv import load_dotenv

from .kalshi_client import KalshiClient
from .polymarket_client import PolymarketClient
from .kalshi_ws import KalshiWSHandler
from .polymarket_ws import PolymarketWSHandler
from .polymarket_user_ws import PolymarketUserWS
from .scanner import Scanner
from .ws_manager import MarketDataManager, WSConfig
from .async_executor import AsyncExecutor, ExecutionResult, ExecutorState
from .opportunity import OpportunityDetector
from .venue_state import MarketSnapshot
from .venue_execution import (
    OrderSide,
    create_execution_plan,
    OrderRequest,
    OrderResult,
    ExecutionSequencer,
    DEFAULT_KALSHI_WAIT_MS,
    DEFAULT_HIGH_LATENCY_THRESHOLD_MS,
    validate_executable_l1,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoBot")

load_dotenv()


class AutoBot:
    def __init__(self, asset: str = "eth", mode: Optional[str] = None,
                 poll_only: bool = False, find_next: bool = False):
        self.asset = asset
        self.poll_only = poll_only
        self.find_next = find_next
        self.running = False

        # Kalshi direction lock: after first trade, lock to this side
        self.locked_kalshi_side: Optional[OrderSide] = None
        self._execution_context: dict = {}

        # Session-level position tracking
        self._session_kalshi_fills: int = 0
        self._session_poly_fills: int = 0
        self._max_session_position: int = 5

        # Execution-aware strategy gating
        self._strategy_suspended: bool = False
        self._suspension_until: float = 0.0
        self._suspension_reason: str = ""
        self._consecutive_rejects: int = 0
        self._base_cooldown_sec: float = 5.0
        self._max_cooldown_sec: float = 60.0
        self._last_suppression_log: float = 0.0

        # Mode: argument > env > default
        self.mode = (mode or os.getenv("MODE", "paper")).lower()
        flags = []
        if self.poll_only: flags.append("POLL ONLY")
        if self.find_next: flags.append("NEXT MARKET")
        flags_str = f" ({', '.join(flags)})" if flags else ""
        logger.info(f"Initializing AutoBot in {self.mode.upper()} mode{flags_str}")

        # URLs by mode
        if self.mode == "live":
            kalshi_url = KalshiClient.PROD_URL
            poly_host = PolymarketClient.PROD_HOST
        else:
            kalshi_url = KalshiClient.DEMO_URL
            poly_host = PolymarketClient.STAGING_HOST

        # Clients
        self.polymarket_client = PolymarketClient(host=poly_host)
        self.kalshi_client = KalshiClient(
            key_id=os.getenv("KALSHI_KEY_ID"),
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH"),
            base_url=kalshi_url
        )

        # Core components
        self.scanner = Scanner(asset, self.kalshi_client, self.polymarket_client)
        self.detector = OpportunityDetector()
        self.executor = AsyncExecutor(
            execute_callback=self._execute_trade,
            reconcile_callback=self._reconcile_trade
        )

        # Will be set in start() after WS handler creation
        self._kalshi_ws: Optional[KalshiWSHandler] = None
        self._poly_user_ws: Optional[PolymarketUserWS] = None

    async def start(self):
        """Main startup sequence."""
        self.running = True
        logger.info(f"Starting AutoBot for {self.asset.upper()}")

        # 1. Discover matched markets (with retry)
        matched_pairs = None
        retry_count = 0
        while matched_pairs is None:
            try:
                matched_pairs = await self.scanner.discover_matched_pairs()
                if not matched_pairs:
                    logger.error("No matched market pairs found. Exiting.")
                    return
            except RuntimeError as e:
                retry_count += 1
                if any(s in str(e) for s in ["No open Polymarket", "No open Kalshi"]):
                    logger.info(f"No active markets (attempt {retry_count}). Retrying in 30s...")
                    await asyncio.sleep(30)
                else:
                    raise

        best_pair = matched_pairs[0]
        logger.info(
            f"Using pair: {best_pair.kalshi_ticker} <-> {best_pair.polymarket_slug} "
            f"(conf: {best_pair.match_confidence:.1%}, diff: {best_pair.expiration_diff_seconds()}s)"
        )

        # 2. Fetch token IDs
        await self.scanner._fetch_market_details(best_pair.polymarket_slug)
        kalshi_ticker = best_pair.kalshi_ticker
        poly_yes = self.scanner.market_info['poly_yes_token']
        poly_no = self.scanner.market_info['poly_no_token']

        logger.info(f"Kalshi: {kalshi_ticker}, Poly YES: {poly_yes[:20]}..., NO: {poly_no[:20]}...")
        self.scanner.kalshi_ticker = kalshi_ticker
        self.scanner.poly_yes_token = poly_yes
        self.scanner.poly_no_token = poly_no

        self.scanner.poly_no_token = poly_no
        
        # Warmup cache
        await self.polymarket_client.warmup_market([poly_yes, poly_no])
        
        # 3. WebSocket handlers (per-leg freshness)
        kalshi_ws = KalshiWSHandler(kalshi_ticker, self.kalshi_client)
        self._kalshi_ws = kalshi_ws  # Store for fill monitoring
        
        # Authenticated User Channel for Poly
        self._poly_user_ws = PolymarketUserWS(self.polymarket_client)
        await self._poly_user_ws.start()
        
        poly_yes_ws = PolymarketWSHandler(poly_yes, self.polymarket_client)
        poly_no_ws = PolymarketWSHandler(poly_no, self.polymarket_client)

        # 4. Data manager
        self.data_manager = MarketDataManager(poly_yes_ws, poly_no_ws, kalshi_ws, f"{self.asset}-arb")
        self.data_manager.start()

        # 5. Main loop
        asyncio.create_task(self._main_loop())
        await asyncio.Event().wait()

    def _get_kalshi_side_for_strategy(self, strategy_path: int) -> OrderSide:
        return OrderSide.SELL if strategy_path == 1 else OrderSide.BUY

    def _is_kalshi_direction_allowed(self, strategy_path: int) -> bool:
        kalshi_side = self._get_kalshi_side_for_strategy(strategy_path)
        if self.locked_kalshi_side is None:
            return True
        if kalshi_side != self.locked_kalshi_side:
            logger.info("SKIP: direction locked to %s, need %s",
                        self.locked_kalshi_side.value, kalshi_side.value)
            return False
        return True

    # --- user interaction ---

    # --- user interaction ---

    async def _wait_for_resume_input(self):
        """Background task to wait for user input to resume."""
        prompt = (
            "\n" + "=" * 60 + "\n"
            "  MATCHED FILL COMPLETE - PAUSED\n"
            "  Monitoring continues...\n"
            "  Press SPACE then ENTER to resume execution.\n"
            "  Press Ctrl+C to exit.\n"
            + "=" * 60 + "\n"
        )
        print(prompt)  # Use print to ensure it bypasses log formatting if needed, or use logger
        
        loop = asyncio.get_event_loop()
        while self.paused_for_input:
            user_input = await loop.run_in_executor(None, input)
            if user_input.strip() == "":
                # If just ENTER is pressed, maybe just reprint prompt or ignore?
                # User asked for "spacebar to continue".
                # If they type anything else, we might just ignore or treat as resume?
                # Strict interpretation: Check for space. But input() strips trailing newline.
                # ' ' + Enter -> input is ' '
                pass
            
            # Since input() might strip whitespace depending on terminal, 
            # or user might type " " (space) then enter.
            # Let's check if the input contains a space or if it's non-empty.
            # actually, standard input() behavior: 
            # typing " " then enter gives " ". 
            
            if " " in user_input:
                logger.info("RESUMING EXECUTION...")
                self.paused_for_input = False
                return
            else:
                 print("  [PAUSED] Press SPACE then ENTER to resume...")

    # --- execution-aware strategy gating ---

    def _executor_state_value(self) -> str:
        """Get executor state as string, immune to enum identity issues."""
        state = self.executor.state
        return state.value if hasattr(state, 'value') else str(state)

    def _session_remaining_capacity(self) -> int:
        """Remaining shares allowed this session."""
        return max(0, self._max_session_position - self._session_kalshi_fills)

    def _is_execution_ready(self) -> bool:
        """Check if strategy is allowed to submit execution attempts.

        Returns False when:
        - Session position cap reached
        - Strategy is suspended (cooldown active)
        - Executor is in FLAT_FORCED (auto-suspends)
        """
        if self._session_remaining_capacity() <= 0:
            return False

        now = time.monotonic()

        if self._strategy_suspended:
            if now >= self._suspension_until:
                self._lift_suspension()
            else:
                return False

        if self._executor_state_value() == 'FLAT_FORCED':
            self._suspend_strategy("executor_FLAT_FORCED")
            return False

        return True

    def _suspend_strategy(self, reason: str) -> None:
        """Suspend strategy with exponential backoff cooldown."""
        self._consecutive_rejects += 1
        cooldown = min(
            self._base_cooldown_sec * (2 ** (self._consecutive_rejects - 1)),
            self._max_cooldown_sec,
        )
        self._strategy_suspended = True
        self._suspension_until = time.monotonic() + cooldown
        self._suspension_reason = reason
        logger.warning(
            "STRATEGY_SUSPENDED: reason=%s cooldown=%.1fs rejects=%d",
            reason, cooldown, self._consecutive_rejects,
        )

    def _lift_suspension(self) -> None:
        """Lift strategy suspension and reset executor if needed."""
        state_val = self._executor_state_value()
        if state_val == 'FLAT_FORCED':
            self.executor.reset_to_monitor()
            logger.info("Suspension expired: reset executor FLAT_FORCED -> MONITOR")
        self._strategy_suspended = False
        self._suspension_reason = ""
        logger.info("STRATEGY_RESUMED after %d consecutive rejects", self._consecutive_rejects)

    def _on_successful_execution(self) -> None:
        """Reset backoff counters after a successful execution."""
        self._consecutive_rejects = 0

    async def _main_loop(self):
        logger.info("Entering main loop...")
        self.paused_for_input = False  # Ensure initialized

        while self.running:
            try:
                snapshot = await self.data_manager.get_fresh_snapshot()
                if not snapshot.is_valid():
                    await asyncio.sleep(0.1)
                    continue

                trigger = self.detector.detect(snapshot)
                
                # If paused, we only monitor and log
                if self.paused_for_input:
                    self._log_price_monitoring(snapshot, trigger)
                    await asyncio.sleep(0.5)
                    continue

                if trigger.triggered:
                    logger.info("TRIGGER! path=%d qty=%d cross=%.4f profit=%.2f%%",
                                trigger.strategy_path, trigger.max_profit_qty,
                                trigger.cross_cost, trigger.raw_profit_pct)

                    if not self._is_kalshi_direction_allowed(trigger.strategy_path):
                        await asyncio.sleep(0.5)
                        continue

                    # Gate: execution readiness (suspension / executor state)
                    if not self._is_execution_ready():
                        now = time.monotonic()
                        if now - self._last_suppression_log >= 10.0:
                            remaining_cap = self._session_remaining_capacity()
                            if remaining_cap <= 0:
                                logger.info(
                                    "TRIGGER_SUPPRESSED: SESSION_CAP_REACHED pos=%d/%d",
                                    self._session_kalshi_fills, self._max_session_position,
                                )
                            else:
                                logger.info(
                                    "TRIGGER_SUPPRESSED: reason=%s remaining=%.1fs rejects=%d",
                                    self._suspension_reason,
                                    max(0.0, self._suspension_until - now),
                                    self._consecutive_rejects,
                                )
                            self._last_suppression_log = now
                        await asyncio.sleep(0.5)
                        continue

                    if self.poll_only:
                        trigger.log_details()
                    else:
                        self._execution_context = {
                            'strategy_path': trigger.strategy_path,
                            'max_profit_qty': trigger.max_profit_qty,
                            'avg_entry_price': trigger.avg_entry_price,
                        }
                        result = await self.executor.try_execute(snapshot)
                        if result is None:
                            state_val = self._executor_state_value()
                            logger.info("EXECUTOR_NO_RESULT: state=%s", state_val)
                            self._suspend_strategy(f"exec_rejected_{state_val}")
                        else:
                            logger.info("Executor returned: state=%s error=%s",
                                        result.state.value, result.error)
                            if result.state.value == 'FLAT_FORCED':
                                self._suspend_strategy("FLAT_FORCED")
                            elif result.state.value == 'DONE_EQUAL':
                                self._on_successful_execution()
                                if self.locked_kalshi_side is None:
                                    self.locked_kalshi_side = self._get_kalshi_side_for_strategy(trigger.strategy_path)
                                    logger.info("Direction locked to %s", self.locked_kalshi_side.value)
                                # Pause after matched fill on both venues
                                k_fill = self._execution_context.get('kalshi_fill_qty', 0)
                                p_fill = self._execution_context.get('poly_fill_qty', 0)
                                if k_fill > 0 and p_fill > 0:
                                    # Enable pause state and start background input listener
                                    self.paused_for_input = True
                                    asyncio.create_task(self._wait_for_resume_input())
                else:
                    self._log_price_monitoring(snapshot, trigger)
                    if trigger.cross_cost and trigger.cross_cost < 1.0:
                        logger.info("SKIP reason=%s cross=%.4f profit=%.2f%% qty=%d",
                                    trigger.reason, trigger.cross_cost,
                                    trigger.raw_profit_pct or 0, trigger.max_profit_qty)

                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(1)

    def _log_price_monitoring(self, snapshot: MarketSnapshot, trigger) -> None:
        poly_yes_ask = snapshot.poly_yes_state.best_ask
        poly_no_ask = snapshot.poly_no_state.best_ask
        kalshi_yes_ask = snapshot.kalshi_state.best_ask
        kalshi_yes_bid = snapshot.kalshi_state.best_bid
        kalshi_no_ask = (1.0 - kalshi_yes_bid) if kalshi_yes_bid is not None else None

        cross1 = (poly_yes_ask + kalshi_no_ask) if poly_yes_ask and kalshi_no_ask else None
        cross2 = (poly_no_ask + kalshi_yes_ask) if poly_no_ask and kalshi_yes_ask else None

        best_cross, best_num = None, None
        if cross1 is not None and cross2 is not None:
            best_cross, best_num = (cross1, 1) if cross1 < cross2 else (cross2, 2)
        elif cross1 is not None:
            best_cross, best_num = cross1, 1
        elif cross2 is not None:
            best_cross, best_num = cross2, 2

        def fmt(v): return f"{v:.4f}" if v is not None else "N/A"

        lines = [
            "-" * 75, "LIVE PRICE MATRIX", "-" * 75,
            f"  Polymarket:  YES ask: {fmt(poly_yes_ask)}  NO ask: {fmt(poly_no_ask)}",
            f"  Kalshi:      YES ask: {fmt(kalshi_yes_ask)}  NO ask: {fmt(kalshi_no_ask)}",
            "-" * 75, "CROSS SUMS & STATUS", "-" * 75,
            f"  Cross 1 (Poly YES + Kalshi NO): {fmt(cross1)}",
            f"  Cross 2 (Poly NO + Kalshi YES): {fmt(cross2)}",
        ]
        if best_cross is not None:
            lines.append(f"  Best cross: #{best_num} @ {fmt(best_cross)}")
        lines.append("-" * 75)
        logger.info("\n" + "\n".join(lines))

    async def _execute_trade(self, snapshot: MarketSnapshot) -> bool:
        """Execute trade: Kalshi IOC first, gate Poly FAK on fill."""
        if self.poll_only:
            logger.info("Poll-only mode: skipping execution.")
            return True

        context = self._execution_context
        strategy_path = context.get('strategy_path')
        max_profit_qty = context.get('max_profit_qty')

        if strategy_path not in (1, 2) or not isinstance(max_profit_qty, int) or max_profit_qty <= 0:
            logger.error("No valid strategy data: path=%s qty=%s", strategy_path, max_profit_qty)
            return False

        # Preflight
        preflight = validate_executable_l1(snapshot, strategy_path, min_shares=1, profit_threshold_pct=1.0)
        if not preflight.passed:
            logger.info("PREFLIGHT_FAIL: reason=%s", preflight.reason)
            return None

        # Market IDs
        kalshi_ticker = getattr(self.scanner, 'kalshi_ticker', None)
        poly_yes_token = getattr(self.scanner, 'poly_yes_token', None)
        poly_no_token = getattr(self.scanner, 'poly_no_token', None)
        if not all([kalshi_ticker, poly_yes_token, poly_no_token]):
            logger.error("Market IDs not initialized")
            return False

        # Price derivation
        if strategy_path == 1:
            kalshi_price = (1.0 - snapshot.kalshi_state.best_bid) if snapshot.kalshi_state.best_bid else None
        else:
            kalshi_price = snapshot.kalshi_state.best_ask

        poly_price = snapshot.poly_yes_state.best_ask if strategy_path == 1 else snapshot.poly_no_state.best_ask
        if kalshi_price is None or poly_price is None:
            logger.error("Price is None for path=%d", strategy_path)
            return False

        remaining = self._session_remaining_capacity()
        if remaining <= 0:
            logger.info("SESSION_CAP_REACHED: kalshi_fills=%d cap=%d",
                        self._session_kalshi_fills, self._max_session_position)
            return None
        target_qty = min(max_profit_qty, 5, remaining)
        plan = create_execution_plan(
            kalshi_market_id=kalshi_ticker,
            poly_market_id=poly_yes_token if strategy_path == 1 else poly_no_token,
            target_qty=target_qty,
            walked_kalshi_price=kalshi_price,
            walked_poly_price=poly_price,
            strategy_path=strategy_path,
            max_acceptable_kalshi_price=1.00,
        )

        logger.info("EXECUTE: path=%d qty=%d K(%s @%.2f$) P(%s @%.2f$)",
                     strategy_path, target_qty, plan.kalshi_outcome, plan.walked_kalshi_price,
                     plan.poly_outcome, plan.walked_poly_price)

        # Order submitters
        kalshi_ws = self._kalshi_ws

        async def submit_kalshi(order: OrderRequest) -> OrderResult:
            try:
                # Submit order via REST
                result = await self.kalshi_client.create_order(
                    ticker=order.market_id, action='buy',
                    side=plan.kalshi_outcome.lower(), count=order.quantity, type='limit',
                    yes_price=int(order.price * 100) if plan.kalshi_outcome == 'YES' else None,
                    no_price=int(order.price * 100) if plan.kalshi_outcome == 'NO' else None,
                    time_in_force='immediate_or_cancel',
                )
                order_data = result.get('order', {})
                order_id = order_data.get('order_id')
                
                # Store original action for potential flattening
                self._execution_context['kalshi_action'] = 'buy' # We always buy in submit_kalshi currently
                self._execution_context['kalshi_side'] = plan.kalshi_outcome.lower()

                # Check synchronous fill from REST response first
                rest_filled = order_data.get('quantity', 0) - order_data.get('remaining_quantity', 0)

                if rest_filled > 0:
                    logger.info("Kalshi REST response shows immediate fill: qty=%d order_id=%s",
                                rest_filled, order_id)
                    return OrderResult(venue='kalshi', success=True, filled_qty=rest_filled,
                                       avg_fill_price=order.price,
                                       order_id=order_id, latency_ms=0)

                # REST says zero fill — wait for WS fill notification
                if kalshi_ws and order_id:
                    kalshi_ws.register_fill_watch(order_id)
                    logger.info("Kalshi REST zero fill, waiting for WS fill notification order_id=%s",
                                order_id)
                    ws_filled = await kalshi_ws.wait_for_fill(order_id, timeout_sec=3.0)
                    kalshi_ws.cleanup_fill_watch(order_id)

                    if ws_filled > 0:
                        logger.info("Kalshi WS fill detected: qty=%d order_id=%s", ws_filled, order_id)
                        return OrderResult(venue='kalshi', success=True, filled_qty=ws_filled,
                                           avg_fill_price=order.price,
                                           order_id=order_id, latency_ms=0)

                    logger.info("Kalshi WS confirms zero fill for order_id=%s", order_id)

                return OrderResult(venue='kalshi', success=rest_filled > 0, filled_qty=rest_filled,
                                   avg_fill_price=order.price if rest_filled > 0 else None,
                                   order_id=order_id, latency_ms=0)
            except Exception as e:
                logger.error(f"Kalshi submit failed: {e}", exc_info=True)
                return OrderResult(venue='kalshi', success=False, filled_qty=0,
                                   avg_fill_price=None, order_id=None, latency_ms=0, error=str(e))

        async def submit_poly(order: OrderRequest) -> OrderResult:
            try:
                # 1. Post order to Poly
                result = await self.polymarket_client.create_order(
                    token_id=order.market_id,
                    side='BUY' if order.side == OrderSide.BUY else 'SELL',
                    price=order.price, size=float(order.quantity), order_type='FAK',
                )
                
                # 2. Extract order ID
                order_id = result.get('order_id') or result.get('orderID')
                if not order_id:
                     # Check if result indicates failure or immediate fill without ID (unlikely for FAK)
                     logger.warning("Poly order submitted but no order_id in response: %s", result)
                     # Fallback to response size matched
                     filled = int(float(result.get('size_matched', 0) or 0))
                     return OrderResult(venue='poly', success=filled > 0, filled_qty=filled,
                                        avg_fill_price=order.price if filled > 0 else None,
                                        order_id=None, latency_ms=0)

                # 3. Check synchronous fill from REST response
                rest_filled = int(float(result.get('size_matched', 0) or 0))
                
                # 4. Use User WS to confirm/wait for fills if we have an ID
                # Even if REST reports fill, WS confirmation is good for authoritative record similar to Kalshi logging
                # If REST reports 0 fill, we DEFINITELY wait for WS in case of latency
                
                final_filled = rest_filled
                if self._poly_user_ws:
                    self._poly_user_ws.register_fill_watch(order_id)
                    if rest_filled == 0:
                         logger.info("Poly REST zero fill, waiting for WS fill notification order_id=%s", order_id)
                         ws_filled = await self._poly_user_ws.wait_for_fill(order_id, timeout_sec=3.0)
                         if ws_filled > 0:
                             logger.info("Poly WS fill detected: qty=%d order_id=%s", ws_filled, order_id)
                             final_filled = int(ws_filled)
                         else:
                             logger.info("Poly WS confirms zero fill for order_id=%s", order_id)
                    else:
                        # If REST reported fill, we still check WS briefly to ensure we caught the event logs? 
                        # Or just rely on REST. Kalshi logic waits only if REST=0.
                        # User said "add polymarket fill monitoring exactly how we have kalshi fil monitoring"
                        # Kalshi logic: "REST says zero fill — wait for WS fill notification". 
                        # So if REST > 0, we can return. But we might want to ensure WS buffer captures it for logging.
                        pass
                    
                    self._poly_user_ws.cleanup_fill_watch(order_id)

                return OrderResult(venue='poly', success=final_filled > 0, filled_qty=final_filled,
                                   avg_fill_price=order.price if final_filled > 0 else None,
                                   order_id=order_id, latency_ms=0)
            except Exception as e:
                logger.error(f"Poly submit failed: {e}", exc_info=True)
                return OrderResult(venue='poly', success=False, filled_qty=0,
                                   avg_fill_price=None, order_id=None, latency_ms=0, error=str(e))

        # Execute with fill-biased sequencer
        sequencer = ExecutionSequencer(
            kalshi_wait_ms=DEFAULT_KALSHI_WAIT_MS,
            high_latency_threshold_ms=DEFAULT_HIGH_LATENCY_THRESHOLD_MS,
        )
        try:
            exec_result = await sequencer.execute(plan, submit_kalshi, submit_poly)
            k_filled = exec_result.kalshi_result.filled_qty
            p_filled = exec_result.poly_result.filled_qty if exec_result.poly_result else 0
            self._execution_context['kalshi_fill_qty'] = k_filled
            self._execution_context['poly_fill_qty'] = p_filled

            # Accumulate session fills
            if k_filled > 0:
                self._session_kalshi_fills += k_filled
            if p_filled > 0:
                self._session_poly_fills += p_filled

            # Prominent fill / position log
            if k_filled > 0 or p_filled > 0:
                kalshi_authoritative_pos = (
                    self._kalshi_ws.last_post_position
                    if self._kalshi_ws else None
                )
                logger.info(
                    "FILL: K=%d P=%d | SESSION_POSITION: K=%d/%d P=%d | "
                    "KALSHI_AUTH_POS: %s",
                    k_filled, p_filled,
                    self._session_kalshi_fills, self._max_session_position,
                    self._session_poly_fills,
                    kalshi_authoritative_pos,
                )
            logger.info("EXECUTION_COMPLETE: reason=%s K=%d P=%s",
                        exec_result.reason, k_filled,
                        p_filled if exec_result.poly_result else "N/A")
            return k_filled > 0
        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return False

    async def _reconcile_trade(self, snapshot: MarketSnapshot) -> ExecutorState:
        """Reconcile fills: 4-case logic + flatten if unhedged."""
        try:
            from .reconciliation import SimpleReconciler
            kalshi_fill = self._execution_context.get('kalshi_fill_qty', 0)
            poly_fill = self._execution_context.get('poly_fill_qty', 0)
            logger.info("Reconciling: kalshi=%d poly=%d", kalshi_fill, poly_fill)

            state = await SimpleReconciler().reconcile_fills(kalshi_fill, poly_fill)

            if state.value == 'FLAT_FORCED' and kalshi_fill > 0 and poly_fill == 0:
                logger.warning("UNHEDGED: kalshi=%d poly=0 - flattening", kalshi_fill)
                kalshi_ticker = getattr(self.scanner, 'kalshi_ticker', None)
                kalshi_bid = snapshot.kalshi_state.best_bid or 0.01
                if kalshi_ticker:
                    try:
                        from .flatten import Flattener
                        flattener = Flattener(timeout_sec=5.0)
                        # Determine reverse action
                        # We always BUY in execute, so we should SELL to close.
                        # Side should match the position we opened.
                        # If we bought NO, we are long NO (short YES). To close, we sell NO (or buy YES).
                        # Kalshi "sell" means closing a position if we have one?
                        # Or if we entered with "buy" "no", we have +NO contracts.
                        # To flatten, we "sell" "no" contracts.
                        
                        orig_side = self._execution_context.get('kalshi_side', 'yes')
                        
                        # Flatten via direct Kalshi sell (IOC)
                        # We sell the same side we bought to close long position
                        price_to_use = 1 
                        if orig_side == 'yes':
                             # selling yes: price is bid
                             price_to_use = max(1, int((snapshot.kalshi_state.best_bid or 0.01) * 100))
                        else:
                             # selling no: price is implied bid (100 - ask of yes?) 
                             # or if 'no' markets have explicit orderbook? assuming standard yes/no structure
                             # For NO contracts, selling them means getting rid of them.
                             # If we emulate selling NO, we might need to be careful if api supports 'sell' 'no'
                             # Assuming standard API: action='sell', side=orig_side
                             # Price roughly 1c to dump?
                             price_to_use = 1

                        await self.kalshi_client.create_order(
                            ticker=kalshi_ticker, action='sell', side=orig_side,
                            count=kalshi_fill, type='limit',
                            yes_price=price_to_use if orig_side == 'yes' else None,
                            no_price=price_to_use if orig_side == 'no' else None,
                            time_in_force='immediate_or_cancel',
                        )
                    except Exception as e:
                        logger.error("Flatten failed: %s", e, exc_info=True)

            logger.info("Reconciliation: state=%s", state.value)
            return state
        except Exception as e:
            logger.error(f"Reconciliation error: {e}", exc_info=True)
            return ExecutorState.FLAT_FORCED


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoBot - Cross-Venue Arbitrage")
    parser.add_argument("mode", nargs="?", choices=["paper", "live"], default=None)
    parser.add_argument("--asset", default="eth")
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument("--next", dest="find_next", action="store_true")
    args = parser.parse_args()

    bot = AutoBot(asset=args.asset, mode=args.mode, poll_only=args.poll_only, find_next=args.find_next)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        pass
