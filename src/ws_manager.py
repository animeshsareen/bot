"""WebSocket manager for real-time market data ingestion.

Maintains independent async WS tasks per venue with automatic reconnection
and REST fallback when WebSocket data becomes stale.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Any

from .venue_state import (
    VenueState,
    OrderLevel,
    MarketSnapshot,
    create_venue_state,
    create_snapshot,
    parse_kalshi_orderbook,
    parse_poly_orderbook,
    DEFAULT_STALE_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class WSConfig:
    """WebSocket configuration.

    Attributes:
        stale_ms: Staleness threshold in milliseconds
        reconnect_delay_ms: Initial delay before reconnection attempt
        max_reconnect_delay_ms: Maximum reconnection delay (exponential backoff cap)
        rest_timeout_ms: Maximum time to wait for REST fallback response
    """

    stale_ms: int = DEFAULT_STALE_MS
    reconnect_delay_ms: int = 1000
    max_reconnect_delay_ms: int = 30000
    rest_timeout_ms: int = 1000  # Skip if REST exceeds 1000ms (loosened from 300ms)


class VenueWSHandler(ABC):
    """Abstract base class for venue-specific WebSocket handlers.

    Subclasses implement venue-specific connection, subscription, and parsing.
    """

    def __init__(self, venue: str, market_id: str, config: WSConfig):
        self.venue = venue
        self.market_id = market_id
        self.config = config
        self._state: VenueState = create_venue_state(venue, market_id)
        self._state_lock = asyncio.Lock()
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_delay = config.reconnect_delay_ms

    @property
    def state(self) -> VenueState:
        """Current venue state (immutable snapshot)."""
        return self._state

    async def _update_state(self, new_state: VenueState) -> None:
        """Thread-safe state update."""
        async with self._state_lock:
            self._state = new_state

    @abstractmethod
    async def connect_and_subscribe(self) -> Any:
        """Connect to WebSocket and subscribe to market updates.

        Returns:
            WebSocket connection object
        """
        pass

    @abstractmethod
    async def process_message(self, message: dict) -> Optional[VenueState]:
        """Process incoming WebSocket message.

        Args:
            message: Parsed JSON message from WebSocket

        Returns:
            New VenueState if message contained orderbook update, else None
        """
        pass

    @abstractmethod
    async def fetch_rest_orderbook(self) -> Optional[VenueState]:
        """Fetch orderbook via REST API as fallback.

        Returns:
            New VenueState if fetch succeeded, else None
        """
        pass

    async def run(self) -> None:
        """Main WebSocket loop with reconnection handling."""
        self._running = True
        self._reconnect_delay = self.config.reconnect_delay_ms

        while self._running:
            try:
                # Mark state as stale on reconnect
                await self._update_state(self._state.mark_reconnecting())
                logger.info(
                    "%s: connecting to WebSocket for %s", self.venue, self.market_id
                )

                ws = await self.connect_and_subscribe()
                self._reconnect_delay = self.config.reconnect_delay_ms  # Reset on success

                async for raw_message in ws:
                    if not self._running:
                        break

                    try:
                        message = self._parse_message(raw_message)
                        if message is None:
                            continue

                        new_state = await self.process_message(message)
                        if new_state is not None:
                            await self._update_state(new_state)
                            logger.debug(
                                "%s: updated state, best_bid=%s, best_ask=%s",
                                self.venue,
                                new_state.best_bid,
                                new_state.best_ask,
                            )
                    except Exception as e:
                        logger.warning("%s: error processing message: %s", self.venue, e)

            except asyncio.CancelledError:
                logger.info("%s: WebSocket task cancelled", self.venue)
                break
            except Exception as e:
                logger.warning("%s: WebSocket error: %s", self.venue, e)

            if self._running:
                # Exponential backoff for reconnection
                delay_sec = self._reconnect_delay / 1000.0
                logger.info("%s: reconnecting in %.1fs", self.venue, delay_sec)
                await asyncio.sleep(delay_sec)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self.config.max_reconnect_delay_ms
                )

    def _parse_message(self, raw: Any) -> Optional[dict]:
        """Parse raw WebSocket message to dict."""
        import json

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None
        return raw if isinstance(raw, dict) else None

    def start(self) -> asyncio.Task:
        """Start the WebSocket task.

        Returns:
            The running asyncio Task
        """
        if self._ws_task is not None and not self._ws_task.done():
            return self._ws_task
        self._ws_task = asyncio.create_task(self.run())
        return self._ws_task

    async def stop(self) -> None:
        """Stop the WebSocket task gracefully."""
        self._running = False
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None


class MockVenueWSHandler(VenueWSHandler):
    """Mock WebSocket handler for testing.

    Allows injection of orderbook updates without actual WebSocket connection.
    """

    def __init__(
        self,
        venue: str,
        market_id: str,
        config: Optional[WSConfig] = None,
    ):
        super().__init__(venue, market_id, config or WSConfig())
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._rest_response: Optional[tuple[list[OrderLevel], list[OrderLevel]]] = None

    async def connect_and_subscribe(self) -> Any:
        """Return self as a mock async iterator."""
        return self

    def __aiter__(self):
        return self

    async def __anext__(self) -> dict:
        """Yield messages from queue."""
        try:
            return await asyncio.wait_for(self._message_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            # Keep yielding empty to allow checking _running
            if not self._running:
                raise StopAsyncIteration
            return {}

    async def process_message(self, message: dict) -> Optional[VenueState]:
        """Process mock orderbook message."""
        if "bid_levels" not in message and "ask_levels" not in message:
            return None

        bid_levels = message.get("bid_levels", [])
        ask_levels = message.get("ask_levels", [])
        update_ms = message.get("update_ms", int(time.time() * 1000))

        return self._state.with_orderbook(bid_levels, ask_levels, update_ms)

    async def fetch_rest_orderbook(self) -> Optional[VenueState]:
        """Return mock REST response."""
        if self._rest_response is None:
            return None
        bid_levels, ask_levels = self._rest_response
        return self._state.with_orderbook(bid_levels, ask_levels)

    def inject_orderbook(
        self,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        update_ms: Optional[int] = None,
    ) -> None:
        """Inject an orderbook update into the message queue."""
        self._message_queue.put_nowait(
            {
                "bid_levels": bid_levels,
                "ask_levels": ask_levels,
                "update_ms": update_ms or int(time.time() * 1000),
            }
        )

    def set_rest_response(
        self,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
    ) -> None:
        """Set the response for REST fallback."""
        self._rest_response = (bid_levels, ask_levels)

    async def update_state_directly(
        self,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        update_ms: Optional[int] = None,
    ) -> None:
        """Directly update state without going through WS loop (for testing)."""
        new_state = self._state.with_orderbook(
            bid_levels, ask_levels, update_ms or int(time.time() * 1000)
        )
        await self._update_state(new_state)


class MarketDataManager:
    """Manages market data from both venues with per-leg freshness tracking.

    Coordinates WebSocket handlers for Poly (YES and NO tokens) and Kalshi.
    Ensures per-leg freshness: ALL FOUR LEGS must be fresh before execution.
    """

    def __init__(
        self,
        poly_yes_handler: VenueWSHandler,
        poly_no_handler: VenueWSHandler,
        kalshi_handler: VenueWSHandler,
        aligned_market_id: str,
        config: Optional[WSConfig] = None,
    ):
        self.poly_yes_handler = poly_yes_handler
        self.poly_no_handler = poly_no_handler
        self.kalshi_handler = kalshi_handler
        self.aligned_market_id = aligned_market_id
        self.config = config or WSConfig()
        self._running = False
        # Rate limit REST fetches: track last fetch time per handler (ms)
        self._last_rest_fetch_ms: dict[str, int] = {}
        self._rest_fetch_gap_ms = 1000  # Don't fetch again within 1s

    def start(self) -> tuple[asyncio.Task, asyncio.Task, asyncio.Task]:
        """Start all three WebSocket handlers.

        Returns:
            Tuple of (poly_yes_task, poly_no_task, kalshi_task)
        """
        self._running = True
        return (
            self.poly_yes_handler.start(),
            self.poly_no_handler.start(),
            self.kalshi_handler.start(),
        )

    async def stop(self) -> None:
        """Stop all WebSocket handlers."""
        self._running = False
        await asyncio.gather(
            self.poly_yes_handler.stop(),
            self.poly_no_handler.stop(),
            self.kalshi_handler.stop(),
            return_exceptions=True,
        )

    def get_snapshot(self, now_ms: Optional[int] = None) -> MarketSnapshot:
        """Get current snapshot from both venues.

        Args:
            now_ms: Current time in ms (defaults to actual current time)

        Returns:
            MarketSnapshot (may be invalid if data is stale)
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        return create_snapshot(
            poly_yes_state=self.poly_yes_handler.state,
            poly_no_state=self.poly_no_handler.state,
            kalshi_state=self.kalshi_handler.state,
            aligned_market_id=self.aligned_market_id,
            snapshot_ms=now_ms,
        )

    async def get_fresh_snapshot(self, now_ms: Optional[int] = None) -> MarketSnapshot:
        """Get a fresh snapshot, using REST fallback if needed.

        Implements per-leg freshness: ALL FOUR LEGS must be fresh.
        If any leg is stale, issues async REST requests to refresh.

        Args:
            now_ms: Current time in ms (defaults to actual current time)

        Returns:
            MarketSnapshot (check is_valid() before trading)
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        poly_yes_state = self.poly_yes_handler.state
        poly_no_state = self.poly_no_handler.state
        kalshi_state = self.kalshi_handler.state

        # Check which legs need REST refresh (per-leg freshness)
        poly_yes_stale = poly_yes_state.is_stale(now_ms, self.config.stale_ms)
        poly_no_stale = poly_no_state.is_stale(now_ms, self.config.stale_ms)
        kalshi_stale = kalshi_state.is_stale(now_ms, self.config.stale_ms)

        if not poly_yes_stale and not poly_no_stale and not kalshi_stale:
            # All legs fresh, no REST needed
            return create_snapshot(
                poly_yes_state=poly_yes_state,
                poly_no_state=poly_no_state,
                kalshi_state=kalshi_state,
                aligned_market_id=self.aligned_market_id,
                snapshot_ms=now_ms,
            )

        # Issue REST requests for stale legs (with rate limiting to avoid hammering API)
        rest_tasks = []
        if poly_yes_stale and self._should_fetch_rest("poly_yes", now_ms):
            logger.debug("poly_yes: WS stale, fetching REST fallback")
            rest_tasks.append(("poly_yes", self._fetch_with_timeout(self.poly_yes_handler)))
            self._last_rest_fetch_ms["poly_yes"] = now_ms
        if poly_no_stale and self._should_fetch_rest("poly_no", now_ms):
            logger.debug("poly_no: WS stale, fetching REST fallback")
            rest_tasks.append(("poly_no", self._fetch_with_timeout(self.poly_no_handler)))
            self._last_rest_fetch_ms["poly_no"] = now_ms
        if kalshi_stale and self._should_fetch_rest("kalshi", now_ms):
            logger.debug("kalshi: WS stale, fetching REST fallback")
            rest_tasks.append(("kalshi", self._fetch_with_timeout(self.kalshi_handler)))
            self._last_rest_fetch_ms["kalshi"] = now_ms

        # Wait for REST responses concurrently
        results = await asyncio.gather(
            *[task for _, task in rest_tasks], return_exceptions=True
        )

        # Update states with REST results
        for i, (leg, _) in enumerate(rest_tasks):
            result = results[i]
            if isinstance(result, Exception):
                logger.warning("%s: REST fallback failed: %s", leg, result)
            elif result is not None:
                if leg == "poly_yes":
                    poly_yes_state = result
                elif leg == "poly_no":
                    poly_no_state = result
                else:
                    kalshi_state = result
                logger.debug("%s: REST fallback succeeded", leg)

        # Create snapshot with potentially refreshed states
        now_ms = int(time.time() * 1000)
        return create_snapshot(
            poly_yes_state=poly_yes_state,
            poly_no_state=poly_no_state,
            kalshi_state=kalshi_state,
            aligned_market_id=self.aligned_market_id,
            snapshot_ms=now_ms,
        )

    def _should_fetch_rest(self, leg: str, now_ms: int) -> bool:
        """Check if enough time has passed since last REST fetch for this leg.

        Rate limit REST fetches to avoid hammering the API.

        Args:
            leg: Leg identifier ("poly_yes", "poly_no", "kalshi")
            now_ms: Current time in milliseconds

        Returns:
            True if we should fetch, False if within gap window
        """
        if leg not in self._last_rest_fetch_ms:
            return True
        last_fetch = self._last_rest_fetch_ms[leg]
        elapsed = now_ms - last_fetch
        return elapsed >= self._rest_fetch_gap_ms

    async def _fetch_with_timeout(
        self, handler: VenueWSHandler
    ) -> Optional[VenueState]:
        """Fetch REST orderbook with timeout, cancellable if WS becomes fresh.

        Args:
            handler: Venue handler to fetch from

        Returns:
            New VenueState if fetch succeeded within timeout, else None
        """
        fetch_task = asyncio.create_task(handler.fetch_rest_orderbook())
        timeout_sec = self.config.rest_timeout_ms / 1000.0
        start_time = time.time()
        
        try:
            while True:
                # Check if WS became fresh
                if handler.state.is_fresh_after_reconnect:
                    logger.debug("%s: WS became fresh, cancelling REST fallback", handler.venue)
                    fetch_task.cancel()
                    try:
                        await fetch_task
                    except asyncio.CancelledError:
                        pass
                    return None

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_sec:
                    raise asyncio.TimeoutError()

                # Wait a bit for the task or freshness check
                # We use a short timeout for wait_for to allow frequent polling
                try:
                    # shield() ensures fetching isn't cancelled during the small wait if outer task is cancelled
                    # but here we want to wait for completion
                    return await asyncio.wait_for(asyncio.shield(fetch_task), timeout=0.01)
                except asyncio.TimeoutError:
                    # Continue loop to check freshness
                    continue

        except asyncio.TimeoutError:
            if not fetch_task.done():
                fetch_task.cancel()
                try:
                    await fetch_task
                except asyncio.CancelledError:
                    pass
            
            logger.warning(
                "%s: REST fallback exceeded %dms timeout",
                handler.venue,
                self.config.rest_timeout_ms,
            )
            return None
        except asyncio.CancelledError:
            if not fetch_task.done():
                fetch_task.cancel()
                try:
                    await fetch_task
                except asyncio.CancelledError:
                    pass
            raise
