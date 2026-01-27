import asyncio
import httpx
import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

from .kalshi_client import KalshiClient
from .polymarket_client import PolymarketClient
from .market_pair import MarketPair

logger = logging.getLogger(__name__)

# Market discovery constants
MIN_MARKET_TIME_REMAINING = 120  # 2 minutes minimum time left in market

_KALSHI_SERIES = {
    "btc": ("KXBTC15M", "KXBTC"),
    "eth": ("KXETH15M", "KXETH"),
    "sol": ("KXSOL15M", "KXSOL"),
}

class Scanner:
    """Multi-venue market scanner for Kalshi and Polymarket 15-minute markets.

    Monitors active 15-minute markets across both Kalshi and Polymarket venues,
    discovering matched market pairs using fuzzy text matching with strict expiration
    time validation. Supports real-time arbitrage and pricing analysis across venues.

    Core Features:
        - Multi-venue market discovery: Fetches all active 15m markets from both Kalshi
          and Polymarket for BTC, ETH, and SOL assets
        - Fuzzy market matching: Uses token_set_ratio (75% threshold) to match market
          names across venues by comparing normalized asset names (e.g., "BTC 15m")
        - Expiration validation: Enforces strict <2 minute (120 second) tolerance on
          market expiration times to ensure synchronized market pairs
        - Market pair discovery: Returns MarketPair objects with cross-venue metadata
          including match confidence scores

    Primary Methods:
        discover_matched_pairs(): Main entry point - discovers and returns all valid
            matched market pairs across venues
        _fetch_kalshi_15m_markets(): Fetches all open Kalshi 15m markets (primary
            series: KXBTC15M, KXETH15M, KXSOL15M; fallback to hourly if unavailable)
        _fetch_polymarket_15m_markets(): Fetches all active Polymarket 15m markets
            by scraping crypto/15M page
        _match_markets(): Core matching algorithm using fuzzy string similarity +
            expiration time validation

    Example Usage:
        scanner = Scanner('btc', kalshi_client, polymarket_client)
        pairs = await scanner.discover_matched_pairs()
        for pair in pairs:
            print(f"Kalshi: {pair.kalshi_ticker} closes {pair.kalshi_end_time}")
            print(f"Polymarket: {pair.polymarket_slug} closes {pair.polymarket_end_time}")
            print(f"Confidence: {pair.match_confidence:.1%}")

    Constraints:
        - Expiration Time Tolerance: Markets must expire within 120 seconds (<2 min)
          for pairing. This ensures trades can be executed synchronously.
        - Minimum Market Age: Polymarket markets must have >=120 seconds remaining
          to be considered valid (MIN_MARKET_TIME_REMAINING constant)
        - Fuzzy Matching Threshold: 75% string similarity required for market names
    """

    def __init__(self, asset: str, kalshi_client: KalshiClient, polymarket_client: PolymarketClient):
        self.asset = asset
        self.kalshi_client = kalshi_client
        self.polymarket_client = polymarket_client
        self.market_info = {
            'poly_yes_token': None,
            'poly_no_token': None,
            'kalshi_ticker': None,
            'market_end_time': None,
            'poly_slug': None
        }

    async def _find_current_15min_market(self, asset: str, wait_for_start: bool = True, find_next: bool = False) -> str:
        """Find the current active or next upcoming 15min up/down market on Polymarket.

        Args:
            asset: 'btc' or 'eth'
            wait_for_start: If True, will wait for a market to start if one is found that hasn't yet
            find_next: If True, return the next upcoming market (not yet started) instead of current

        Returns:
            Market slug string

        Raises:
            RuntimeError if no valid market found
        """
        if asset not in ("btc", "eth"):
            raise ValueError("asset must be 'btc' or 'eth'")

        try:
            page_url = "https://polymarket.com/crypto/15M"
            async with httpx.AsyncClient() as client:
                resp = await client.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                resp.raise_for_status()
                html_text = resp.text

            pattern = rf"{asset}-updown-15m-(\d+)"
            matches = re.findall(pattern, html_text)

            if not matches:
                raise RuntimeError(f"No active {asset.upper()} 15min market found")

            now_ts = int(datetime.now().timestamp())
            all_ts = sorted((int(ts) for ts in set(matches)), reverse=False)  # Sort ascending

            # If finding next market, look for future markets first
            if find_next:
                future_ts = [ts for ts in all_ts if ts > now_ts]
                if future_ts:
                    chosen_ts = future_ts[0]
                    time_until_start = chosen_ts - now_ts
                    slug = f"{asset}-updown-15m-{chosen_ts}"
                    print(f"  Selected NEXT market: opens in {time_until_start:.0f}s")
                    return slug
                else:
                    raise RuntimeError(f"No upcoming {asset.upper()} 15min markets found")

            # Filter for markets that:
            # 1. Have STARTED (now_ts >= ts)
            # 2. Have sufficient time remaining before close
            valid_ts = [
                ts for ts in all_ts
                if now_ts >= ts  # Market has started
                and (ts + 900) - now_ts >= MIN_MARKET_TIME_REMAINING  # Has time remaining
            ]

            if not valid_ts:
                # Check for markets that haven't started yet
                future_ts = [ts for ts in all_ts if ts > now_ts]
                if future_ts:
                    next_start = future_ts[0]
                    wait_seconds = next_start - now_ts

                    if wait_for_start:
                        # For short waits (< 30 min), wait directly
                        if wait_seconds <= 1800:  # 30 minutes max
                            wait_mins = wait_seconds / 60
                            print(f"  ⏳ Next market starts in {wait_mins:.1f}min, waiting...")
                            await asyncio.sleep(wait_seconds + 2)  # Wait for market start + buffer
                            # Recurse to find the now-active market
                            return await self._find_current_15min_market(asset, wait_for_start=False)
                        else:
                            # For long waits (> 30 min), let the caller retry
                            raise RuntimeError(f"Next market starts in {wait_seconds:.0f}s (too far ahead)")
                    else:
                        raise RuntimeError(f"Next market starts in {wait_seconds:.0f}s (too far ahead)")

                # Check if there are markets in their final minutes (< MIN_MARKET_TIME_REMAINING)
                ending_ts = [ts for ts in all_ts if now_ts >= ts and (ts + 900) - now_ts > 0]
                if ending_ts:
                    time_left = (ending_ts[-1] + 900) - now_ts
                    raise RuntimeError(f"Current market ending soon ({time_left:.0f}s remaining < {MIN_MARKET_TIME_REMAINING}s min)")

                raise RuntimeError(f"No open {asset.upper()} 15min markets found")

            # Pick the soonest expiring valid market (first in ascending order)
            chosen_ts = valid_ts[0]
            time_remaining = (chosen_ts + 900) - now_ts
            time_since_start = now_ts - chosen_ts
            slug = f"{asset}-updown-15m-{chosen_ts}"

            print(f"  Selected market: {time_since_start:.0f}s into cycle, {time_remaining:.0f}s remaining")
            return slug

        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    async def _fetch_market_details(self, slug: str):
        """Fetch market details from Polymarket event page (port from legacy arb.py).

        Scrapes the __NEXT_DATA__ payload from the event page to get token IDs.

        Args:
            slug: Market slug
        """
        slug = slug.split("?")[0]
        url = f"https://polymarket.com/event/{slug}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            resp.raise_for_status()
            html_text = resp.text

        # Parse __NEXT_DATA__ script tag
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html_text, re.DOTALL)
        if not m:
            raise RuntimeError("__NEXT_DATA__ payload not found on page")

        payload = json.loads(m.group(1))
        queries = payload.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
        market = None

        for q in queries:
            data = q.get("state", {}).get("data")
            if isinstance(data, dict) and "markets" in data:
                for mk in data["markets"]:
                    if mk.get("slug") == slug:
                        market = mk
                        break
            if market:
                break

        if not market:
            raise RuntimeError("Market slug not found in dehydrated state")

        clob_tokens = market.get("clobTokenIds") or []
        outcomes = market.get("outcomes") or []

        if len(clob_tokens) != 2 or len(outcomes) != 2:
            raise RuntimeError(f"Expected binary market with two clob tokens, got {len(clob_tokens)}")

        # Extract token IDs and end time
        self.market_info['poly_yes_token'] = clob_tokens[0]
        self.market_info['poly_no_token'] = clob_tokens[1]

        # Parse end time
        end_date_str = market.get("endDate")
        if end_date_str:
            self.market_info['market_end_time'] = datetime.fromisoformat(
                end_date_str.replace('Z', '+00:00')
            )

        print(f"    YES token: {clob_tokens[0][:20]}...")
        print(f"    NO token: {clob_tokens[1][:20]}...")

        # Find matching Kalshi ticker
        await self._find_kalshi_ticker()

    async def _find_kalshi_ticker(self):
        """Find matching Kalshi ticker for the current Polymarket (improved version).

        Uses KXBTC15M/KXETH15M series for 15-minute markets, with hourly fallback.
        Matches based on close time within 10-minute tolerance.
        """
        primary, fallback = _KALSHI_SERIES[self.asset]
        close_time_target = self.market_info['market_end_time']

        try:
            response = await self.kalshi_client.get_markets(series_ticker=primary, status="open", limit=20)
            markets = response.get("markets", [])

            if not markets:
                response = await self.kalshi_client.get_markets(series_ticker=fallback, status="open", limit=100)
                markets = response.get("markets", [])

            if not markets:
                raise RuntimeError(f"No open {primary} markets found")

            # Sort by close_time (soonest first)
            markets.sort(key=lambda x: x.get("close_time", "z"))

            # Find market with close time within 10 minutes of target
            best_match = None
            min_time_diff = float("inf")

            for market in markets:
                close_time_str = market.get("close_time")
                if not close_time_str:
                    continue

                try:
                     close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                except ValueError:
                     # Handle weird formats if necessary
                     continue
                     
                time_diff = abs((close_time - close_time_target).total_seconds())

                if time_diff <= 600 and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = market

            if not best_match:
                best_match = markets[0]

            self.market_info['kalshi_ticker'] = best_match.get("ticker", "")
            print(f"    Matched Kalshi ticker: {best_match['ticker']}")

        except Exception as e:
            raise RuntimeError(f"Kalshi market lookup failed: {e}")

    async def _fetch_kalshi_15m_markets(self) -> Dict[str, Dict]:
        """Fetch all open Kalshi 15m markets for the asset.

        Returns dict of {ticker: market_data} for all active 15m markets.
        For 'eth' asset, fetches KXETH15M series.
        """
        primary, fallback = _KALSHI_SERIES[self.asset]

        try:
            # Try primary 15m series first
            response = await self.kalshi_client.get_markets(
                series_ticker=primary,
                status="open",
                limit=50
            )
            markets = response.get("markets", [])

            if not markets:
                # Fallback to hourly if no 15m markets
                response = await self.kalshi_client.get_markets(
                    series_ticker=fallback,
                    status="open",
                    limit=100
                )
                markets = response.get("markets", [])

            if not markets:
                logger.warning(f"No open Kalshi markets found for {self.asset}")
                return {}

            # Build dict of ticker -> market data
            market_dict = {}
            for market in markets:
                ticker = market.get("ticker")
                if ticker:
                    market_dict[ticker] = market

            logger.info(f"Fetched {len(market_dict)} Kalshi 15m markets")
            return market_dict

        except Exception as e:
            logger.error(f"Error fetching Kalshi 15m markets: {e}")
            raise

    async def _fetch_polymarket_15m_markets(self) -> Dict[str, Dict]:
        """Fetch all active Polymarket 15m markets for the asset.

        Scrapes Polymarket crypto page to find 15m markets.
        Returns dict of {slug: market_data} containing basic market info.
        """
        try:
            page_url = "https://polymarket.com/crypto/15M"
            async with httpx.AsyncClient() as client:
                resp = await client.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                resp.raise_for_status()
                html_text = resp.text

            # Find all 15m markets for this asset
            pattern = rf"{self.asset}-updown-15m-(\d+)"
            matches = re.findall(pattern, html_text)

            if not matches:
                logger.warning(f"No Polymarket {self.asset} 15m markets found")
                return {}

            now_ts = int(datetime.now().timestamp())
            market_dict = {}

            # Build market dict with timestamps as keys
            for ts_str in set(matches):
                ts = int(ts_str)
                slug = f"{self.asset}-updown-15m-{ts}"

                # Only include markets that:
                # 1. Have started (now_ts >= ts)
                # 2. Have time remaining >= MIN_MARKET_TIME_REMAINING
                if now_ts >= ts and (ts + 900) - now_ts >= MIN_MARKET_TIME_REMAINING:
                    market_dict[slug] = {
                        "slug": slug,
                        "start_ts": ts,
                        "end_ts": ts + 900,
                    }

            logger.info(f"Fetched {len(market_dict)} Polymarket 15m markets")
            return market_dict

        except Exception as e:
            logger.error(f"Error fetching Polymarket 15m markets: {e}")
            raise

    async def _match_markets(
        self,
        kalshi_markets: Dict[str, Dict],
        polymarket_markets: Dict[str, Dict]
    ) -> list:
        """Match Kalshi and Polymarket 15m markets using fuzzy string matching.

        Matches based on:
        1. Fuzzy string similarity of market names (threshold: 75%)
        2. Expiration time within 120 seconds

        Args:
            kalshi_markets: Dict of {ticker: market_data}
            polymarket_markets: Dict of {slug: market_data}

        Returns:
            List of MarketPair objects for valid matches
        """
        from fuzzywuzzy import fuzz
        from .market_pair import MarketPair

        matched_pairs = []

        if not kalshi_markets or not polymarket_markets:
            logger.warning("Empty market dict(s), cannot match")
            return matched_pairs

        # For each Kalshi market, find best Polymarket match
        for kal_ticker, kal_market in kalshi_markets.items():
            kal_close_time_str = kal_market.get("close_time")
            if not kal_close_time_str:
                continue

            try:
                kal_close_time = datetime.fromisoformat(kal_close_time_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse Kalshi close_time for {kal_ticker}")
                continue

            # Build Kalshi market name for fuzzy matching (e.g., "ETH 15m")
            kal_name = f"{self.asset.upper()} 15m"

            best_match = None
            best_score = 0

            # Find best Polymarket match by fuzzy string similarity
            for poly_slug, poly_market in polymarket_markets.items():
                poly_name = poly_slug.replace("-", " ").upper()

                # Calculate fuzzy match score
                score = fuzz.token_set_ratio(kal_name, poly_name)

                if score > best_score and score >= 75:  # 75% similarity threshold
                    best_score = score
                    best_match = (poly_slug, poly_market)

            if best_match:
                poly_slug, poly_market = best_match

                # Validate expiration time difference
                poly_end_ts = poly_market.get("end_ts")
                if not poly_end_ts:
                    continue

                poly_close_time = datetime.fromtimestamp(poly_end_ts, tz=timezone.utc)

                # Create MarketPair and validate
                pair = MarketPair(
                    kalshi_ticker=kal_ticker,
                    kalshi_end_time=kal_close_time,
                    polymarket_slug=poly_slug,
                    polymarket_end_time=poly_close_time,
                    match_confidence=best_score / 100.0,
                )

                if pair.is_valid_pair(max_expiration_diff=120):  # < 2 min difference
                    matched_pairs.append(pair)
                    logger.info(f"Matched: {pair}")
                else:
                    logger.debug(f"Rejected pair (expiration diff {pair.expiration_diff_seconds()}s): {pair}")

        logger.info(f"Found {len(matched_pairs)} valid market pairs")
        return matched_pairs

    async def discover_market(self, find_next: bool = False) -> Dict:
        """Main entry point to discover market with graceful retry for gaps.

        Args:
            find_next: If True, find the next upcoming market instead of current active one
        """
        market_type = "next upcoming" if find_next else "active"
        print(f"  - Discovering {market_type} {self.asset.upper()} 15m market...")

        while True:
            try:
                slug = await self._find_current_15min_market(self.asset, find_next=find_next)
                self.market_info['poly_slug'] = slug
                print(f"    Found: {slug}")
                print("  - Fetching market details...")
                await self._fetch_market_details(slug)
                return self.market_info
            except RuntimeError as e:
                # Handle known market gaps (inter-cycle or ending)
                if "Next market" in str(e) or "ending soon" in str(e) or "No open" in str(e) or "No upcoming" in str(e):
                    print(f"  ⏳ {e} - retrying in 10s...")
                    await asyncio.sleep(10)
                    continue
                raise # Re-raise unexpected errors

    async def discover_matched_pairs(self) -> list:
        """Discover matched market pairs across Kalshi and Polymarket.

        Main entry point for multi-venue scanning:
        1. Fetches all active Kalshi 15m markets
        2. Fetches all active Polymarket 15m markets
        3. Matches them by fuzzy name similarity + expiration time

        Returns:
            List of MarketPair objects

        Raises:
            RuntimeError if no markets found on either venue
        """
        print(f"  - Discovering matched {self.asset.upper()} 15m markets...")

        try:
            print("    Fetching Kalshi 15m markets...")
            kalshi_markets = await self._fetch_kalshi_15m_markets()

            print("    Fetching Polymarket 15m markets...")
            polymarket_markets = await self._fetch_polymarket_15m_markets()

            if not kalshi_markets:
                raise RuntimeError("No open Kalshi 15m markets found")
            if not polymarket_markets:
                raise RuntimeError("No open Polymarket 15m markets found")

            print("    Matching markets across venues...")
            matched_pairs = await self._match_markets(kalshi_markets, polymarket_markets)

            if not matched_pairs:
                raise RuntimeError("No valid market pairs found (expiration mismatch?)")

            print(f"  ✓ Found {len(matched_pairs)} matched pairs")
            return matched_pairs

        except Exception as e:
            logger.error(f"Error discovering matched pairs: {e}")
            raise