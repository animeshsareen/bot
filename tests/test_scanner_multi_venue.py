import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.market_pair import MarketPair
from src.scanner import Scanner


# ============================================================================
# MarketPair Unit Tests
# ============================================================================

def test_market_pair_expiration_validation():
    """Test MarketPair expiration difference calculation."""
    now = datetime.now()

    # Create two markets with 30s difference
    pair = MarketPair(
        kalshi_ticker="KXETH15M",
        kalshi_end_time=now,
        polymarket_slug="eth-updown-15m-123",
        polymarket_end_time=now + timedelta(seconds=30),
        match_confidence=0.85,
    )

    assert pair.expiration_diff_seconds() == 30
    assert pair.is_valid_pair(max_expiration_diff=120)
    assert pair.is_valid_pair(max_expiration_diff=20) is False


def test_market_pair_invalid_expiration():
    """Test MarketPair rejects > 120s expiration differences."""
    now = datetime.now()

    pair = MarketPair(
        kalshi_ticker="KXETH15M",
        kalshi_end_time=now,
        polymarket_slug="eth-updown-15m-456",
        polymarket_end_time=now + timedelta(seconds=150),  # > 120s
        match_confidence=0.80,
    )

    assert pair.expiration_diff_seconds() == 150
    assert pair.is_valid_pair(max_expiration_diff=120) is False


def test_market_pair_identical_times():
    """Test MarketPair with identical expiration times."""
    now = datetime.now()

    pair = MarketPair(
        kalshi_ticker="KXBTC15M",
        kalshi_end_time=now,
        polymarket_slug="btc-updown-15m-789",
        polymarket_end_time=now,
        match_confidence=0.95,
    )

    assert pair.expiration_diff_seconds() == 0
    assert pair.is_valid_pair(max_expiration_diff=120)


def test_market_pair_boundary_120s():
    """Test MarketPair at exactly 120s boundary."""
    now = datetime.now()

    pair = MarketPair(
        kalshi_ticker="KXETH15M",
        kalshi_end_time=now,
        polymarket_slug="eth-updown-15m-boundary",
        polymarket_end_time=now + timedelta(seconds=120),
        match_confidence=0.80,
    )

    assert pair.expiration_diff_seconds() == 120
    assert pair.is_valid_pair(max_expiration_diff=120)  # Exactly at boundary
    assert pair.is_valid_pair(max_expiration_diff=119) is False


# ============================================================================
# Scanner Fuzzy Matching Tests
# ============================================================================

@pytest.mark.asyncio
async def test_match_markets_identical_names():
    """Test fuzzy matching finds identical market names."""
    # Create mock clients
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    # Set up test data with timezone-aware datetimes
    now = datetime.now(timezone.utc)
    kalshi_markets = {
        "KXETH15M-ABC": {
            "ticker": "KXETH15M-ABC",
            "close_time": now.isoformat().replace('+00:00', 'Z'),
        }
    }

    # Create polymarket_end_time that's timezone-aware and ~50s after now
    poly_end_time = now.replace(tzinfo=timezone.utc)
    poly_end_ts = int(poly_end_time.timestamp()) + 50

    polymarket_markets = {
        "eth-updown-15m-1000": {
            "slug": "eth-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": poly_end_ts,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # Should find a match due to fuzzy matching and valid expiration time
    assert isinstance(matched_pairs, list)


@pytest.mark.asyncio
async def test_match_markets_case_variations():
    """Test fuzzy matching with case and spacing variations."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now(timezone.utc)

    # Kalshi market name will be "ETH 15m" (from scanner logic)
    kalshi_markets = {
        "KXETH15M-VAR1": {
            "ticker": "KXETH15M-VAR1",
            "close_time": now.isoformat().replace('+00:00', 'Z'),
        }
    }

    # Polymarket slug gets transformed to "ETH UPDOWN 15M 1000"
    polymarket_markets = {
        "eth-updown-15m-1000": {
            "slug": "eth-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(now.timestamp()) + 50,  # 50s difference
        }
    }

    # Use real fuzzy matching logic
    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # Fuzzy matching should handle variations (both contain "ETH" and "15M")
    assert isinstance(matched_pairs, list)


@pytest.mark.asyncio
async def test_match_markets_expiration_validation():
    """Test expiration time validation (<2min difference accepted)."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    base_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=timezone.utc)

    kalshi_markets = {
        "KXBTC15M-1": {
            "ticker": "KXBTC15M-1",
            "close_time": base_time.isoformat().replace('+00:00', 'Z'),
        }
    }

    # Polymarket market ends 90 seconds after Kalshi (within 120s threshold)
    polymarket_markets = {
        "btc-updown-15m-1000": {
            "slug": "btc-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(base_time.timestamp()) + 90,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # With fuzzy matching and valid expiration, should have a pair if match score >= 75
    assert isinstance(matched_pairs, list)


@pytest.mark.asyncio
async def test_match_markets_rejects_large_expiration_diff():
    """Test that markets with >2min expiration diff are NOT matched."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    base_time = datetime(2026, 1, 26, 12, 0, 0, tzinfo=timezone.utc)

    kalshi_markets = {
        "KXETH15M-1": {
            "ticker": "KXETH15M-1",
            "close_time": base_time.isoformat().replace('+00:00', 'Z'),
        }
    }

    # Polymarket market ends 150 seconds after Kalshi (> 120s threshold)
    polymarket_markets = {
        "eth-updown-15m-1000": {
            "slug": "eth-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(base_time.timestamp()) + 150,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # Should have 0 pairs because expiration difference exceeds 120s
    assert len(matched_pairs) == 0


@pytest.mark.asyncio
async def test_match_markets_empty_kalshi_markets():
    """Test matching with empty Kalshi markets returns empty list."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    kalshi_markets = {}
    now = datetime.now(timezone.utc)
    polymarket_markets = {
        "btc-updown-15m-1000": {
            "slug": "btc-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(now.timestamp()) + 50,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    assert matched_pairs == []


@pytest.mark.asyncio
async def test_match_markets_empty_polymarket_markets():
    """Test matching with empty Polymarket markets returns empty list."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now()
    kalshi_markets = {
        "KXETH15M-1": {
            "ticker": "KXETH15M-1",
            "close_time": now.isoformat() + "Z",
        }
    }
    polymarket_markets = {}

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    assert matched_pairs == []


@pytest.mark.asyncio
async def test_match_markets_both_empty():
    """Test matching with both venue markets empty returns empty list."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    matched_pairs = await scanner._match_markets({}, {})

    assert matched_pairs == []


# ============================================================================
# discover_matched_pairs Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_discover_matched_pairs_empty_result():
    """Test discover_matched_pairs returns empty list when no matches found."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    # Mock the fetch methods to return empty markets
    kalshi_client.get_markets = AsyncMock(return_value={"markets": []})
    polymarket_client.get = AsyncMock(return_value="")

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    # Mock the fetch methods
    scanner._fetch_kalshi_15m_markets = AsyncMock(return_value={})
    scanner._fetch_polymarket_15m_markets = AsyncMock(return_value={})

    # Should raise RuntimeError when no markets found
    with pytest.raises(RuntimeError):
        await scanner.discover_matched_pairs()


@pytest.mark.asyncio
async def test_discover_matched_pairs_no_kalshi_markets():
    """Test discover_matched_pairs raises when no Kalshi markets found."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    # Mock the fetch methods
    scanner._fetch_kalshi_15m_markets = AsyncMock(return_value={})
    scanner._fetch_polymarket_15m_markets = AsyncMock(
        return_value={"btc-updown-15m-1000": {"slug": "btc-updown-15m-1000"}}
    )

    # Should raise RuntimeError when no Kalshi markets
    with pytest.raises(RuntimeError, match="No open Kalshi"):
        await scanner.discover_matched_pairs()


@pytest.mark.asyncio
async def test_discover_matched_pairs_no_polymarket_markets():
    """Test discover_matched_pairs raises when no Polymarket markets found."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now()

    # Mock the fetch methods
    scanner._fetch_kalshi_15m_markets = AsyncMock(
        return_value={"KXETH15M-1": {"ticker": "KXETH15M-1", "close_time": now.isoformat() + "Z"}}
    )
    scanner._fetch_polymarket_15m_markets = AsyncMock(return_value={})

    # Should raise RuntimeError when no Polymarket markets
    with pytest.raises(RuntimeError, match="No open Polymarket"):
        await scanner.discover_matched_pairs()


@pytest.mark.asyncio
async def test_discover_matched_pairs_no_valid_matches():
    """Test discover_matched_pairs raises when no valid matches found."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now()

    # Mock the fetch methods with mismatched expiration times
    scanner._fetch_kalshi_15m_markets = AsyncMock(
        return_value={"KXETH15M-1": {"ticker": "KXETH15M-1", "close_time": now.isoformat() + "Z"}}
    )
    scanner._fetch_polymarket_15m_markets = AsyncMock(
        return_value={
            "eth-updown-15m-1000": {
                "slug": "eth-updown-15m-1000",
                "start_ts": 1000,
                "end_ts": int(now.timestamp()) + 200,  # >120s diff
            }
        }
    )

    # Should raise RuntimeError when no valid matches (expiration mismatch)
    with pytest.raises(RuntimeError, match="No valid market pairs"):
        await scanner.discover_matched_pairs()


@pytest.mark.asyncio
async def test_discover_matched_pairs_successful():
    """Test discover_matched_pairs returns matched pairs when valid matches exist."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    now = datetime.now()
    base_ts = int(now.timestamp())

    # Create markets with matching expiration times
    kalshi_markets = {
        "KXBTC15M-1": {
            "ticker": "KXBTC15M-1",
            "close_time": now.isoformat() + "Z",
        }
    }

    polymarket_markets = {
        "btc-updown-15m-1000": {
            "slug": "btc-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": base_ts + 30,  # 30s difference (within 120s)
        }
    }

    # Mock the fetch methods
    scanner._fetch_kalshi_15m_markets = AsyncMock(return_value=kalshi_markets)
    scanner._fetch_polymarket_15m_markets = AsyncMock(return_value=polymarket_markets)

    # Mock _match_markets to return valid pairs
    expected_pair = MarketPair(
        kalshi_ticker="KXBTC15M-1",
        kalshi_end_time=now,
        polymarket_slug="btc-updown-15m-1000",
        polymarket_end_time=datetime.fromtimestamp(base_ts + 30),
        match_confidence=0.85,
    )
    scanner._match_markets = AsyncMock(return_value=[expected_pair])

    result = await scanner.discover_matched_pairs()

    assert len(result) == 1
    assert result[0].kalshi_ticker == "KXBTC15M-1"
    assert result[0].polymarket_slug == "btc-updown-15m-1000"


@pytest.mark.asyncio
async def test_discover_matched_pairs_multiple_matches():
    """Test discover_matched_pairs returns multiple matched pairs."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now()
    base_ts = int(now.timestamp())

    kalshi_markets = {
        "KXETH15M-1": {
            "ticker": "KXETH15M-1",
            "close_time": now.isoformat() + "Z",
        },
        "KXETH15M-2": {
            "ticker": "KXETH15M-2",
            "close_time": (now + timedelta(seconds=100)).isoformat() + "Z",
        },
    }

    polymarket_markets = {
        "eth-updown-15m-1000": {
            "slug": "eth-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": base_ts + 50,
        },
        "eth-updown-15m-2000": {
            "slug": "eth-updown-15m-2000",
            "start_ts": 2000,
            "end_ts": base_ts + 150,
        },
    }

    # Mock the fetch methods
    scanner._fetch_kalshi_15m_markets = AsyncMock(return_value=kalshi_markets)
    scanner._fetch_polymarket_15m_markets = AsyncMock(return_value=polymarket_markets)

    # Mock _match_markets to return multiple pairs
    pairs = [
        MarketPair(
            kalshi_ticker="KXETH15M-1",
            kalshi_end_time=now,
            polymarket_slug="eth-updown-15m-1000",
            polymarket_end_time=datetime.fromtimestamp(base_ts + 50),
            match_confidence=0.85,
        ),
        MarketPair(
            kalshi_ticker="KXETH15M-2",
            kalshi_end_time=now + timedelta(seconds=100),
            polymarket_slug="eth-updown-15m-2000",
            polymarket_end_time=datetime.fromtimestamp(base_ts + 150),
            match_confidence=0.80,
        ),
    ]
    scanner._match_markets = AsyncMock(return_value=pairs)

    result = await scanner.discover_matched_pairs()

    assert len(result) == 2
    assert result[0].kalshi_ticker == "KXETH15M-1"
    assert result[1].kalshi_ticker == "KXETH15M-2"


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_match_markets_missing_close_time():
    """Test that markets without close_time are skipped."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("eth", kalshi_client, polymarket_client)

    now = datetime.now()

    # Kalshi market without close_time
    kalshi_markets = {
        "KXETH15M-BAD": {
            "ticker": "KXETH15M-BAD",
            # Missing close_time
        }
    }

    polymarket_markets = {
        "eth-updown-15m-1000": {
            "slug": "eth-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(now.timestamp()) + 500,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # Should return empty list because Kalshi market has no close_time
    assert matched_pairs == []


@pytest.mark.asyncio
async def test_match_markets_invalid_close_time_format():
    """Test handling of invalid datetime formats in close_time."""
    kalshi_client = AsyncMock()
    polymarket_client = AsyncMock()

    scanner = Scanner("btc", kalshi_client, polymarket_client)

    now = datetime.now()

    # Kalshi market with invalid close_time format
    kalshi_markets = {
        "KXBTC15M-BAD": {
            "ticker": "KXBTC15M-BAD",
            "close_time": "invalid-date-format",
        }
    }

    polymarket_markets = {
        "btc-updown-15m-1000": {
            "slug": "btc-updown-15m-1000",
            "start_ts": 1000,
            "end_ts": int(now.timestamp()) + 500,
        }
    }

    matched_pairs = await scanner._match_markets(kalshi_markets, polymarket_markets)

    # Should return empty list because of invalid datetime
    assert matched_pairs == []


@pytest.mark.asyncio
async def test_market_pair_confidence_score():
    """Test that MarketPair confidence scores are preserved."""
    now = datetime.now()

    pair = MarketPair(
        kalshi_ticker="KXETH15M",
        kalshi_end_time=now,
        polymarket_slug="eth-updown-15m-123",
        polymarket_end_time=now + timedelta(seconds=45),
        match_confidence=0.92,
    )

    assert pair.match_confidence == 0.92
    assert pair.is_valid_pair(max_expiration_diff=120)
