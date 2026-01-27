import pytest
import asyncio
from unittest.mock import MagicMock
from src.polymarket_ws import PolymarketWSHandler
from src.venue_state import VenueState

@pytest.fixture
def mock_client():
    return MagicMock()

@pytest.fixture
def handler(mock_client):
    return PolymarketWSHandler("context_id", mock_client)

@pytest.mark.asyncio
async def test_process_message_verbose_format(handler):
    """Test parsing of verbose message format {"price": ..., "size": ...}"""
    msg = {
        "event_type": "book",
        "market": "context_id",
        "bids": [{"price": "0.55", "size": "100"}],
        "asks": [{"price": "0.60", "size": "200"}],
        "timestamp": "123456789"
    }
    
    state = await handler.process_message(msg)
    
    assert state is not None
    assert len(state.bid_levels) == 1
    assert state.bid_levels[0].price == 0.55
    assert state.bid_levels[0].size == 100
    
    assert len(state.ask_levels) == 1
    assert state.ask_levels[0].price == 0.60
    assert state.ask_levels[0].size == 200

@pytest.mark.asyncio
async def test_process_message_compressed_format(handler):
    """Test parsing of compressed message format {"p": ..., "s": ...}"""
    msg = {
        "ev": "book",
        "m": "context_id",
        "b": [{"p": "0.45", "s": "50"}],
        "a": [{"p": "0.65", "s": "75"}],
        "t": "123456789"
    }
    
    state = await handler.process_message(msg)
    
    assert state is not None
    assert len(state.bid_levels) == 1
    assert state.bid_levels[0].price == 0.45
    assert state.bid_levels[0].size == 50
    
    assert len(state.ask_levels) == 1
    assert state.ask_levels[0].price == 0.65
    assert state.ask_levels[0].size == 75

@pytest.mark.asyncio
async def test_process_message_mixed_keys(handler):
    """Test robustness with mixed keys if allowed/encountered"""
    # This might happen if API behavior involves both
    msg = {
        "event_type": "book",
        "bids": [{"price": "0.55", "size": "100"}],  # Verbose
        "asks": [{"p": "0.65", "s": "75"}]           # Compressed
    }
    
    state = await handler.process_message(msg)
    
    assert state is not None
    assert len(state.bid_levels) == 1
    assert state.bid_levels[0].price == 0.55
    
    assert len(state.ask_levels) == 1
    assert state.ask_levels[0].price == 0.65
