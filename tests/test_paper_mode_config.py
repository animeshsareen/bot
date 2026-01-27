import os
import pytest
from src.bot import AutoBot
from src.kalshi_client import KalshiClient
from src.polymarket_client import PolymarketClient

def test_autobot_paper_mode_selection(monkeypatch):
    """Verify that AutoBot selects demo/staging URLs in paper mode."""
    monkeypatch.setenv("MODE", "paper")
    bot = AutoBot("eth")
    
    assert bot.mode == "paper"
    assert bot.kalshi_client.base_url == KalshiClient.DEMO_URL
    assert bot.polymarket_client.host == PolymarketClient.STAGING_HOST

def test_autobot_live_mode_selection(monkeypatch):
    """Verify that AutoBot selects production URLs in live mode."""
    monkeypatch.setenv("MODE", "live")
    bot = AutoBot("eth")
    
    assert bot.mode == "live"
    assert bot.kalshi_client.base_url == KalshiClient.PROD_URL
    assert bot.polymarket_client.host == PolymarketClient.PROD_HOST

def test_autobot_default_mode_is_paper(monkeypatch):
    """Verify that AutoBot defaults to paper mode if not specified."""
    monkeypatch.delenv("MODE", raising=False)
    bot = AutoBot("eth")
    
    assert bot.mode == "paper"
    assert bot.kalshi_client.base_url == KalshiClient.DEMO_URL
