
import asyncio
from src.polymarket_client import PolymarketClient
import py_clob_client.http_helpers.helpers as http_helpers

def verify_patch():
    print("Verifying monkeypatch...")
    headers = http_helpers.overloadHeaders("GET", {})
    ua = headers.get("User-Agent")
    print(f"User-Agent: {ua}")
    
    if "Mozilla" in ua:
        print("SUCCESS: User-Agent patched.")
    else:
        print("FAILURE: User-Agent not patched.")

if __name__ == "__main__":
    verify_patch()
