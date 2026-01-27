
import asyncio
import os
from dotenv import load_dotenv
from src.polymarket_client import PolymarketClient

load_dotenv()

async def check_balance_allowance():
    try:
        print("Initializing PolymarketClient (PROD)...")
        # Force PROD host since user keys are likely mainnet
        client = PolymarketClient(host=PolymarketClient.PROD_HOST)
        if not client.can_trade:
            print("ERROR: API credentials missing.")
            return

        print(f"Derived Address: {client.client.get_address()}")
        print("Fetching Balance & Allowance...")
        info = client.get_balance()
        print(f"Info: {info}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_balance_allowance())
