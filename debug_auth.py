
import os
import asyncio
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams
from py_clob_client.constants import POLYGON

load_dotenv()

def debug_auth():
    pk = os.getenv("POLYMARKET_PRIVATE_KEY") or os.getenv("PK")
    api_key = os.getenv("POLYMARKET_API_KEY")
    api_secret = os.getenv("POLYMARKET_API_SECRET")
    api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")
    
    if not (pk and api_key):
        print("Missing PK or API Key")
        return

    print("Initializing ClobClient (Direct)...")
    creds = ApiCreds(api_key, api_secret, api_passphrase)
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=POLYGON,
        creds=creds
    )
    
    print(f"Signer Address: {client.get_address()}")
    
    try:
        print("Checking Balance (EOA)...")
        params = BalanceAllowanceParams(asset_type="COLLATERAL")
        # Direct call, no monkeypatch
        bal = client.get_balance_allowance(params=params)
        print(f"Balance result: {bal}")
        print("Checking API Keys...")
        keys = client.get_api_keys()
        print(f"API Keys: {keys}")
        
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    debug_auth()
