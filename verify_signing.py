
import asyncio
import os
from dotenv import load_dotenv
from src.polymarket_client import PolymarketClient
# from py_clob_client.clob_types import OrderArgs # Not needed if we use client wrapper method

load_dotenv()

async def verify_signing():
    print("Initializing PolymarketClient (PROD)...")
    # Use PROD explicitly
    client = PolymarketClient(host=PolymarketClient.PROD_HOST)
    
    if not client.can_trade:
        print("ERROR: API credentials missing, cannot trade/sign.")
        return

    print(f"Address: {client.client.get_address()}")
    
    # Create a dummy order to test signing (won't be posted)
    # Use a likely valid token ID from .env if available, else a real one from logical deductions
    token_id = os.getenv("TOKEN_ID_YES") or "54035466476492568351566157079015373095584269294414751338607324860913714443158"
    
    print(f"Using Token ID: {token_id}")
    
    print("Attempting to create (sign) an order locally...")
    try:
        # Just create the order object, which involves signing in py-clob-client

        
        # This method in py_clob_client typically returns a signed order dict/object
        signed_order = await client.create_order(
            price=0.50,
            size=10.0,
            side="BUY",
            token_id=token_id,
        )
        
        print("Order signed successfully!")
        print(f"Signed Order Keys: {signed_order.keys() if hasattr(signed_order, 'keys') else signed_order}")
        
        # Check if signature exists in the object
        # The structure usually has 'signature' or similar fields
        if hasattr(signed_order, 'signature') or 'signature' in str(signed_order):
            print("Signature found in order object.")
        else:
            print("WARNING: Signature NOT explicitly found in order object (might be hidden or different struct).")
            print(f"Order: {signed_order}")

    except Exception as e:
        print(f"Signing FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify_signing())
