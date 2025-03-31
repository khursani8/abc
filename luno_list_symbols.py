import asyncio
import aiohttp
import json

async def get_luno_symbols():
    """Fetches and prints available trading pairs from Luno API."""
    url = "https://api.luno.com/api/1/tickers"
    print(f"Fetching symbols from: {url}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                print(f"Status Code: {response.status}")
                response.raise_for_status() # Raise exceptions for 4xx/5xx errors
                data = await response.json()

                tickers = data.get('tickers', [])
                if not tickers:
                    print("No tickers found in the response.")
                    return

                pairs = sorted([ticker.get('pair') for ticker in tickers if ticker.get('pair')])
                if pairs:
                    print("\nAvailable Luno Trading Pairs:")
                    # Print in columns for better readability if many pairs
                    col_width = max(len(p) for p in pairs) + 2 # Find max length for padding
                    num_cols = 4 # Adjust number of columns as needed
                    for i in range(0, len(pairs), num_cols):
                         row = pairs[i:i+num_cols]
                         print("".join(f"{p:<{col_width}}" for p in row))
                else:
                    print("Could not extract pairs from the ticker data.")

    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error fetching symbols: {e.status} {e.message}")
        try:
            error_content = await e.json()
            print("Error details:", json.dumps(error_content, indent=2))
        except Exception:
            print("Could not parse error response.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure uvloop is used if available
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop")
    except ImportError:
        print("uvloop not found, using default asyncio loop")
    asyncio.run(get_luno_symbols())
