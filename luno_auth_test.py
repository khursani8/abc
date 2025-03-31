import asyncio
import aiohttp
import os
from aiohttp import BasicAuth
import json

async def test_luno_auth():
    key_id = os.getenv('LUNO_API_KEY_ID')
    key_secret = os.getenv('LUNO_API_KEY_SECRET')
    api_base = "https://api.luno.com/api/1"
    endpoint = "/balance"
    url = f"{api_base}{endpoint}"

    if not key_id or not key_secret:
        print("Error: LUNO_API_KEY_ID or LUNO_API_KEY_SECRET not set.")
        return

    auth = BasicAuth(key_id, key_secret)
    print(f"Attempting to fetch: {url}")
    print(f"Using Key ID: {key_id[:4]}...") # Print partial key ID for confirmation

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=auth) as response:
                print(f"Status Code: {response.status}")
                try:
                    # Try to parse as JSON first
                    content = await response.json()
                    print("Response JSON:")
                    print(json.dumps(content, indent=2))
                except Exception as json_err:
                    # If JSON fails, print raw text
                    print(f"Could not decode JSON response: {json_err}")
                    try:
                        text_content = await response.text()
                        print("Response Text:")
                        print(text_content)
                    except Exception as text_err:
                        print(f"Could not read response text: {text_err}")

    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error: {e.status} {e.message}")
    except Exception as e:
        print(f"General Error: {e}")

if __name__ == "__main__":
    # Ensure uvloop is used if available, matching the main script
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop for test")
    except ImportError:
        print("uvloop not found, using default asyncio loop for test")
    asyncio.run(test_luno_auth())
