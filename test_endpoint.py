import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        print("Sending request...")
        async with client.stream("POST", "http://127.0.0.1:8000/chat/stream", json={"message": "I wanna know about return policy"}, timeout=120.0) as response:
            print("Response status:", response.status_code)
            async for chunk in response.aiter_text():
                print(chunk, end="")
        print("Done")

asyncio.run(main())
