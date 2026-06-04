import httpx
import json

def test_stream():
    api_base = "http://localhost:8000"
    message = "I need a refund for order ORD-005-2"
    
    try:
        with httpx.Client(timeout=10) as client:
            with client.stream(
                "POST",
                f"{api_base}/chat/stream",
                json={"message": message, "thread_id": "test-thread-123"},
            ) as response:
                for line in response.iter_lines():
                    print(line)
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_stream()
