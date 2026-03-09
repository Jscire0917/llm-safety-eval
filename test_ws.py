import websocket
import json

print("Connecting to WebSocket...")
ws = websocket.create_connection("ws://localhost:8000/ws/evaluate?api_key=sk-dummy-local")

payload = {
    "model_name": "gpt-4o-mini",
    "provider": "openai",
    "metrics": ["bias", "toxicity", "hallucination"]
}

print("Sending payload:", payload)
ws.send(json.dumps(payload))

print("Receiving messages...")
while True:
    try:
        msg = ws.recv()
        print(msg)
        if "complete" in msg.lower():
            print("Evaluation complete – closing connection")
            break
    except websocket.WebSocketConnectionClosedException:
        print("Connection closed by server")
        break
    except Exception as e:
        print("Error:", e)
        break

ws.close()