#!/usr/bin/env python3
"""
Quick test of WebSocket connection to ensure MessageType.COMPLETE works
"""

import asyncio
import json
from websockets import connect


async def test_websocket():
    uri = "ws://localhost:8080/ws"
    
    try:
        async with connect(uri) as websocket:
            print("✓ Connected to WebSocket")
            
            # Send a simple request
            request = {
                "type": "content_request",
                "data": {
                    "topic": "test topic",
                    "max_iterations": 1,
                    "evaluator_profile": "general"
                },
                "workflow": "orchestrator"
            }
            
            await websocket.send(json.dumps(request))
            print("✓ Sent test request")
            
            # Wait for a few messages
            message_count = 0
            complete_received = False
            
            while message_count < 20:  # Limit messages to prevent infinite loop
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(message)
                    msg_type = data.get("type", "unknown")
                    
                    print(f"  Received: {msg_type}")
                    
                    if msg_type == "complete":
                        complete_received = True
                        print("✓ COMPLETE message received successfully!")
                        break
                    elif msg_type == "error":
                        print(f"✗ Error: {data.get('content', 'Unknown error')}")
                        break
                        
                    message_count += 1
                    
                except asyncio.TimeoutError:
                    print("✗ Timeout waiting for messages")
                    break
            
            if complete_received:
                print("\n✅ Test PASSED: MessageType.COMPLETE works correctly")
                return True
            else:
                print("\n⚠️ Test INCOMPLETE: Did not receive COMPLETE message")
                return False
                
    except ConnectionRefusedError:
        print("✗ Could not connect to server at ws://localhost:8080/ws")
        print("  Make sure the server is running: python main.py")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_websocket())
    exit(0 if result else 1)