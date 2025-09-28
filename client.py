#!/usr/bin/env python3
import asyncio
import json
import argparse
import sys
from datetime import datetime
from websockets import connect
from websockets.exceptions import ConnectionClosed


async def send_content_request(websocket, topic: str, max_iterations: int = 3):
    request = {
        "type": "content_request",
        "data": {
            "topic": topic,
            "max_iterations": max_iterations
        },
        "workflow": "orchestrator"
    }
    await websocket.send(json.dumps(request))
    print(f"📤 Sent content request for topic: {topic}\n")


async def receive_messages(websocket):
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            content = data.get("content", "")
            timestamp = data.get("timestamp", "")
            
            if msg_type == "status":
                print(f"🔔 [{timestamp}] STATUS: {content}")
            
            elif msg_type == "agent_response":
                agent_id = data.get("agent_id", "unknown")
                metadata = data.get("metadata", {})
                iteration = metadata.get("iteration", "?")
                is_final = metadata.get("is_final", False)
                
                print(f"\n{'='*80}")
                print(f"🤖 AGENT: {agent_id.upper()} | Iteration: {iteration} | Final: {is_final}")
                print(f"{'='*80}")
                print(content)
                print(f"{'='*80}\n")
            
            elif msg_type == "complete":
                print(f"\n✅ [{timestamp}] {content}")
                print("\n🏁 Session completed. Disconnecting...\n")
                break
            
            elif msg_type == "error":
                print(f"\n❌ [{timestamp}] ERROR: {content}\n", file=sys.stderr)
                break
            
            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            
            elif msg_type == "pong":
                pass
            
            else:
                print(f"📨 [{timestamp}] {msg_type.upper()}: {content}")
    
    except ConnectionClosed:
        print("\n⚠️  Connection closed by server\n")
    except Exception as e:
        print(f"\n❌ Error receiving messages: {e}\n", file=sys.stderr)


async def run_client(topic: str, max_iterations: int, server_url: str):
    print(f"\n🔌 Connecting to {server_url}...\n")
    
    try:
        async with connect(server_url) as websocket:
            print("✅ Connected to Musequill server\n")
            
            await send_content_request(websocket, topic, max_iterations)
            
            await receive_messages(websocket)
            
    except ConnectionRefusedError:
        print(f"❌ Could not connect to {server_url}", file=sys.stderr)
        print("   Make sure the server is running: python main.py\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Disconnecting...\n")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Musequill WebSocket CLI Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py "Write a story about AI"
  python client.py "Quantum computing overview" --max-iterations 5
  python client.py "Space exploration" --server ws://localhost:8080/ws
        """
    )
    
    parser.add_argument(
        "topic",
        help="The topic for content generation"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations (default: 3)"
    )
    
    parser.add_argument(
        "--server",
        default="ws://localhost:8080/ws",
        help="WebSocket server URL (default: ws://localhost:8080/ws)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" 🎨 MUSEQUILL CLI CLIENT")
    print("="*80)
    
    try:
        asyncio.run(run_client(args.topic, args.max_iterations, args.server))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        sys.exit(0)


if __name__ == "__main__":
    main()