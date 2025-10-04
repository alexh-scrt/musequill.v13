#!/usr/bin/env python3
"""
Enhanced Musequill CLI Client with Evaluator Profile Support

Usage:
    python client.py "Topic" --profile technology
    python client.py "Topic" --profile scientific
    python client.py "Topic" --list-profiles
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime
from websockets import connect
from websockets.exceptions import ConnectionClosed

from src.agents.profiles import EvaluatorProfileFactory


async def send_content_request(
    websocket, 
    topic: str, 
    max_iterations: int = 3,
    profile: str = "general"
):
    """Send content request with profile specification"""
    request = {
        "type": "content_request",
        "data": {
            "topic": topic,
            "max_iterations": max_iterations,
            "evaluator_profile": profile
        },
        "workflow": "orchestrator"
    }
    await websocket.send(json.dumps(request))
    print(f"📤 Sent content request for topic: {topic}")
    print(f"   Profile: {profile}")
    print()


async def receive_messages(websocket):
    """Receive and display messages from server"""
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
                await websocket.send(json.dumps({
                    "type": "pong", 
                    "timestamp": datetime.now().isoformat()
                }))
            
            elif msg_type == "pong":
                pass
            
            else:
                print(f"📨 [{timestamp}] {msg_type.upper()}: {content}")
    
    except ConnectionClosed:
        print("\n⚠️  Connection closed by server\n")
    except Exception as e:
        print(f"\n❌ Error receiving messages: {e}\n", file=sys.stderr)


async def run_client(
    topic: str, 
    max_iterations: int, 
    server_url: str,
    profile: str
):
    """Run the client with specified parameters"""
    print(f"\n🔌 Connecting to {server_url}...\n")
    
    try:
        async with connect(server_url) as websocket:
            print("✅ Connected to Musequill server\n")
            
            await send_content_request(websocket, topic, max_iterations, profile)
            
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


def list_profiles():
    """Display available evaluator profiles"""
    print("\n" + "="*80)
    print(" 📊 AVAILABLE EVALUATOR PROFILES")
    print("="*80)
    
    profiles = EvaluatorProfileFactory.list_profiles()
    
    for profile_id, description in profiles.items():
        print(f"\n🔹 {profile_id}")
        print(f"   {description}")
        
        # Get detailed info
        try:
            info = EvaluatorProfileFactory.get_profile_info(profile_id)
            print(f"   Critical metrics: {', '.join(info['critical_metrics'])}")
            print(f"   Top priorities:")
            for name, config in info['top_priorities']:
                print(f"     • {name}: {config['weight']} points")
        except Exception as e:
            print(f"   Error loading details: {e}")
    
    print("\n" + "="*80)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Musequill WebSocket CLI Client with Evaluator Profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available profiles
  python client.py --list-profiles
  
  # Use technology profile for tech content
  python client.py "Review of the new M4 MacBook Pro" --profile technology
  
  # Use scientific profile for research content
  python client.py "Quantum computing advances" --profile scientific
  
  # Use investment profile for financial analysis
  python client.py "Analysis of tech sector valuations" --profile investment
  
  # Use popular science for accessible explanations
  python client.py "How does CRISPR gene editing work?" --profile popular_science
  
  # Use creative profile for storytelling
  python client.py "Write a sci-fi story about AI" --profile creative
  
Available Profiles:
  - scientific: Rigorous academic/research content
  - popular_science: Accessible science communication
  - technology: Tech reviews and tutorials
  - investment: Financial analysis and insights
  - general: Balanced, broad-audience content (default)
  - creative: Narrative and storytelling
        """
    )
    
    parser.add_argument(
        "topic",
        nargs='?',
        help="The topic for content generation"
    )
    
    parser.add_argument(
        "--profile",
        choices=[
            "scientific",
            "popular_science",
            "technology",
            "investment",
            "general",
            "creative"
        ],
        default="general",
        help="Evaluator profile to use (default: general)"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all available evaluator profiles and exit"
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
    
    # Handle --list-profiles
    if args.list_profiles:
        list_profiles()
        sys.exit(0)
    
    # Require topic if not listing profiles
    if not args.topic:
        parser.error("the following arguments are required: topic")
    
    print("\n" + "="*80)
    print(" 🎨 MUSEQUILL CLI CLIENT")
    print("="*80)
    print(f"\n📋 Topic: {args.topic}")
    print(f"📊 Evaluator Profile: {args.profile}")
    print(f"🔄 Max Iterations: {args.max_iterations}")
    print()
    
    try:
        asyncio.run(run_client(
            args.topic, 
            args.max_iterations, 
            args.server,
            args.profile
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
        sys.exit(0)


if __name__ == "__main__":
    main()