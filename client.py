#!/usr/bin/env python3
"""
Musequill CLI Client with Profile-Aware and Depth Support

Usage:
    # Use WebSocket endpoint (original):
    python client.py "Topic" --profile technology
    
    # Use new profile-aware HTTP endpoint:
    python client.py "Topic" --profile scientific --depth 3 --use-http
    python client.py "Topic" --profile popular_science --single-depth 2 --use-http
"""

import asyncio
import json
import argparse
import sys
import httpx
from datetime import datetime
from websockets import connect
from websockets.exceptions import ConnectionClosed

from src.agents.evaluator_profiles import EvaluatorProfileFactory


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


async def run_websocket_client(
    topic: str, 
    max_iterations: int, 
    server_url: str,
    profile: str
):
    """Run the WebSocket client (original orchestrator)"""
    print(f"\n🔌 Connecting to {server_url}...\n")
    
    try:
        async with connect(server_url) as websocket:
            print("✅ Connected to Musequill WebSocket server\n")
            
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


async def run_http_client(
    topic: str,
    profile: str,
    max_depth: int,
    single_depth: int,
    server_base_url: str,
    stream: bool = False
):
    """Run the HTTP client for profile-aware generation"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Build request payload
        payload = {
            "topic": topic,
            "profile": profile,
            "max_depth": max_depth,
            "stream": stream
        }
        
        if single_depth:
            payload["single_depth"] = single_depth
        
        # Determine endpoint
        endpoint = f"{server_base_url}/api/profile/generate"
        if stream:
            endpoint = f"{server_base_url}/api/profile/generate/stream"
        
        print(f"\n🔌 Sending request to {endpoint}...\n")
        
        try:
            if stream:
                # Streaming response
                async with client.stream("POST", endpoint, json=payload) as response:
                    response.raise_for_status()
                    print("✅ Connected to profile generation endpoint (streaming)\n")
                    
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            msg_type = data.get("type")
                            
                            if msg_type == "status":
                                print(f"🔔 {data.get('message', '')}")
                            elif msg_type == "progress":
                                depth = data.get("depth")
                                total = data.get("total", max_depth)
                                print(f"⏳ Generating depth {depth}/{total}...")
                            elif msg_type == "content":
                                depth = data.get("depth")
                                content = data.get("content", "")
                                metadata = data.get("metadata", {})
                                print(f"\n{'='*80}")
                                print(f"📝 DEPTH {depth} | Decision: {metadata.get('decision', 'UNKNOWN')}")
                                print(f"{'='*80}")
                                print(content[:500] if len(content) > 500 else content)
                                if len(content) > 500:
                                    print("... [content truncated for display]")
                                print(f"{'='*80}\n")
                            elif msg_type == "complete":
                                print(f"\n✅ {data.get('message', 'Generation complete')}")
                            elif msg_type == "error":
                                print(f"\n❌ ERROR: {data.get('error', 'Unknown error')}")
            else:
                # Regular response
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                
                result = response.json()
                print("✅ Generation complete\n")
                
                # Display results
                print(f"📄 Session ID: {result['session_id']}")
                print(f"📊 Profile: {result['profile']}")
                print(f"📚 Depths generated: {result['depths_generated']}")
                print(f"⏱️  Generation time: {result['generation_time']:.2f} seconds")
                
                # Display content for each depth
                for depth in sorted(result['depths_generated']):
                    content = result['content'][str(depth)]
                    metadata = result['metadata'][str(depth)]
                    
                    print(f"\n{'='*80}")
                    print(f"📝 DEPTH {depth}")
                    print(f"   Decision: {metadata.get('decision', 'UNKNOWN')}")
                    print(f"   Attempts: {metadata.get('attempts', 0)}")
                    if 'similarity_score' in metadata:
                        print(f"   Similarity: {metadata['similarity_score']:.2%}")
                    print(f"{'='*80}")
                    print(content[:500] if len(content) > 500 else content)
                    if len(content) > 500:
                        print("... [content truncated for display]")
                    print(f"{'='*80}\n")
                    
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}", file=sys.stderr)
            sys.exit(1)
        except httpx.ConnectError:
            print(f"❌ Could not connect to {server_base_url}", file=sys.stderr)
            print("   Make sure the server is running: python main.py\n", file=sys.stderr)
            sys.exit(1)
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
            print(f"   Critical metrics: {', '.join(info['critical_metrics']) if info['critical_metrics'] else 'None'}")
            print(f"   Top priorities:")
            for name, config in info['top_priorities']:
                print(f"     • {name}: {config['weight']} points")
        except Exception as e:
            print(f"   Error loading details: {e}")
    
    print("\n" + "="*80)
    print("\nNote: For profile-aware generation with depth control, use --use-http flag")
    print("Profile-aware profiles: scientific, popular_science, educational")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Musequill CLI Client with Profile and Depth Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original WebSocket orchestrator:
  python client.py "Topic about AI" --profile technology
  
  # New profile-aware HTTP endpoint with depth control:
  python client.py "Quantum computing" --profile scientific --depth 3 --use-http
  python client.py "Black holes" --profile popular_science --single-depth 2 --use-http
  python client.py "Entropy" --profile educational --depth 2 --use-http --stream
  
  # List available profiles:
  python client.py --list-profiles
  
Available Profiles (WebSocket):
  - scientific, popular_science, technology, investment, general, creative
  
Available Profiles (HTTP with --use-http):
  - scientific: Technical content with equations and citations
  - popular_science: Accessible content without jargon
  - educational: Progressive learning with examples
        """
    )
    
    parser.add_argument(
        "topic",
        nargs='?',
        help="The topic for content generation"
    )
    
    parser.add_argument(
        "--profile",
        default="general",
        help="Content profile to use (default: general)"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Maximum depth level for generation (HTTP only, default: 3)"
    )
    
    parser.add_argument(
        "--single-depth",
        type=int,
        choices=[1, 2, 3],
        help="Generate only a single depth level (HTTP only)"
    )
    
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Use HTTP endpoint for profile-aware generation instead of WebSocket"
    )
    
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming response (HTTP only)"
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
        help="Maximum number of refinement iterations (WebSocket only, default: 3)"
    )
    
    parser.add_argument(
        "--server",
        default="localhost:8080",
        help="Server address (default: localhost:8080)"
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
    print(f"📊 Profile: {args.profile}")
    
    if args.use_http:
        # Use HTTP endpoint for profile-aware generation
        print(f"🔧 Mode: Profile-Aware Generation (HTTP)")
        if args.single_depth:
            print(f"📚 Single Depth: {args.single_depth}")
        else:
            print(f"📚 Max Depth: {args.depth}")
        print(f"🔄 Streaming: {args.stream}")
        
        # Validate profile for HTTP endpoint
        valid_http_profiles = ["scientific", "popular_science", "educational"]
        if args.profile not in valid_http_profiles:
            print(f"\n⚠️  Warning: Profile '{args.profile}' may not be available for HTTP endpoint")
            print(f"   Available profiles: {', '.join(valid_http_profiles)}")
            print("   Attempting anyway...\n")
        
        server_base_url = f"http://{args.server}"
        
        try:
            asyncio.run(run_http_client(
                args.topic,
                args.profile,
                args.depth,
                args.single_depth,
                server_base_url,
                args.stream
            ))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user\n")
            sys.exit(0)
    else:
        # Use WebSocket endpoint (original orchestrator)
        print(f"🔧 Mode: Orchestrator Workflow (WebSocket)")
        print(f"🔄 Max Iterations: {args.max_iterations}")
        
        server_url = f"ws://{args.server}/ws"
        
        try:
            asyncio.run(run_websocket_client(
                args.topic, 
                args.max_iterations, 
                server_url,
                args.profile
            ))
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user\n")
            sys.exit(0)


if __name__ == "__main__":
    main()