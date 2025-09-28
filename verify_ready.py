#!/usr/bin/env python3
import asyncio
import sys
import os

async def verify():
    print("=" * 70)
    print("  MUSEQUILL PRE-FLIGHT VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    print("\n📦 1. Checking Python packages...")
    packages = {
        'redis': 'redis',
        'chromadb': 'chromadb', 
        'websockets': 'websockets',
        'fastapi': 'fastapi',
        'langgraph': 'langgraph'
    }
    
    for pkg_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {pkg_name}")
        except ImportError:
            print(f"   ❌ {pkg_name} - MISSING")
            all_good = False
    
    print("\n🐳 2. Checking Docker services...")
    services = [
        ('Redis', 'redis://localhost:16379'),
        ('ChromaDB', 'http://localhost:8000/api/v2/heartbeat'),
        ('Ollama', 'http://localhost:11434/api/tags')
    ]
    
    import redis.asyncio as redis
    try:
        r = await redis.from_url('redis://localhost:16379')
        await r.ping()
        print("   ✅ Redis (port 16379)")
        await r.aclose()
    except Exception as e:
        print(f"   ❌ Redis - {e}")
        all_good = False
    
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get('http://localhost:8000/api/v2/heartbeat', timeout=5.0)
            if resp.status_code == 200:
                print("   ✅ ChromaDB (port 8000)")
            else:
                print(f"   ❌ ChromaDB - HTTP {resp.status_code}")
                all_good = False
    except Exception as e:
        print(f"   ❌ ChromaDB - {e}")
        all_good = False
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get('http://localhost:11434/api/tags', timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                print(f"   ✅ Ollama (port 11434) - {len(models)} models")
            else:
                print(f"   ❌ Ollama - HTTP {resp.status_code}")
                all_good = False
    except Exception as e:
        print(f"   ❌ Ollama - {e}")
        all_good = False
    
    print("\n⚙️  3. Checking environment variables...")
    env_vars = {
        'REDIS_URL': 'redis://localhost:16379',
        'CHROMA_URL': 'http://localhost:8000',
        'OLLAMA_BASE_URL': 'http://localhost:11434',
        'OLLAMA_MODEL': 'qwen3:8b'
    }
    
    for var, default in env_vars.items():
        value = os.getenv(var, default)
        print(f"   ✅ {var}={value}")
    
    print("\n🧪 4. Testing memory components...")
    try:
        from src.storage.redis_client import RedisClient
        from src.storage.memory_manager import MemoryManager
        
        rc = RedisClient()
        await rc.connect()
        await rc.set("verify_test", "ok")
        result = await rc.get("verify_test")
        await rc.delete("verify_test")
        await rc.disconnect()
        
        if result == "ok":
            print("   ✅ RedisClient works")
        else:
            print("   ❌ RedisClient - unexpected result")
            all_good = False
        
        mem = MemoryManager("verify_session", "verify_agent")
        await mem.add_turn("user", "test message")
        turns = await mem.get_recent_context(n=1)
        await mem.clear_session()
        
        if len(turns) == 1:
            print("   ✅ MemoryManager works")
        else:
            print("   ❌ MemoryManager - unexpected result")
            all_good = False
            
    except Exception as e:
        print(f"   ❌ Memory components - {e}")
        all_good = False
    
    print("\n🤖 5. Testing agent initialization...")
    try:
        from src.agents.generator import GeneratorAgent
        from src.agents.discriminator import DiscriminatorAgent
        from src.workflow.orchestrator import WorkflowOrchestrator
        
        gen = GeneratorAgent(session_id="verify_session")
        disc = DiscriminatorAgent(session_id="verify_session")
        orch = WorkflowOrchestrator()
        
        print(f"   ✅ GeneratorAgent (model: {gen.model})")
        print(f"   ✅ DiscriminatorAgent (model: {disc.model})")
        print(f"   ✅ WorkflowOrchestrator (session: {orch.session_id})")
        
    except Exception as e:
        print(f"   ❌ Agent initialization - {e}")
        all_good = False
    
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - READY FOR TESTING!")
        print("=" * 70)
        print("\n🚀 To start testing:")
        print("   Terminal 1: python main.py")
        print("   Terminal 2: python client.py \"Your message here\"")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(verify())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(1)