# Testing Guide

## Pre-flight Check

Run the setup test to verify all components:
```bash
python3 test_setup.py
```

Expected output: ✅ ALL TESTS PASSED

## Service Status

Check Docker services:
```bash
docker compose ps
```

Expected services running:
- ✅ poc-redis (port 16379)
- ✅ poc-chromadb (port 8000)  
- ✅ poc-ollama (port 11434)

## Start the Server

```bash
python main.py
```

Server will start on `http://localhost:8080`

Check health:
```bash
curl http://localhost:8080/health
```

## Test with CLI Client

### Basic conversation:
```bash
python client.py "What is the meaning of life?"
```

### Multi-turn conversation test:
The agents will automatically maintain context:
1. Generator responds with answer + follow-up question
2. Discriminator provides deeper insight + new question
3. Continues until Discriminator outputs "STOP"

### Example flow:
```bash
python client.py "Tell me about quantum computing"
```

**Expected behavior:**
- Generator gives concise answer + asks follow-up
- Discriminator adds depth + asks deeper question
- Generator responds with context from previous turns
- Continues cycling until topic is exhausted
- Discriminator outputs "STOP" when complete

## Memory Verification

### Check Redis (recent context):
```python
python3 -c "
import asyncio
from src.storage.redis_client import RedisClient

async def check():
    r = RedisClient()
    await r.connect()
    turns = await r.lrange('conversation:generator:YOUR_SESSION_ID', 0, -1)
    print(f'Stored {len(turns)} turns in Redis')
    await r.disconnect()

asyncio.run(check())
"
```

### Check ChromaDB (semantic memory):
Visit: http://localhost:8000/docs

Or query collections:
```python
python3 -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
collections = client.list_collections()
print(f'Collections: {[c.name for c in collections]}')
"
```

## Key Features to Test

### 1. Context Retention
- **Test:** Send multiple messages in sequence
- **Expected:** Agents reference previous exchanges

### 2. Semantic Memory
- **Test:** Ask similar question after several turns
- **Expected:** Agent finds relevant past context

### 3. Session Isolation
- **Test:** Start two separate client sessions
- **Expected:** Each maintains independent memory

### 4. Conversation Termination
- **Test:** Deep dive into a topic until exhausted
- **Expected:** Discriminator outputs "STOP"

### 5. Error Handling
- **Test:** Stop Redis: `docker compose stop redis`
- **Expected:** Graceful degradation (in-memory fallback)

## Troubleshooting

### "Connection refused" error:
```bash
docker compose up -d
```

### "Model not found" error:
Check available models:
```bash
curl http://localhost:11434/api/tags
```

### Memory not persisting:
Check `.env` has correct URLs:
```
REDIS_URL=redis://localhost:16379
CHROMA_URL=http://localhost:8000
```

### Agent not using context:
Check logs for:
- "✅ Redis OK"
- "✅ MemoryManager initialized"
- "Retrieved X turns from Redis"

## Cleanup

Remove test data:
```bash
# Clear Redis
docker compose exec redis redis-cli FLUSHDB

# Clear ChromaDB collections (done automatically per session)
```

Stop services:
```bash
docker compose down
```