import logging
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings

from src.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)


class MemoryManager:
    
    def __init__(self, session_id: str, agent_id: str):
        self.session_id = session_id
        self.agent_id = agent_id
        self.redis_client = RedisClient()
        
        chroma_url = os.getenv("CHROMA_URL", "http://localhost:8000")
        chroma_host = chroma_url.replace("http://", "").replace("https://", "").split(":")[0]
        chroma_port = int(chroma_url.split(":")[-1]) if ":" in chroma_url else 8000
        
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False)
            )
            
            collection_name = self._sanitize_collection_name(f"memory_{agent_id}")
            self.memory_collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Conversation memory for {agent_id}"}
            )
            
            logger.info(f"✅ MemoryManager initialized for {agent_id} session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MemoryManager: {e}")
            raise
    
    def _sanitize_collection_name(self, name: str) -> str:
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', name)
        sanitized = sanitized.strip('._-')
        
        if not sanitized or len(sanitized) < 3:
            sanitized = f"memory_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        if len(sanitized) > 50:
            sanitized = sanitized[:50].rstrip('._-')
        
        return sanitized
    
    def _get_redis_key(self) -> str:
        return f"conversation:{self.agent_id}:{self.session_id}"
    
    def _get_chroma_id(self, turn_index: int) -> str:
        return f"{self.session_id}_{self.agent_id}_{turn_index}"
    
    async def add_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        turn_data = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        redis_key = self._get_redis_key()
        await self.redis_client.lpush(redis_key, json.dumps(turn_data))
        
        await self.redis_client.ltrim(redis_key, 0, 49)
        
        try:
            recent_turns = await self.redis_client.lrange(redis_key, 0, -1)
            turn_index = len(recent_turns) - 1
            
            chroma_id = self._get_chroma_id(turn_index)
            chroma_metadata = {
                "session_id": self.session_id,
                "agent_id": self.agent_id,
                "role": role,
                "timestamp": turn_data["timestamp"],
                "turn_index": turn_index
            }
            if metadata:
                chroma_metadata.update(metadata)
            
            self.memory_collection.upsert(
                documents=[content],
                ids=[chroma_id],
                metadatas=[chroma_metadata]
            )
            
            logger.debug(f"Saved turn to Redis and ChromaDB: {role}")
            
        except Exception as e:
            logger.warning(f"Failed to save to ChromaDB: {e}")
    
    async def get_recent_context(self, n: int = 10) -> List[Dict[str, Any]]:
        redis_key = self._get_redis_key()
        turns_json = await self.redis_client.lrange(redis_key, 0, n - 1)
        
        turns = []
        for turn_json in reversed(turns_json):
            try:
                turn = json.loads(turn_json)
                turns.append(turn)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse turn from Redis: {e}")
        
        return turns
    
    def get_relevant_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            results = self.memory_collection.query(
                query_texts=[query],
                n_results=k,
                where={"session_id": self.session_id},
                include=["documents", "metadatas", "distances"]
            )
            
            relevant_turns = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    relevant_turns.append({
                        "content": doc,
                        "metadata": meta,
                        "relevance": 1 - dist
                    })
            
            logger.debug(f"Found {len(relevant_turns)} relevant turns for query")
            return relevant_turns
            
        except Exception as e:
            logger.warning(f"Failed to query relevant context: {e}")
            return []
    
    async def clear_session(self):
        redis_key = self._get_redis_key()
        await self.redis_client.delete(redis_key)
        
        try:
            results = self.memory_collection.get(
                where={"session_id": self.session_id},
                include=["metadatas"]
            )
            
            if results and results.get("ids"):
                self.memory_collection.delete(ids=results["ids"])
                logger.info(f"Cleared {len(results['ids'])} turns from ChromaDB")
                
        except Exception as e:
            logger.warning(f"Failed to clear ChromaDB session: {e}")
    
    async def get_conversation_summary(self, max_turns: int = 20) -> str:
        recent = await self.get_recent_context(n=max_turns)
        
        if not recent:
            return "No conversation history available."
        
        summary_parts = []
        for i, turn in enumerate(recent[-10:], 1):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            preview = content[:150] + "..." if len(content) > 150 else content
            summary_parts.append(f"{i}. {role.upper()}: {preview}")
        
        return "\n".join(summary_parts)