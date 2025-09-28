import logging
import os
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manage chapter content and summaries in ChromaDB for efficient context retrieval"""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize ChromaDB connection and collections
        
        Args:
            session_id: Optional session identifier for collection naming
        """
        raw_session_id = session_id or self._generate_session_id()
        # Sanitize session_id to meet ChromaDB naming requirements
        # Remove trailing underscores and ensure it's valid
        self.session_id = self._sanitize_collection_name(raw_session_id)
        
        # Connect to ChromaDB
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        try:
            # Connect to ChromaDB server
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collections with session-specific names
            self.chapters_collection = self.client.get_or_create_collection(
                name=f"chapters_{self.session_id}",
                metadata={"description": "Expanded chapter content"}
            )
            
            self.summaries_collection = self.client.get_or_create_collection(
                name=f"summaries_{self.session_id}",
                metadata={"description": "Running story summaries"}
            )
            
            logger.info(f"ChromaDB initialized for session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to meet ChromaDB requirements
        
        ChromaDB requires names to:
        - Be 3-512 characters long
        - Contain only [a-zA-Z0-9._-]
        - Start and end with [a-zA-Z0-9]
        
        Args:
            name: Raw name to sanitize
            
        Returns:
            Sanitized name that meets ChromaDB requirements
        """
        import re
        
        # Remove any characters that aren't alphanumeric, dot, underscore, or hyphen
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', name)
        
        # Remove leading/trailing non-alphanumeric characters
        sanitized = sanitized.strip('._-')
        
        # If empty or too short, generate a default
        if not sanitized or len(sanitized) < 3:
            sanitized = f"session_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Truncate if too long (leave room for prefixes like "chapters_" or "summaries_")
        if len(sanitized) > 50:  # Conservative limit to allow for prefixes
            sanitized = sanitized[:50].rstrip('._-')
        
        return sanitized
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID based on timestamp"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def store_chapter(self, chapter_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Store an expanded chapter with its metadata
        
        Args:
            chapter_id: Unique identifier for the chapter (e.g., "chapter_1")
            content: The expanded chapter content
            metadata: Additional metadata (index, title, quality_score, etc.)
        """
        try:
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.now().isoformat()
            metadata["chapter_id"] = chapter_id
            
            # Store in ChromaDB
            self.chapters_collection.upsert(
                documents=[content],
                ids=[chapter_id],
                metadatas=[metadata]
            )
            
            logger.info(f"Stored chapter {chapter_id} with {len(content)} characters")
            
        except Exception as e:
            logger.error(f"Failed to store chapter {chapter_id}: {str(e)}")
            raise
    
    def get_chapter(self, chapter_id: str) -> Optional[str]:
        """Retrieve a specific chapter's content
        
        Args:
            chapter_id: The chapter identifier
            
        Returns:
            Chapter content or None if not found
        """
        try:
            results = self.chapters_collection.get(ids=[chapter_id])
            
            if results and results["documents"]:
                return results["documents"][0]
            
            logger.warning(f"Chapter {chapter_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve chapter {chapter_id}: {str(e)}")
            return None
    
    def get_previous_chapters(self, current_index: int, limit: int = 2) -> List[Dict[str, str]]:
        """Get the most recent previous chapters
        
        Args:
            current_index: Current chapter index
            limit: Maximum number of previous chapters to retrieve
            
        Returns:
            List of chapter dictionaries with content and metadata
        """
        try:
            # Query for chapters with index less than current
            where_clause = {"index": {"$lt": current_index}}
            
            results = self.chapters_collection.query(
                query_texts=[""],  # Empty query to get all matching
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas"]
            )
            
            chapters = []
            if results["documents"] and results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    chapters.append({
                        "content": doc,
                        "metadata": meta
                    })
            
            # Sort by index to maintain order
            chapters.sort(key=lambda x: x["metadata"].get("index", 0))
            
            logger.info(f"Retrieved {len(chapters)} previous chapters")
            return chapters
            
        except Exception as e:
            logger.error(f"Failed to get previous chapters: {str(e)}")
            return []
    
    def update_summary(self, summary: str, chapter_count: int = 0) -> None:
        """Store or update the running summary
        
        Args:
            summary: The updated summary text
            chapter_count: Number of chapters included in summary
        """
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "chapter_count": chapter_count,
                "version": "latest"
            }
            
            # Always use same ID for the running summary
            self.summaries_collection.upsert(
                documents=[summary],
                ids=["current_summary"],
                metadatas=[metadata]
            )
            
            logger.info(f"Updated summary with {len(summary)} characters, covering {chapter_count} chapters")
            
        except Exception as e:
            logger.error(f"Failed to update summary: {str(e)}")
            raise
    
    def get_summary(self) -> Optional[str]:
        """Retrieve the current running summary
        
        Returns:
            Summary text or None if not found
        """
        try:
            results = self.summaries_collection.get(ids=["current_summary"])
            
            if results and results["documents"]:
                return results["documents"][0]
            
            logger.info("No summary found, starting fresh")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve summary: {str(e)}")
            return None
    
    def search_relevant_content(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Vector search for relevant content across all chapters
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of relevant content snippets with metadata
        """
        try:
            results = self.chapters_collection.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            relevant_content = []
            if results["documents"] and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0], 
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Extract a relevant snippet (first 500 chars for context)
                    snippet = doc[:500] + "..." if len(doc) > 500 else doc
                    
                    relevant_content.append({
                        "snippet": snippet,
                        "metadata": meta,
                        "relevance_score": 1 - dist  # Convert distance to similarity
                    })
            
            logger.info(f"Found {len(relevant_content)} relevant content pieces for query")
            return relevant_content
            
        except Exception as e:
            logger.error(f"Failed to search content: {str(e)}")
            return []
    
    def get_all_chapters(self) -> List[Dict[str, Any]]:
        """Retrieve all chapters in order
        
        Returns:
            List of all chapters with content and metadata
        """
        try:
            results = self.chapters_collection.get(
                include=["documents", "metadatas"]
            )
            
            chapters = []
            if results["documents"]:
                # IDs are returned automatically even if not requested
                ids = results.get("ids", [f"chapter_{i}" for i in range(len(results["documents"]))])
                for i, (doc, meta) in enumerate(zip(
                    results["documents"],
                    results["metadatas"]
                )):
                    chapters.append({
                        "id": ids[i] if i < len(ids) else f"chapter_{i}",
                        "content": doc,
                        "metadata": meta
                    })
            
            # Sort by index to maintain chapter order
            chapters.sort(key=lambda x: x["metadata"].get("index", 0))
            
            logger.info(f"Retrieved {len(chapters)} total chapters")
            return chapters
            
        except Exception as e:
            logger.error(f"Failed to get all chapters: {str(e)}")
            return []
    
    def cleanup_session(self) -> None:
        """Clean up collections for this session"""
        try:
            self.client.delete_collection(f"chapters_{self.session_id}")
            self.client.delete_collection(f"summaries_{self.session_id}")
            logger.info(f"Cleaned up session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup session: {str(e)}")