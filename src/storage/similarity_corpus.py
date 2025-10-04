"""ChromaDB-based corpus for storing and searching similar content."""

import os
import logging
import re
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import chromadb
from chromadb.config import Settings

from src.common import ParagraphMatch
from src.exceptions import SimilarityCorpusError

# Configure logging
logger = logging.getLogger(__name__)


class SimilarityCorpus:
    """Manages a corpus of generated content for similarity detection."""
    
    def __init__(self, session_id: str):
        """Initialize the similarity corpus with ChromaDB backend.
        
        Args:
            session_id: Unique session identifier for this corpus
        """
        self.session_id = session_id
        
        # Load configuration
        self.paragraph_min_length = int(os.getenv("PARAGRAPH_MIN_LENGTH", "50"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        self.sliding_window_size = int(os.getenv("SLIDING_WINDOW_SIZE", "3"))
        self.sliding_window_activation_threshold = int(
            os.getenv("SLIDING_WINDOW_ACTIVATION_THRESHOLD", "100")
        )
        
        # Connect to ChromaDB
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        try:
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collection for similarity corpus
            collection_name = self._sanitize_collection_name(f"similarity_corpus_{session_id}")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Similarity detection corpus", "session_id": session_id}
            )
            
            logger.info(f"SimilarityCorpus initialized for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SimilarityCorpus: {str(e)}")
            raise SimilarityCorpusError(f"Failed to initialize corpus: {str(e)}", "init")
    
    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to meet ChromaDB requirements.
        
        ChromaDB requires names to:
        - Be 3-512 characters long
        - Contain only [a-zA-Z0-9._-]
        - Start and end with [a-zA-Z0-9]
        
        Args:
            name: Raw name to sanitize
            
        Returns:
            Sanitized name that meets ChromaDB requirements
        """
        # Remove any characters that aren't alphanumeric, dot, underscore, or hyphen
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', name)
        
        # Remove leading/trailing non-alphanumeric characters
        sanitized = sanitized.strip('._-')
        
        # If empty or too short, generate a default
        if not sanitized or len(sanitized) < 3:
            sanitized = f"corpus_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Truncate if too long
        if len(sanitized) > 512:
            sanitized = sanitized[:512].rstrip('._-')
        
        return sanitized
    
    def store_content(self, content: str, agent_id: str, metadata: Dict[str, Any]) -> str:
        """Store content in the similarity corpus.
        
        Args:
            content: The full content to store
            agent_id: ID of the agent that generated this content
            metadata: Additional metadata about the content
            
        Returns:
            Unique ID for the stored content
            
        Raises:
            SimilarityCorpusError: If storage fails
        """
        try:
            # Generate unique ID
            iteration = metadata.get("iteration", 0)
            revision = metadata.get("revision_number", 0)
            content_id = f"{agent_id}_iter{iteration}_rev{revision}_{datetime.now().timestamp()}"
            
            # Store full content as one document
            full_metadata = {
                **metadata,
                "agent_id": agent_id,
                "content_id": content_id,
                "is_chunk": False,
                "timestamp": datetime.now().isoformat()
            }
            
            self.collection.add(
                documents=[content],
                metadatas=[full_metadata],
                ids=[content_id]
            )
            
            # Split and store paragraphs as chunks
            paragraphs = self._split_paragraphs(content)
            
            if paragraphs:
                chunk_ids = []
                chunk_docs = []
                chunk_metadatas = []
                
                for idx, paragraph in enumerate(paragraphs):
                    chunk_id = f"{content_id}_chunk_{idx}"
                    chunk_metadata = {
                        "parent_id": content_id,
                        "chunk_index": idx,
                        "chunk_text": paragraph[:500],  # Store preview
                        "is_chunk": True,
                        "agent_id": agent_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    chunk_ids.append(chunk_id)
                    chunk_docs.append(paragraph)
                    chunk_metadatas.append(chunk_metadata)
                
                # Store all chunks in batch
                if chunk_docs:
                    self.collection.add(
                        documents=chunk_docs,
                        metadatas=chunk_metadatas,
                        ids=chunk_ids
                    )
            
            logger.info(f"Stored content {content_id} with {len(paragraphs)} paragraphs")
            return content_id
            
        except Exception as e:
            logger.error(f"Failed to store content: {str(e)}", exc_info=True)
            raise SimilarityCorpusError(f"Failed to store content: {str(e)}", "store")
    
    def _split_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs for chunk-level storage.
        
        Args:
            content: The content to split
            
        Returns:
            List of paragraph strings
        """
        # Split on double newlines
        paragraphs = content.split("\n\n")
        
        # Filter and clean paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            # Only keep paragraphs above minimum length
            if len(para) >= self.paragraph_min_length:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def search_similar_paragraphs(self, paragraph: str, threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar paragraphs in the corpus.
        
        Args:
            paragraph: The paragraph to search for
            threshold: Similarity threshold (uses default if not provided)
            
        Returns:
            List of similar paragraphs with metadata
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        try:
            # Query ChromaDB for similar chunks
            results = self.collection.query(
                query_texts=[paragraph],
                n_results=5,
                where={"is_chunk": True},
                include=["documents", "metadatas", "distances"]
            )
            
            matches = []
            if results and results["documents"] and results["documents"][0]:
                for idx, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    similarity = 1.0 - dist
                    
                    # Filter by threshold
                    if similarity >= threshold:
                        matches.append({
                            "document": doc,
                            "metadata": meta,
                            "similarity": similarity
                        })
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search similar paragraphs: {str(e)}", exc_info=True)
            return []
    
    def search_similar_content(self, content: str) -> List[ParagraphMatch]:
        """Search for similar content in the corpus.
        
        Args:
            content: The content to check for similarity
            
        Returns:
            List of ParagraphMatch objects for similar paragraphs
        """
        paragraphs = self._split_paragraphs(content)
        all_matches = []
        
        # Check if we should use sliding window
        use_sliding = self.should_use_sliding_window()
        
        if use_sliding:
            logger.debug("Using sliding window search for large corpus")
            return self._sliding_window_search(paragraphs)
        
        # Standard paragraph-by-paragraph search
        for para_idx, paragraph in enumerate(paragraphs):
            similar = self.search_similar_paragraphs(paragraph)
            
            for match in similar:
                paragraph_match = ParagraphMatch(
                    query_paragraph=paragraph,
                    matched_paragraph=match["document"],
                    similarity_score=match["similarity"],
                    stored_content_id=match["metadata"].get("parent_id", "unknown"),
                    paragraph_index=para_idx,
                    matched_index=match["metadata"].get("chunk_index", -1)
                )
                all_matches.append(paragraph_match)
        
        return all_matches
    
    def should_use_sliding_window(self) -> bool:
        """Check if sliding window search should be activated.
        
        Returns:
            True if corpus is large enough for sliding window
        """
        try:
            # Count total chunks in corpus
            chunk_count = self.collection.count()
            return chunk_count > self.sliding_window_activation_threshold
            
        except Exception as e:
            logger.error(f"Failed to check corpus size: {str(e)}")
            return False
    
    def _sliding_window_search(self, paragraphs: List[str]) -> List[ParagraphMatch]:
        """Perform sliding window search for better context matching.
        
        Args:
            paragraphs: List of paragraphs to search
            
        Returns:
            List of ParagraphMatch objects
        """
        matches = []
        window_size = self.sliding_window_size
        
        for i in range(len(paragraphs) - window_size + 1):
            # Create window of consecutive paragraphs
            window = paragraphs[i:i + window_size]
            concatenated = "\n\n".join(window)
            
            # Search for similar windows
            similar = self.search_similar_paragraphs(concatenated)
            
            # Map results back to individual paragraphs
            for match in similar:
                for j, para in enumerate(window):
                    paragraph_match = ParagraphMatch(
                        query_paragraph=para,
                        matched_paragraph=match["document"],
                        similarity_score=match["similarity"],
                        stored_content_id=match["metadata"].get("parent_id", "unknown"),
                        paragraph_index=i + j,
                        matched_index=match["metadata"].get("chunk_index", -1)
                    )
                    matches.append(paragraph_match)
        
        return matches
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the similarity corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        try:
            # Get all documents
            all_docs = self.collection.get(include=["metadatas"])
            
            if not all_docs or not all_docs["metadatas"]:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "agents": [],
                    "oldest_timestamp": None
                }
            
            # Separate documents and chunks
            documents = [m for m in all_docs["metadatas"] if not m.get("is_chunk", False)]
            chunks = [m for m in all_docs["metadatas"] if m.get("is_chunk", False)]
            
            # Get unique agents
            agents = list(set(m.get("agent_id", "unknown") for m in documents))
            
            # Find oldest timestamp
            timestamps = [m.get("timestamp") for m in documents if m.get("timestamp")]
            oldest = min(timestamps) if timestamps else None
            
            return {
                "total_documents": len(documents),
                "total_chunks": len(chunks),
                "agents": agents,
                "oldest_timestamp": oldest
            }
            
        except Exception as e:
            logger.error(f"Failed to get corpus stats: {str(e)}", exc_info=True)
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "agents": [],
                "oldest_timestamp": None,
                "error": str(e)
            }
    
    def clear_session(self):
        """Clear the collection for this session."""
        try:
            collection_name = self._sanitize_collection_name(f"similarity_corpus_{self.session_id}")
            self.client.delete_collection(name=collection_name)
            logger.info(f"Cleared similarity corpus for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear session corpus: {str(e)}", exc_info=True)