"""Exception classes for similarity detection system."""

from typing import Optional, Dict, Any


class TopicExhaustionError(Exception):
    """Raised when max similarity attempts are exhausted without finding unique content.
    
    This exception indicates that the system has tried multiple times to generate
    unique content but keeps producing similar results, suggesting the topic may
    be exhausted or too constrained.
    """
    
    def __init__(
        self, 
        message: str, 
        attempts: Optional[int] = None,
        best_similarity: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize TopicExhaustionError.
        
        Args:
            message: Error message describing the exhaustion
            attempts: Number of attempts made before exhaustion
            best_similarity: Best (lowest) similarity score achieved
            metadata: Additional context about the exhaustion
        """
        super().__init__(message)
        self.attempts = attempts
        self.best_similarity = best_similarity
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        base_msg = super().__str__()
        if self.attempts is not None:
            base_msg += f" (attempts: {self.attempts}"
            if self.best_similarity is not None:
                base_msg += f", best_similarity: {self.best_similarity:.2%}"
            base_msg += ")"
        return base_msg


class SimilarityCorpusError(Exception):
    """Raised when there are errors with the similarity corpus storage.
    
    This exception indicates issues with ChromaDB operations, collection
    management, or document storage/retrieval in the similarity corpus.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None):
        """Initialize SimilarityCorpusError.
        
        Args:
            message: Error message describing the corpus issue
            operation: The operation that failed (e.g., 'store', 'search', 'clear')
        """
        super().__init__(message)
        self.operation = operation
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        base_msg = super().__str__()
        if self.operation:
            base_msg = f"[{self.operation}] {base_msg}"
        return base_msg