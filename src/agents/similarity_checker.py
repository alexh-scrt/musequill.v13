"""Similarity detection system for preventing content repetition."""

import os
import logging
from typing import List, Optional, Tuple

from src.exceptions import TopicExhaustionError
from src.common import ParagraphMatch, SimilarityResult

# Configure logging
logger = logging.getLogger(__name__)


class SimilarityChecker:
    """Core similarity detection logic for content uniqueness validation."""
    
    def __init__(self, session_id: str, corpus):
        """Initialize the similarity checker.
        
        Args:
            session_id: Unique session identifier
            corpus: SimilarityCorpus instance for storage and search
        """
        self.session_id = session_id
        self.corpus = corpus
        self.threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
        self.max_attempts = int(os.getenv("MAX_SIMILARITY_ATTEMPTS", "5"))
        self._attempts = 0
        
        logger.info(f"Initialized SimilarityChecker for session {session_id}")
        logger.debug(f"Similarity threshold: {self.threshold}, Max attempts: {self.max_attempts}")
    
    async def check_similarity(self, content: str) -> SimilarityResult:
        """Check if content is similar to previously generated content.
        
        Args:
            content: The content to check for similarity
            
        Returns:
            SimilarityResult with detailed similarity information
        """
        try:
            # Import feedback generator here to avoid circular imports
            from src.agents.similarity_feedback import filter_natural_repetitions, generate_similarity_feedback
            
            # Search for similar content in the corpus
            matches = self.corpus.search_similar_content(content)
            
            # Filter out naturally repetitive patterns
            filtered_matches = filter_natural_repetitions(matches)
            
            # Calculate overall similarity
            overall_similarity = max([m.similarity_score for m in filtered_matches], default=0.0)
            
            # Determine if content is unique
            is_unique = overall_similarity < self.threshold
            
            # Get unique paragraph indices
            unique_paragraphs = self.get_unique_paragraph_indices(content, filtered_matches)
            
            # Generate feedback if not unique
            feedback = generate_similarity_feedback(filtered_matches) if not is_unique else ""
            
            # Calculate remaining attempts
            attempts_remaining = max(0, self.max_attempts - self._attempts - 1)
            
            logger.debug(f"Similarity check result: overall={overall_similarity:.2%}, unique={is_unique}, "
                        f"matches={len(filtered_matches)}, attempts_remaining={attempts_remaining}")
            
            return SimilarityResult(
                overall_similarity=overall_similarity,
                is_unique=is_unique,
                paragraph_matches=filtered_matches,
                unique_paragraphs=unique_paragraphs,
                feedback=feedback,
                attempts_remaining=attempts_remaining
            )
            
        except Exception as e:
            logger.error(f"Error checking similarity: {e}", exc_info=True)
            # Return a result that allows content through on error
            return SimilarityResult(
                overall_similarity=0.0,
                is_unique=True,
                unique_paragraphs=list(range(len(content.split("\n\n")))),
                attempts_remaining=max(0, self.max_attempts - self._attempts)
            )
    
    def get_unique_paragraph_indices(self, content: str, matches: List[ParagraphMatch]) -> List[int]:
        """Identify which paragraphs in the content are unique.
        
        Args:
            content: The full content being checked
            matches: List of paragraph matches found
            
        Returns:
            List of indices for paragraphs that are unique
        """
        paragraphs = content.split("\n\n")
        matched_indices = {match.paragraph_index for match in matches}
        unique_indices = [i for i in range(len(paragraphs)) if i not in matched_indices]
        return unique_indices
    
    @property
    def attempts_remaining(self) -> int:
        """Get the number of attempts remaining.
        
        Returns:
            Number of regeneration attempts left
        """
        return max(0, self.max_attempts - self._attempts)
    
    def increment_attempts(self):
        """Increment the attempt counter."""
        self._attempts += 1
        logger.debug(f"Similarity attempts: {self._attempts}/{self.max_attempts}")
        
        if self._attempts >= self.max_attempts:
            logger.warning(f"Max similarity attempts ({self.max_attempts}) reached")
    
    def reset_attempts(self):
        """Reset the attempt counter."""
        self._attempts = 0
        logger.debug("Similarity attempts counter reset")