"""Common data types for similarity detection system."""

from typing import List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParagraphMatch:
    """Represents a similarity match between two paragraphs.
    
    Attributes:
        query_paragraph: The paragraph being checked for similarity
        matched_paragraph: The similar paragraph found in the corpus
        similarity_score: Similarity score between 0.0 and 1.0 (1.0 = identical)
        stored_content_id: ID of the stored content containing the match
        paragraph_index: Index of the query paragraph in the original content
        matched_index: Index of the matched paragraph in stored content
    """
    query_paragraph: str
    matched_paragraph: str
    similarity_score: float
    stored_content_id: str
    paragraph_index: int
    matched_index: int


@dataclass(frozen=True)
class SimilarityResult:
    """Result of a similarity check operation.
    
    Attributes:
        overall_similarity: Maximum similarity score found across all paragraphs
        is_unique: Whether content is considered unique (below threshold)
        paragraph_matches: List of paragraph-level matches found
        unique_paragraphs: List of paragraph indices that are unique
        feedback: Human-readable feedback about similarity issues
        attempts_remaining: Number of regeneration attempts left
    """
    overall_similarity: float
    is_unique: bool
    paragraph_matches: List[ParagraphMatch] = field(default_factory=list)
    unique_paragraphs: List[int] = field(default_factory=list)
    feedback: str = ""
    attempts_remaining: int = 0