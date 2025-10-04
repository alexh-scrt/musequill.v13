"""Utilities for generating feedback about content similarity."""

import re
from typing import List

from src.common import ParagraphMatch

# Patterns that indicate naturally repetitive content that should be ignored
IGNORE_PATTERNS = [
    r'^As (discussed|mentioned|stated) (earlier|previously|before)',
    r'^(Recall|Remember) (that|from|how)',
    r'^In summary',
    r'^To recap',
    r'^Follow-up:',
]


def is_naturally_repetitive(paragraph: str) -> bool:
    """Check if a paragraph contains naturally repetitive phrases.
    
    Args:
        paragraph: The paragraph text to check
        
    Returns:
        True if the paragraph matches natural repetition patterns
    """
    for pattern in IGNORE_PATTERNS:
        if re.match(pattern, paragraph, re.IGNORECASE):
            return True
    return False


def filter_natural_repetitions(matches: List[ParagraphMatch]) -> List[ParagraphMatch]:
    """Filter out matches that are naturally repetitive.
    
    Args:
        matches: List of paragraph matches to filter
        
    Returns:
        Filtered list excluding natural repetitions
    """
    filtered = []
    for match in matches:
        if not is_naturally_repetitive(match.query_paragraph):
            filtered.append(match)
    return filtered


def generate_similarity_feedback(matches: List[ParagraphMatch]) -> str:
    """Generate human-readable feedback about similarity issues.
    
    Args:
        matches: List of paragraph matches found
        
    Returns:
        Formatted feedback string for the user/agent
    """
    # Filter natural repetitions first
    filtered_matches = filter_natural_repetitions(matches)
    
    if not filtered_matches:
        return ""
    
    # Categorize matches by similarity level
    high_similarity = []
    medium_similarity = []
    
    for match in filtered_matches:
        if match.similarity_score >= 0.90:
            high_similarity.append(match)
        elif match.similarity_score >= 0.85:
            medium_similarity.append(match)
    
    # Build feedback sections
    feedback_parts = []
    
    # Header
    feedback_parts.append(f"⚠️ Content Similarity Detected ({len(filtered_matches)} similar paragraphs found)\n")
    feedback_parts.append("=" * 60 + "\n")
    
    # Critical matches
    if high_similarity:
        feedback_parts.append("\n🔴 CRITICAL - Near-duplicate paragraphs:\n")
        feedback_parts.append("-" * 40 + "\n")
        
        for idx, match in enumerate(high_similarity[:3], 1):  # Top 3 high similarity
            preview = match.query_paragraph[:100]
            if len(match.query_paragraph) > 100:
                preview += "..."
            
            feedback_parts.append(
                f"{idx}. Paragraph {match.paragraph_index + 1}: {match.similarity_score:.0%} similar\n"
                f"   Preview: {preview}\n"
                f"   ➜ REWRITE this section with a fresh perspective.\n\n"
            )
    
    # Medium similarity matches
    if medium_similarity:
        feedback_parts.append("\n🟡 Moderately similar paragraphs:\n")
        feedback_parts.append("-" * 40 + "\n")
        
        for idx, match in enumerate(medium_similarity[:2], 1):  # Top 2 medium similarity
            preview = match.query_paragraph[:100]
            if len(match.query_paragraph) > 100:
                preview += "..."
            
            feedback_parts.append(
                f"{idx}. Paragraph {match.paragraph_index + 1}: {match.similarity_score:.0%} similar\n"
                f"   Preview: {preview}\n"
                f"   ➜ Consider rephrasing or adding new details.\n\n"
            )
    
    # Suggestions section
    feedback_parts.append("\n💡 Suggestions to avoid repetition:\n")
    feedback_parts.append("-" * 40 + "\n")
    
    suggestions = [
        "• Introduce new perspectives or angles on the topic",
        "• Add specific examples or case studies not previously mentioned",
        "• Explore different aspects or dimensions of the subject",
        "• Use varied vocabulary and sentence structures",
        "• Focus on unique insights rather than restating known information"
    ]
    
    for suggestion in suggestions:
        feedback_parts.append(suggestion + "\n")
    
    feedback_parts.append("\n" + "=" * 60 + "\n")
    
    return "".join(feedback_parts)


def augment_prompt_with_feedback(original_prompt: str, feedback: str) -> str:
    """Augment the original prompt with similarity feedback.
    
    Args:
        original_prompt: The original generation prompt
        feedback: Similarity feedback to prepend
        
    Returns:
        Augmented prompt with feedback and instructions
    """
    if not feedback:
        return original_prompt
    
    # Build augmented prompt
    augmented_parts = [
        "⚠️ IMPORTANT: Your previous response was too similar to earlier content.\n\n",
        feedback,
        "\n📝 Please regenerate with the following requirements:\n",
        "• Avoid repeating ideas, phrases, or structures from previous content\n",
        "• Introduce fresh perspectives and unique insights\n",
        "• Use varied vocabulary and writing style\n",
        "• Focus on adding new value rather than restating existing content\n\n",
        "Original request:\n",
        "-" * 40 + "\n",
        original_prompt
    ]
    
    return "".join(augmented_parts)