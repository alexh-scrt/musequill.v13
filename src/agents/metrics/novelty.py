import re
from typing import Optional, Dict, Set, Any
from collections import Counter


def calculate_cnr(
    content: str,
    previous_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate Conceptual Novelty Rate
    
    Returns dict with:
        - percentage: CNR as percentage (0-100)
        - score: Normalized score for metric system
        - new_concepts: Set of new concepts
        - total_concepts: Set of all concepts
        - details: Additional analysis
    """
    # Extract concepts from current content
    current_concepts = extract_concepts(content)
    
    # Extract concepts from previous content
    if previous_content:
        previous_concepts = extract_concepts(previous_content)
    else:
        previous_concepts = set()
    
    # Calculate novelty
    new_concepts = current_concepts - previous_concepts
    total_concepts = current_concepts
    
    if not total_concepts:
        cnr_percentage = 0.0
    else:
        cnr_percentage = (len(new_concepts) / len(total_concepts)) * 100
    
    # Calculate score (normalized to 15 points max)
    if cnr_percentage >= 40:
        score = min(cnr_percentage / 40 * 15, 15)
    elif cnr_percentage >= 25:
        score = 6 + (cnr_percentage - 25) / 15 * 6
    else:
        score = cnr_percentage / 25 * 6
    
    return {
        'percentage': cnr_percentage,
        'score': score,
        'new_concepts': list(new_concepts)[:20],  # Sample
        'total_concepts': len(total_concepts),
        'new_count': len(new_concepts),
        'details': {
            'has_previous': previous_content is not None,
            'assessment': _assess_novelty(cnr_percentage)
        }
    }


def extract_concepts(text: str) -> Set[str]:
    """
    Extract technical concepts from text
    
    Concepts include:
    - Mathematical terms
    - Technical terminology
    - Defined terms
    - Named theorems/lemmas
    - Novel compounds (e.g., "quantum-thermodynamic")
    """
    concepts = set()
    
    # Extract capitalized technical terms
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    concepts.update(capitalized)
    
    # Extract mathematical symbols/terms
    math_terms = re.findall(r'\$[^$]+\$|\\[a-zA-Z]+', text)
    concepts.update(math_terms)
    
    # Extract hyphenated compounds
    compounds = re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', text.lower())
    concepts.update(compounds)
    
    # Extract theorem/lemma references
    theorems = re.findall(
        r'(?:Theorem|Lemma|Proposition|Corollary)\s+\d+',
        text,
        re.IGNORECASE
    )
    concepts.update(theorems)
    
    # Extract defined terms (simple heuristic)
    definitions = re.findall(
        r'(?:define|called|termed|known as)\s+["\']?([^"\',.]+)["\']?',
        text,
        re.IGNORECASE
    )
    concepts.update(definitions)
    
    # Filter out common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    concepts = {c for c in concepts if c.lower() not in stopwords}
    
    return concepts


def _assess_novelty(percentage: float) -> str:
    """Provide qualitative assessment of novelty"""
    if percentage >= 60:
        return "Excellent: Highly novel content"
    elif percentage >= 40:
        return "Good: Sufficient novelty"
    elif percentage >= 25:
        return "Acceptable: Minimum novelty threshold"
    elif percentage >= 15:
        return "Poor: Excessive redundancy"
    else:
        return "Critical: Unacceptable redundancy"