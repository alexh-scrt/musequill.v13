import re
from typing import Dict, Optional, List, Any


def calculate_citation_metrics(
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate Citation Density & Novelty Context
    
    Target: 15-40 citations per 10 pages, clear novelty statement
    
    Returns dict with:
        - score: Normalized score for metric system (0-10)
        - citation_count: Total citations found
        - citations_per_page: Normalized to 10-page equivalent
        - has_related_work: Boolean for related work section
        - has_novelty_claims: Boolean for explicit novelty
        - details: Breakdown and assessment
    """
    if not content:
        return {
            'score': 0.0,
            'percentage': 0.0,
            'citation_count': 0,
            'citations_per_page': 0.0,
            'has_related_work': False,
            'has_novelty_claims': False,
            'details': {'assessment': 'No content'}
        }
    
    # Extract citations
    citations = extract_citations(content)
    citation_count = len(citations)
    
    # Estimate page count (rough: 500 words per page)
    word_count = len(content.split())
    estimated_pages = max(word_count / 500, 1)
    
    # Normalize to citations per 10 pages
    citations_per_10_pages = (citation_count / estimated_pages) * 10
    
    # Check for related work section
    has_related_work = detect_related_work_section(content)
    
    # Check for novelty claims
    has_novelty_claims = detect_novelty_claims(content)
    
    # Check if citations are directly relevant (near key claims)
    relevant_citations_ratio = check_citation_relevance(content, citations)
    
    # Calculate citation density score (0-5 points)
    # Target: 15-40 citations per 10 pages
    if 15 <= citations_per_10_pages <= 40:
        cite_density_score = 5.0
    elif 10 <= citations_per_10_pages < 15:
        cite_density_score = 3.0 + ((citations_per_10_pages - 10) / 5) * 2.0
    elif citations_per_10_pages > 40:
        # Too many citations can be distracting
        cite_density_score = max(0, 5.0 - (citations_per_10_pages - 40) * 0.1)
    else:
        cite_density_score = (citations_per_10_pages / 10) * 3.0
    
    # Calculate novelty clarity score (0-5 points)
    novelty_score = 0.0
    if has_related_work:
        novelty_score += 1.5
    if has_novelty_claims:
        novelty_score += 1.5
    novelty_score += relevant_citations_ratio * 2.0  # Up to 2 points
    
    # Total score (0-10 points)
    score = cite_density_score + novelty_score
    score = min(score, 10.0)
    
    return {
        'score': score,
        'percentage': (score / 10.0) * 100,
        'citation_count': citation_count,
        'citations_per_page': citations_per_10_pages,
        'has_related_work': has_related_work,
        'has_novelty_claims': has_novelty_claims,
        'relevant_citations_ratio': relevant_citations_ratio,
        'details': {
            'cite_density_score': cite_density_score,
            'novelty_score': novelty_score,
            'assessment': _assess_citations(
                citations_per_10_pages,
                has_related_work,
                has_novelty_claims
            ),
            'citation_examples': [c['marker'] for c in citations[:10]]
        }
    }


def extract_citations(text: str) -> List[Dict[str, Any]]:
    """Extract citation markers from text"""
    citations = []
    
    # Pattern 1: Numbered citations [1], [2,3], [1-5]
    numbered_pattern = r'\[(\d+(?:[-,]\s*\d+)*)\]'
    for match in re.finditer(numbered_pattern, text):
        # Expand ranges like [1-5]
        marker = match.group(1)
        if '-' in marker:
            start, end = map(int, marker.split('-'))
            count = end - start + 1
        elif ',' in marker:
            count = len(marker.split(','))
        else:
            count = 1
        
        for _ in range(count):
            citations.append({
                'marker': match.group(0),
                'position': match.start(),
                'type': 'numbered'
            })
    
    # Pattern 2: Author-year citations (Smith 2020), (Smith et al. 2020)
    author_year_pattern = r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s+and\s+[A-Z][a-z]+)?\s+\d{4}[a-z]?)\)'
    for match in re.finditer(author_year_pattern, text):
        citations.append({
            'marker': match.group(1),
            'position': match.start(),
            'type': 'author-year'
        })
    
    # Pattern 3: Multiple author-year (Smith 2020; Jones 2021)
    multi_cite_pattern = r'\(([A-Z][a-z]+\s+\d{4}(?:;\s*[A-Z][a-z]+\s+\d{4})+)\)'
    for match in re.finditer(multi_cite_pattern, text):
        # Count semicolon-separated citations
        count = match.group(1).count(';') + 1
        for _ in range(count):
            citations.append({
                'marker': match.group(0),
                'position': match.start(),
                'type': 'multi-cite'
            })
    
    # Remove duplicates based on position
    unique_citations = []
    seen_positions = set()
    for cite in citations:
        if cite['position'] not in seen_positions:
            unique_citations.append(cite)
            seen_positions.add(cite['position'])
    
    return unique_citations


def detect_related_work_section(text: str) -> bool:
    """Check if document has a related work section"""
    patterns = [
        r'\n#+\s*(?:Related Work|Previous Work|Background|Literature Review)',
        r'\n(?:Related Work|Previous Work|Background|Literature Review)\n[=-]+',
        r'(?:Section|§)\s+\d+\.?\s+(?:Related Work|Previous Work|Background)'
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def detect_novelty_claims(text: str) -> bool:
    """Check if document explicitly states what's novel"""
    patterns = [
        r'(?:our|this)\s+(?:novel|new|original)\s+(?:contribution|approach|method|framework)',
        r'(?:we|this work)\s+(?:introduce|present|propose)\s+(?:a novel|a new|an original)',
        r'(?:for the first time|to our knowledge|to the best of our knowledge)',
        r'(?:unlike|in contrast to|different from)\s+(?:previous|prior|existing)\s+(?:work|approaches|methods)',
        r'(?:our main contribution|key contribution|novel aspect)\s+is'
    ]
    
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def check_citation_relevance(text: str, citations: List[Dict]) -> float:
    """Check if citations appear near important claims"""
    
    # Identify important claims (those that likely need citations)
    important_claim_patterns = [
        r'(?:it is known|well-known|established|shown|demonstrated|proven)\s+that',
        r'(?:as|according to)\s+(?:shown|demonstrated|proven|established)',
        r'(?:previous|prior)\s+(?:work|research|studies)',
        r'(?:theorem|lemma|result)\s+(?:by|from|in)'
    ]
    
    important_positions = []
    for pattern in important_claim_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            important_positions.append(match.start())
    
    if not important_positions:
        return 1.0  # No claims requiring citations
    
    # Check how many important claims have nearby citations
    claims_with_citations = 0
    for claim_pos in important_positions:
        # Look for citation within 100 characters
        has_nearby_cite = any(
            abs(cite['position'] - claim_pos) < 100
            for cite in citations
        )
        if has_nearby_cite:
            claims_with_citations += 1
    
    return claims_with_citations / len(important_positions)


def _assess_citations(
    citations_per_page: float,
    has_related_work: bool,
    has_novelty: bool
) -> str:
    """Provide qualitative assessment of citation context"""
    
    cite_quality = ""
    if citations_per_page >= 15:
        cite_quality = "Good citation density"
    elif citations_per_page >= 10:
        cite_quality = "Acceptable citation density"
    elif citations_per_page >= 5:
        cite_quality = "Low citation density"
    else:
        cite_quality = "Critical: Insufficient citations"
    
    context_quality = ""
    if has_related_work and has_novelty:
        context_quality = ", clear novelty context"
    elif has_related_work or has_novelty:
        context_quality = ", partial novelty context"
    else:
        context_quality = ", missing novelty context"
    
    return cite_quality + context_quality