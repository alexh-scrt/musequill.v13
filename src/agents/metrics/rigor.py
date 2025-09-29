import re
from typing import Dict, List, Any, Set


def calculate_mathematical_rigor(content: str) -> Dict[str, Any]:
    """
    Calculate Mathematical Rigor Index - ratio of proved statements to theorems
    
    Target: 100% for theorems, 80%+ for lemmas, explicit citations for known results
    
    Returns dict with:
        - score: Normalized score for metric system (0-15)
        - proved_ratio: Ratio of theorems with proofs
        - cited_ratio: Ratio of claims with citations
        - total_theorems: Number of theorem-like statements
        - proved_theorems: Number with proofs
        - uncited_claims: Number of claims without citations
        - details: Breakdown and unproved items
    """
    # Extract theorems and check for proofs
    theorems = extract_theorems(content)
    proofs = extract_proofs(content)
    
    # Match theorems to proofs
    proved_theorems = match_theorems_to_proofs(theorems, proofs, content)
    
    # Calculate proof ratio
    if theorems:
        proved_ratio = len(proved_theorems) / len(theorems)
    else:
        proved_ratio = 1.0  # No theorems = perfect rigor (vacuous truth)
    
    # Extract claims that should be cited
    citeable_claims = extract_citeable_claims(content)
    citations = extract_citations(content)
    
    # Check which claims are properly cited
    cited_claims = match_claims_to_citations(citeable_claims, citations, content)
    
    # Calculate citation ratio
    if citeable_claims:
        cited_ratio = len(cited_claims) / len(citeable_claims)
    else:
        cited_ratio = 1.0
    
    # Calculate score (0-15 points)
    # 60% weight on proofs, 40% weight on citations
    score = (proved_ratio * 0.6 + cited_ratio * 0.4) * 15
    score = min(score, 15.0)
    
    # Identify unproved theorems
    unproved = [t['text'][:50] + '...' for t in theorems if t['id'] not in proved_theorems]
    
    # Count uncited claims
    uncited_count = len(citeable_claims) - len(cited_claims)
    
    return {
        'score': score,
        'percentage': (score / 15.0) * 100,
        'proved_ratio': proved_ratio * 100,
        'cited_ratio': cited_ratio * 100,
        'total_theorems': len(theorems),
        'proved_theorems': len(proved_theorems),
        'total_citeable': len(citeable_claims),
        'cited_claims': len(cited_claims),
        'uncited_claims': uncited_count,
        'unproved_theorems': unproved,
        'details': {
            'assessment': _assess_rigor(proved_ratio, cited_ratio),
            'theorem_list': [t['text'][:80] + '...' for t in theorems[:5]],
            'needs_proofs': unproved[:5]
        }
    }


def extract_theorems(text: str) -> List[Dict[str, Any]]:
    """Extract theorem-like statements"""
    theorems = []
    
    patterns = [
        r'(Theorem\s+\d+)[.:]\s*([^.]+(?:\.[^.]+){0,2}\.)',
        r'(Lemma\s+\d+)[.:]\s*([^.]+(?:\.[^.]+){0,2}\.)',
        r'(Proposition\s+\d+)[.:]\s*([^.]+(?:\.[^.]+){0,2}\.)',
        r'(Corollary\s+\d+)[.:]\s*([^.]+(?:\.[^.]+){0,2}\.)'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            theorems.append({
                'id': match.group(1),
                'text': match.group(2),
                'position': match.start(),
                'type': match.group(1).split()[0].lower()
            })
    
    return theorems


def extract_proofs(text: str) -> List[Dict[str, Any]]:
    """Extract proof sections"""
    proofs = []
    
    # Pattern 1: Explicit "Proof:" markers
    proof_pattern = r'Proof[.:]\s*([^□■\n]+(?:[^□■]){0,500})[□■]?'
    for match in re.finditer(proof_pattern, text, re.IGNORECASE | re.DOTALL):
        proofs.append({
            'text': match.group(1),
            'position': match.start(),
            'marker': 'explicit'
        })
    
    # Pattern 2: Proof by construction/contradiction indicators
    implicit_patterns = [
        r'(?:We prove this by|To prove|The proof proceeds|Proof by)\s+([^.]+(?:\.[^.]+){1,10}\.)',
        r'(?:Suppose|Assume|Let)\s+[^.]+\.(?:[^.]+\.){1,10}(?:Therefore|Thus|Hence)'
    ]
    
    for pattern in implicit_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            # Check if not already captured
            if not any(abs(match.start() - p['position']) < 100 for p in proofs):
                proofs.append({
                    'text': match.group(0),
                    'position': match.start(),
                    'marker': 'implicit'
                })
    
    return proofs


def match_theorems_to_proofs(
    theorems: List[Dict],
    proofs: List[Dict],
    text: str
) -> Set[str]:
    """Match theorems to their proofs based on proximity"""
    proved_theorem_ids = set()
    
    for theorem in theorems:
        # Look for proof within 1000 characters after theorem
        theorem_pos = theorem['position']
        
        for proof in proofs:
            proof_pos = proof['position']
            
            # Proof should come after theorem, within reasonable distance
            if 0 < proof_pos - theorem_pos < 1000:
                proved_theorem_ids.add(theorem['id'])
                break
    
    return proved_theorem_ids


def extract_citeable_claims(text: str) -> List[Dict[str, Any]]:
    """Extract claims that should have citations (established results)"""
    claims = []
    
    # Patterns indicating established knowledge
    patterns = [
        r'(?:as shown by|as demonstrated by|according to|as established by|as proven by)\s+([^.]+\.)',
        r'(?:it is known|well-known|established|proven)\s+that\s+([^.]+\.)',
        r'(?:previous work|prior research|earlier studies)\s+(?:has\s+)?(?:shown|demonstrated|proven)\s+([^.]+\.)'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                'text': match.group(1),
                'position': match.start(),
                'type': 'established'
            })
    
    return claims


def extract_citations(text: str) -> List[Dict[str, Any]]:
    """Extract citation markers"""
    citations = []
    
    # Common citation formats
    patterns = [
        r'\[(\d+(?:,\s*\d+)*)\]',  # [1], [1,2,3]
        r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\d{4})\)',  # (Smith 2020)
        r'\(([A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+\d{4})\)',  # (Smith and Jones 2020)
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            citations.append({
                'marker': match.group(1),
                'position': match.start()
            })
    
    return citations


def match_claims_to_citations(
    claims: List[Dict],
    citations: List[Dict],
    text: str
) -> Set[int]:
    """Match citeable claims to nearby citations"""
    cited_claim_indices = set()
    
    for i, claim in enumerate(claims):
        claim_pos = claim['position']
        
        # Look for citation within 200 characters before or after claim
        for citation in citations:
            cite_pos = citation['position']
            
            if abs(cite_pos - claim_pos) < 200:
                cited_claim_indices.add(i)
                break
    
    return cited_claim_indices


def _assess_rigor(proved_ratio: float, cited_ratio: float) -> str:
    """Provide qualitative assessment of mathematical rigor"""
    combined = (proved_ratio * 0.6 + cited_ratio * 0.4) * 100
    
    if combined >= 90:
        return "Excellent: Strong mathematical rigor"
    elif combined >= 80:
        return "Good: Acceptable rigor level"
    elif combined >= 60:
        return "Poor: Insufficient proofs or citations"
    else:
        return "Critical: Unacceptable lack of rigor"