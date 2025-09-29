import re
from typing import Dict, List, Any


def calculate_claim_density(content: str) -> Dict[str, Any]:
    """
    Calculate Claim Density - ratio of falsifiable claims to total words
    
    Target: 1 substantive claim per 100-200 words (0.5-1.0 claims/100w)
    
    Returns dict with:
        - density: Claims per 100 words
        - score: Normalized score for metric system (0-10)
        - total_claims: Number of claims identified
        - word_count: Total word count
        - details: Breakdown by claim type
    """
    word_count = len(content.split())
    
    if word_count == 0:
        return {
            'density': 0.0,
            'score': 0.0,
            'total_claims': 0,
            'word_count': 0,
            'details': {'assessment': 'No content'}
        }
    
    # Extract different types of claims
    claims = extract_claims(content)
    total_claims = len(claims)
    
    # Calculate density (claims per 100 words)
    density = (total_claims / word_count) * 100
    
    # Calculate score (normalized to 10 points max)
    # Target: >= 0.8 claims/100w = 10 points
    #         >= 0.6 claims/100w = 7 points
    #         >= 0.4 claims/100w = 5 points (minimum)
    if density >= 0.8:
        score = 10.0
    elif density >= 0.6:
        score = 7.0 + ((density - 0.6) / 0.2) * 3.0
    elif density >= 0.4:
        score = 5.0 + ((density - 0.4) / 0.2) * 2.0
    else:
        score = (density / 0.4) * 5.0
    
    score = min(score, 10.0)
    
    # Breakdown by claim type
    claim_types = {
        'theorems': len([c for c in claims if c['type'] == 'theorem']),
        'predictions': len([c for c in claims if c['type'] == 'prediction']),
        'assertions': len([c for c in claims if c['type'] == 'assertion']),
        'comparisons': len([c for c in claims if c['type'] == 'comparison']),
        'implications': len([c for c in claims if c['type'] == 'implication'])
    }
    
    return {
        'density': density,
        'score': score,
        'total_claims': total_claims,
        'word_count': word_count,
        'percentage': (score / 10.0) * 100,
        'details': {
            'claim_types': claim_types,
            'assessment': _assess_density(density)
        }
    }


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """
    Extract substantive claims from text
    
    Claims include:
    - Theorems, lemmas, propositions, corollaries
    - Testable predictions
    - Quantitative assertions
    - Comparative statements
    - Causal implications
    """
    claims = []
    
    # 1. Formal mathematical statements
    theorem_patterns = [
        r'(?:Theorem|Lemma|Proposition|Corollary)\s+\d+[.:]\s*([^.]+\.)',
        r'(?:Theorem|Lemma|Proposition|Corollary)[.:]\s*([^.]+\.)'
    ]
    
    for pattern in theorem_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                'type': 'theorem',
                'text': match.group(0),
                'position': match.start()
            })
    
    # 2. Predictions and hypotheses
    prediction_patterns = [
        r'(?:we predict|predicted|prediction|hypothesis|hypothesize|expect|should observe)\s+(?:that\s+)?([^.]+\.)',
        r'(?:will|would|could)\s+(?:result in|lead to|cause|produce)\s+([^.]+\.)'
    ]
    
    for pattern in prediction_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                'type': 'prediction',
                'text': match.group(0),
                'position': match.start()
            })
    
    # 3. Quantitative assertions (with numbers)
    quant_patterns = [
        r'(?:is|are|equals?|measures?)\s+(?:approximately\s+)?[\d.]+\s*[%\w]+',
        r'(?:increases?|decreases?|grows?|declines?)\s+by\s+[\d.]+',
        r'(?:greater|less|more|fewer)\s+than\s+[\d.]+'
    ]
    
    for pattern in quant_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Get full sentence
            sent_start = text.rfind('.', 0, match.start()) + 1
            sent_end = text.find('.', match.end())
            if sent_end == -1:
                sent_end = len(text)
            
            claims.append({
                'type': 'assertion',
                'text': text[sent_start:sent_end+1].strip(),
                'position': sent_start
            })
    
    # 4. Comparative statements
    comp_patterns = [
        r'(?:better|worse|superior|inferior|outperforms?|exceeds?)\s+(?:than\s+)?[^.]+\.',
        r'(?:compared to|in contrast to|unlike)\s+[^,]+,\s*[^.]+\.'
    ]
    
    for pattern in comp_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                'type': 'comparison',
                'text': match.group(0),
                'position': match.start()
            })
    
    # 5. Causal implications
    impl_patterns = [
        r'(?:therefore|thus|hence|consequently|as a result)\s+[^.]+\.',
        r'(?:implies?|implication|entails?|leads? to|results? in)\s+[^.]+\.',
        r'if\s+[^,]+,\s+then\s+[^.]+\.'
    ]
    
    for pattern in impl_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            claims.append({
                'type': 'implication',
                'text': match.group(0),
                'position': match.start()
            })
    
    # Remove duplicates based on position
    seen_positions = set()
    unique_claims = []
    for claim in sorted(claims, key=lambda x: x['position']):
        # Check if position overlaps with existing claim
        if not any(abs(claim['position'] - seen) < 50 for seen in seen_positions):
            unique_claims.append(claim)
            seen_positions.add(claim['position'])
    
    return unique_claims


def _assess_density(density: float) -> str:
    """Provide qualitative assessment of claim density"""
    if density >= 1.0:
        return "Excellent: High density of substantive claims"
    elif density >= 0.8:
        return "Good: Sufficient claim density"
    elif density >= 0.6:
        return "Acceptable: Moderate claim density"
    elif density >= 0.4:
        return "Poor: Below minimum threshold"
    else:
        return "Critical: Too much filler, insufficient claims"