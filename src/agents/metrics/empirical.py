import re
from typing import Dict, List, Any


def calculate_empirical_grounding(content: str) -> Dict[str, Any]:
    """
    Calculate Empirical Grounding Score - connection to testable predictions
    
    Target: 2-3 testable predictions or experimental connections
    
    Returns dict with:
        - score: Normalized score (0-10)
        - predictions: List of testable predictions
        - experimental_refs: References to experiments
        - falsifiable_ratio: Ratio of falsifiable to total claims
    """
    predictions = extract_predictions(content)
    experimental_refs = extract_experimental_references(content)
    tautologies = detect_tautologies(content)
    
    total_statements = len(predictions) + len(experimental_refs) + len(tautologies)
    falsifiable_count = len(predictions) + len(experimental_refs)
    
    if total_statements > 0:
        falsifiable_ratio = falsifiable_count / total_statements
    else:
        falsifiable_ratio = 1.0
    
    # Score calculation
    prediction_score = min(len(predictions) / 3, 1.0) * 5
    experimental_score = min(len(experimental_refs) / 2, 1.0) * 3
    falsifiable_score = falsifiable_ratio * 2
    
    score = prediction_score + experimental_score + falsifiable_score
    
    return {
        'score': score,
        'percentage': (score / 10.0) * 100,
        'predictions': len(predictions),
        'experimental_refs': len(experimental_refs),
        'tautologies': len(tautologies),
        'falsifiable_ratio': falsifiable_ratio * 100,
        'details': {
            'prediction_list': predictions[:5],
            'assessment': _assess_empirical(len(predictions), len(experimental_refs))
        }
    }


def extract_predictions(text: str) -> List[str]:
    """Extract testable predictions"""
    patterns = [
        r'(?:we predict|prediction|hypothesis|expect)\s+(?:that\s+)?([^.]+\.)',
        r'(?:will|would|should)\s+(?:observe|measure|detect|find)\s+([^.]+\.)',
        r'(?:if|when)\s+[^,]+,\s+(?:then|we expect)\s+([^.]+\.)'
    ]
    
    predictions = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            predictions.append(match.group(1)[:100])
    
    return predictions


def extract_experimental_references(text: str) -> List[str]:
    """Extract references to experimental work"""
    patterns = [
        r'(?:experiment|experimental|empirical)\s+(?:results?|data|evidence|validation)',
        r'(?:measured|observed|detected)\s+in\s+(?:experiments?|studies)',
        r'(?:experimental|empirical)\s+(?:support|confirmation|validation)'
    ]
    
    refs = []
    for pattern in patterns:
        refs.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return refs


def detect_tautologies(text: str) -> List[str]:
    """Detect circular or tautological reasoning"""
    patterns = [
        r'by definition',
        r'is true because it is',
        r'follows from itself'
    ]
    
    tautologies = []
    for pattern in patterns:
        tautologies.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return tautologies


def _assess_empirical(predictions: int, refs: int) -> str:
    """Assess empirical grounding"""
    total = predictions + refs
    if total >= 5:
        return "Excellent: Strong empirical grounding"
    elif total >= 3:
        return "Good: Sufficient empirical connection"
    elif total >= 1:
        return "Acceptable: Minimal empirical grounding"
    else:
        return "Poor: Lacks empirical connection"