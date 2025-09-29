import re
from typing import Dict, List, Any


def calculate_parsimony(content: str) -> Dict[str, Any]:
    """
    Calculate Parsimony Score - Ockham's razor
    
    Target: 70%+ of assumptions are essential
    
    Returns dict with:
        - score: Normalized score (0-5)
        - total_assumptions: Number of assumptions
        - essential_assumptions: Number actually used
        - parsimony_ratio: Percentage essential
    """
    assumptions = extract_assumptions(content)
    usage = check_assumption_usage(content, assumptions)
    
    total = len(assumptions)
    if total == 0:
        # No explicit assumptions - neutral score
        return {
            'score': 3.0,
            'percentage': 60.0,
            'total_assumptions': 0,
            'essential_assumptions': 0,
            'parsimony_ratio': 100.0,
            'details': {'assessment': 'No explicit assumptions'}
        }
    
    essential = sum(1 for used in usage.values() if used)
    parsimony_ratio = essential / total
    
    score = parsimony_ratio * 5
    
    return {
        'score': score,
        'percentage': (score / 5.0) * 100,
        'total_assumptions': total,
        'essential_assumptions': essential,
        'parsimony_ratio': parsimony_ratio * 100,
        'details': {
            'unused_assumptions': [a for a, used in usage.items() if not used][:5],
            'assessment': _assess_parsimony(parsimony_ratio)
        }
    }


def extract_assumptions(text: str) -> List[str]:
    """Extract stated assumptions"""
    patterns = [
        r'(?:assume|assumption)\s+(?:that\s+)?([^.]+\.)',
        r'(?:suppose|let us assume)\s+([^.]+\.)',
        r'(?:given|granted)\s+(?:that\s+)?([^.]+\.)'
    ]
    
    assumptions = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            assumptions.append(match.group(1)[:100])
    
    return assumptions


def check_assumption_usage(text: str, assumptions: List[str]) -> Dict[str, bool]:
    """Check which assumptions are actually used"""
    usage = {}
    
    for assumption in assumptions:
        # Extract key terms from assumption
        words = set(assumption.lower().split())
        words = {w for w in words if len(w) > 4}  # Only significant words
        
        # Check if these terms appear elsewhere in text
        # (crude heuristic for "used in derivation")
        rest_of_text = text.lower()
        used = sum(1 for word in words if rest_of_text.count(word) > 1) > len(words) * 0.3
        
        usage[assumption] = used
    
    return usage


def _assess_parsimony(ratio: float) -> str:
    """Assess parsimony"""
    if ratio >= 0.9:
        return "Excellent: All assumptions essential"
    elif ratio >= 0.7:
        return "Good: Most assumptions necessary"
    elif ratio >= 0.5:
        return "Acceptable: Some unnecessary complexity"
    else:
        return "Poor: Overcomplicated with unused assumptions"