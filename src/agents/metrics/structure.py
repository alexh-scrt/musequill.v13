import re
from typing import Dict, Any


def calculate_structural_coherence(content: str) -> Dict[str, Any]:
    """
    Calculate Structural Coherence Index
    
    Target: Linear flow, <10% backward references, no circular reasoning
    
    Returns dict with:
        - score: Normalized score (0-10)
        - forward_refs: Count of forward references
        - backward_refs: Count of backward references
        - circular_patterns: Detected circular reasoning
    """
    forward_refs = count_forward_references(content)
    backward_refs = count_backward_references(content)
    circular = detect_circular_reasoning(content)
    
    total_refs = forward_refs + backward_refs
    if total_refs > 0:
        backward_ratio = backward_refs / total_refs
    else:
        backward_ratio = 0.0
    
    # Score calculation
    # Penalize forward refs, backward refs, and circular reasoning
    score = 10.0
    score -= min(circular * 2, 5)  # -2 per circular pattern, max -5
    score -= min(forward_refs * 0.5, 3)  # -0.5 per forward ref, max -3
    score -= min(backward_ratio * 5, 2)  # Penalize high backward ratio
    
    score = max(score, 0)
    
    return {
        'score': score,
        'percentage': (score / 10.0) * 100,
        'forward_refs': forward_refs,
        'backward_refs': backward_refs,
        'circular_patterns': circular,
        'backward_ratio': backward_ratio * 100,
        'details': {
            'assessment': _assess_structure(circular, forward_refs, backward_ratio)
        }
    }


def count_forward_references(text: str) -> int:
    """Count references to later sections"""
    patterns = [
        r'(?:will be|to be)\s+(?:discussed|shown|proven|described)\s+(?:later|below|in\s+Section)',
        r'(?:see|refer to)\s+(?:Section|Chapter)\s+\d+',
        r'as\s+(?:we will see|will be shown)'
    ]
    
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return count


def count_backward_references(text: str) -> int:
    """Count references to earlier sections"""
    patterns = [
        r'as\s+(?:discussed|shown|proven|mentioned)\s+(?:earlier|above|previously)',
        r'(?:recall|remember)\s+(?:that|from)',
        r'as\s+(?:stated|established|proven)\s+in\s+(?:Section|Chapter)\s+\d+'
    ]
    
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    
    return count


def detect_circular_reasoning(text: str) -> int:
    """Detect circular reasoning patterns"""
    # Simple heuristic: look for A implies B, B implies A patterns
    # This is a simplified check
    patterns = [
        r'(?:because|since)\s+[^.]+\.\s+[^.]+(?:therefore|thus|hence)\s+[^.]+',
    ]
    
    # Count potential circular patterns
    count = 0
    sentences = text.split('.')
    
    for i in range(len(sentences) - 1):
        sent1 = sentences[i].lower()
        sent2 = sentences[i + 1].lower()
        
        # Check if key terms repeat in reverse logical order
        if ('because' in sent1 and 'therefore' in sent2) or \
           ('therefore' in sent1 and 'because' in sent2):
            count += 1
    
    return count


def _assess_structure(circular: int, forward: int, backward_ratio: float) -> str:
    """Assess structural coherence"""
    if circular == 0 and forward < 5 and backward_ratio < 0.1:
        return "Excellent: Clear linear structure"
    elif circular <= 2 and forward < 10 and backward_ratio < 0.15:
        return "Good: Acceptable structure"
    elif circular <= 3 and backward_ratio < 0.2:
        return "Acceptable: Some structural issues"
    else:
        return "Poor: Significant structural problems"