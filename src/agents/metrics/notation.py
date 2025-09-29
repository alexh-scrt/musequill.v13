import re
from typing import Dict, List, Any


def calculate_notation_consistency(content: str) -> Dict[str, Any]:
    """
    Calculate Notation Consistency Rate
    
    Target: 100% - same symbol = same meaning throughout
    
    Returns dict with:
        - score: Normalized score (0-5)
        - consistency_ratio: Percentage of consistent symbols
        - total_symbols: Number of unique symbols
        - redefinitions: List of redefined symbols
    """
    symbols = extract_mathematical_symbols(content)
    redefinitions = detect_redefinitions(content, symbols)
    
    total_symbols = len(symbols)
    if total_symbols > 0:
        consistent_symbols = total_symbols - len(redefinitions)
        consistency_ratio = consistent_symbols / total_symbols
    else:
        consistency_ratio = 1.0
    
    score = consistency_ratio * 5
    
    return {
        'score': score,
        'percentage': (score / 5.0) * 100,
        'consistency_ratio': consistency_ratio * 100,
        'total_symbols': total_symbols,
        'redefinitions': len(redefinitions),
        'details': {
            'redefined_symbols': list(redefinitions)[:10],
            'assessment': _assess_notation(consistency_ratio)
        }
    }


def extract_mathematical_symbols(text: str) -> Dict[str, List[int]]:
    """Extract mathematical symbols and their positions"""
    # Look for LaTeX-style math or Greek letters
    patterns = [
        r'\\[a-zA-Z]+',  # LaTeX commands
        r'\$[^$]+\$',  # Inline math
        r'[α-ωΑ-Ω]',  # Greek letters
    ]
    
    symbols = {}
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            symbol = match.group(0)
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(match.start())
    
    return symbols


def detect_redefinitions(text: str, symbols: Dict[str, List[int]]) -> List[str]:
    """Detect symbols that are redefined"""
    redefined = []
    
    for symbol, positions in symbols.items():
        if len(positions) < 2:
            continue
        
        # Check if symbol appears in multiple definition contexts
        definition_count = 0
        for pos in positions:
            # Look for definition keywords near symbol
            context_start = max(0, pos - 100)
            context_end = min(len(text), pos + 100)
            context = text[context_start:context_end].lower()
            
            if any(word in context for word in ['let', 'define', 'denote', 'represent']):
                definition_count += 1
        
        if definition_count > 1:
            redefined.append(symbol)
    
    return redefined


def _assess_notation(ratio: float) -> str:
    """Assess notation consistency"""
    if ratio >= 0.99:
        return "Excellent: Perfect consistency"
    elif ratio >= 0.95:
        return "Good: Minor inconsistencies"
    elif ratio >= 0.90:
        return "Acceptable: Some inconsistencies"
    else:
        return "Poor: Significant notation issues"