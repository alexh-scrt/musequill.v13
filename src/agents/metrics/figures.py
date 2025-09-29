import re
from typing import Dict, List, Any, Optional


def calculate_figure_utility(
    content: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate Figure/Equation Information Density
    
    Target: Each figure/equation should add information not in text
    
    Returns dict with:
        - score: Normalized score (0-5)
        - total_figures: Number of figures
        - total_equations: Number of equations
        - informative_ratio: Percentage that add information
    """
    figures = extract_figure_references(content)
    equations = extract_equations(content)
    
    total = len(figures) + len(equations)
    
    if total == 0:
        # No figures/equations - neutral score
        return {
            'score': 3.0,
            'percentage': 60.0,
            'total_figures': 0,
            'total_equations': 0,
            'informative_ratio': 100.0,
            'details': {'assessment': 'No figures or equations'}
        }
    
    # Heuristic: check if figures/equations are referenced meaningfully
    informative_count = 0
    for fig in figures:
        if is_informative_figure(content, fig):
            informative_count += 1
    
    for eq in equations:
        if is_informative_equation(content, eq):
            informative_count += 1
    
    informative_ratio = informative_count / total
    score = informative_ratio * 5
    
    return {
        'score': score,
        'percentage': (score / 5.0) * 100,
        'total_figures': len(figures),
        'total_equations': len(equations),
        'informative_ratio': informative_ratio * 100,
        'details': {
            'assessment': _assess_figures(informative_ratio)
        }
    }


def extract_figure_references(text: str) -> List[Dict]:
    """Extract figure references"""
    pattern = r'(?:Figure|Fig\.?)\s+\d+'
    figures = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        figures.append({'ref': match.group(0), 'position': match.start()})
    return figures


def extract_equations(text: str) -> List[Dict]:
    """Extract equation references"""
    patterns = [
        r'Equation\s+\d+',
        r'Eq\.?\s+\d+',
        r'\$\$[^$]+\$\$'  # Display equations
    ]
    
    equations = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            equations.append({'ref': match.group(0), 'position': match.start()})
    
    return equations


def is_informative_figure(text: str, figure: Dict) -> bool:
    """Check if figure adds information beyond text"""
    # Look for analysis keywords near figure reference
    pos = figure['position']
    context_start = max(0, pos - 200)
    context_end = min(len(text), pos + 200)
    context = text[context_start:context_end].lower()
    
    informative_keywords = ['shows', 'illustrates', 'demonstrates', 'depicts', 
                           'visualizes', 'compares', 'plots', 'distribution']
    
    return any(keyword in context for keyword in informative_keywords)


def is_informative_equation(text: str, equation: Dict) -> bool:
    """Check if equation adds information beyond prose"""
    pos = equation['position']
    context_start = max(0, pos - 200)
    context_end = min(len(text), pos + 200)
    context = text[context_start:context_end].lower()
    
    # Check if equation is just restating prose
    restatement_keywords = ['as stated', 'in other words', 'that is', 'namely']
    
    return not any(keyword in context for keyword in restatement_keywords)


def _assess_figures(ratio: float) -> str:
    """Assess figure/equation utility"""
    if ratio >= 0.9:
        return "Excellent: All visuals add information"
    elif ratio >= 0.8:
        return "Good: Most visuals informative"
    elif ratio >= 0.6:
        return "Acceptable: Some decorative elements"
    else:
        return "Poor: Too many redundant visuals"