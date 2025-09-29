import re
from typing import Dict, Optional, List, Set, Any
from collections import Counter


def calculate_compression_ratio(
    content: str,
    previous_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate Semantic Compression Ratio - how much content can be compressed
    
    Target: < 2:1 compression (content shouldn't reduce to half size)
    
    Returns dict with:
        - ratio: Compression ratio (original_size : compressed_size)
        - score: Normalized score for metric system (0-15)
        - original_length: Character count of original
        - compressed_length: Estimated minimal length
        - redundancy_percentage: Percentage of redundant content
        - details: Analysis of redundancy sources
    """
    if not content:
        return {
            'ratio': 1.0,
            'score': 15.0,
            'percentage': 100.0,
            'original_length': 0,
            'compressed_length': 0,
            'redundancy_percentage': 0.0,
            'details': {'assessment': 'No content'}
        }
    
    original_length = len(content)
    
    # Analyze redundancy from multiple sources
    redundancy_analysis = analyze_redundancy(content, previous_content)
    
    # Estimate compressed length by removing redundancy
    compressed_length = estimate_compressed_length(
        content,
        redundancy_analysis
    )
    
    # Calculate compression ratio
    if compressed_length > 0:
        ratio = original_length / compressed_length
    else:
        ratio = 1.0
    
    # Calculate redundancy percentage
    redundancy_pct = ((original_length - compressed_length) / original_length) * 100
    
    # Calculate score (0-15 points)
    # Lower ratio = better (less redundancy)
    # SCR <= 2.0 = 15 points
    # SCR <= 3.0 = 12 points
    # SCR <= 4.0 = 9 points
    # SCR <= 5.0 = 7 points (minimum)
    if ratio <= 2.0:
        score = 15.0
    elif ratio <= 3.0:
        score = 12.0 + ((3.0 - ratio) / 1.0) * 3.0
    elif ratio <= 4.0:
        score = 9.0 + ((4.0 - ratio) / 1.0) * 3.0
    elif ratio <= 5.0:
        score = 7.0 + ((5.0 - ratio) / 1.0) * 2.0
    else:
        score = max(0, 15.0 - ratio * 2.0)
    
    score = max(0.0, min(score, 15.0))
    
    return {
        'ratio': ratio,
        'score': score,
        'percentage': (score / 15.0) * 100,
        'original_length': original_length,
        'compressed_length': compressed_length,
        'redundancy_percentage': redundancy_pct,
        'details': {
            'redundancy_sources': redundancy_analysis,
            'assessment': _assess_compression(ratio)
        }
    }


def analyze_redundancy(
    content: str,
    previous_content: Optional[str]
) -> Dict[str, Any]:
    """Analyze sources of redundancy in content"""
    
    # 1. Repetitive phrases (n-grams that appear multiple times)
    repeated_phrases = find_repeated_phrases(content)
    
    # 2. Semantic duplicates (similar sentences)
    semantic_dupes = find_semantic_duplicates(content)
    
    # 3. Redundant explanations (concept explained multiple times)
    redundant_explanations = find_redundant_explanations(content)
    
    # 4. Overlap with previous content
    if previous_content:
        overlap = calculate_overlap(content, previous_content)
    else:
        overlap = 0.0
    
    return {
        'repeated_phrases': repeated_phrases,
        'semantic_duplicates': semantic_dupes,
        'redundant_explanations': redundant_explanations,
        'previous_overlap': overlap
    }


def find_repeated_phrases(text: str, min_words: int = 5) -> Dict[str, int]:
    """Find phrases that repeat multiple times"""
    words = text.lower().split()
    
    # Generate n-grams
    phrase_counts = Counter()
    for n in range(min_words, min(15, len(words) // 10 + 1)):
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            phrase_counts[phrase] += 1
    
    # Keep only phrases that appear 2+ times
    repeated = {phrase: count for phrase, count in phrase_counts.items() 
                if count >= 2}
    
    # Calculate total redundant characters from repetition
    total_redundant_chars = sum(
        len(phrase) * (count - 1) for phrase, count in repeated.items()
    )
    
    return {
        'count': len(repeated),
        'top_phrases': dict(sorted(repeated.items(), key=lambda x: -x[1])[:10]),
        'redundant_chars': total_redundant_chars
    }


def find_semantic_duplicates(text: str) -> Dict[str, Any]:
    """Find sentences that convey similar meaning"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Simple similarity: shared content words
    def get_content_words(sent: str) -> Set[str]:
        words = sent.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        return set(w for w in words if w not in stopwords and len(w) > 3)
    
    duplicates = []
    for i, sent1 in enumerate(sentences):
        words1 = get_content_words(sent1)
        if not words1:
            continue
            
        for j, sent2 in enumerate(sentences[i+1:], i+1):
            words2 = get_content_words(sent2)
            if not words2:
                continue
            
            # Jaccard similarity
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            similarity = overlap / union if union > 0 else 0
            
            if similarity > 0.6:  # 60% word overlap
                duplicates.append((i, j, similarity))
    
    # Estimate redundant characters from duplicates
    redundant_chars = sum(len(sentences[j]) for _, j, _ in duplicates)
    
    return {
        'count': len(duplicates),
        'redundant_chars': redundant_chars,
        'examples': duplicates[:5]
    }


def find_redundant_explanations(text: str) -> Dict[str, Any]:
    """Find concepts that are explained multiple times"""
    
    # Look for definition patterns
    definition_patterns = [
        r'(?:is defined as|can be defined as|defined as)\s+([^.]+\.)',
        r'(?:refers to|means|indicates)\s+([^.]+\.)',
        r'(?:In other words|That is|i\.e\.|namely)\s+([^.]+\.)'
    ]
    
    definitions = []
    for pattern in definition_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            definitions.append({
                'text': match.group(0),
                'position': match.start()
            })
    
    # Look for re-explanations (same concept words appearing in definitions)
    concept_words = {}
    for defn in definitions:
        words = set(defn['text'].lower().split())
        for other_defn in definitions:
            if defn is other_defn:
                continue
            other_words = set(other_defn['text'].lower().split())
            overlap = len(words & other_words)
            if overlap > 3:
                key = tuple(sorted(words & other_words)[:5])
                concept_words[key] = concept_words.get(key, 0) + 1
    
    redundant_explanations = {k: v for k, v in concept_words.items() if v > 1}
    
    return {
        'count': len(redundant_explanations),
        'total_definitions': len(definitions),
        'redundant_chars': sum(len(' '.join(k)) * v for k, v in redundant_explanations.items())
    }


def calculate_overlap(content: str, previous_content: str) -> float:
    """Calculate character overlap between current and previous content"""
    # Split into words
    current_words = set(content.lower().split())
    previous_words = set(previous_content.lower().split())
    
    # Calculate Jaccard similarity
    if not current_words:
        return 0.0
    
    overlap_words = current_words & previous_words
    overlap_ratio = len(overlap_words) / len(current_words)
    
    return overlap_ratio * 100


def estimate_compressed_length(
    content: str,
    redundancy_analysis: Dict[str, Any]
) -> int:
    """Estimate minimal content length by removing redundancy"""
    original_length = len(content)
    
    # Sum up redundant characters from all sources
    redundant_chars = 0
    
    if 'repeated_phrases' in redundancy_analysis:
        redundant_chars += redundancy_analysis['repeated_phrases'].get('redundant_chars', 0)
    
    if 'semantic_duplicates' in redundancy_analysis:
        redundant_chars += redundancy_analysis['semantic_duplicates'].get('redundant_chars', 0)
    
    if 'redundant_explanations' in redundancy_analysis:
        redundant_chars += redundancy_analysis['redundant_explanations'].get('redundant_chars', 0)
    
    # If there's overlap with previous content, count it as redundant
    if redundancy_analysis.get('previous_overlap', 0) > 0:
        overlap_pct = redundancy_analysis['previous_overlap'] / 100
        redundant_chars += int(original_length * overlap_pct * 0.5)  # 50% penalty
    
    # Estimate compressed length
    compressed = max(original_length - redundant_chars, original_length // 2)
    
    return compressed


def _assess_compression(ratio: float) -> str:
    """Provide qualitative assessment of compression ratio"""
    if ratio <= 2.0:
        return "Excellent: Tight writing, minimal redundancy"
    elif ratio <= 3.0:
        return "Good: Acceptable level of repetition"
    elif ratio <= 4.0:
        return "Acceptable: Some redundancy present"
    elif ratio <= 5.0:
        return "Poor: Excessive redundancy, needs compression"
    else:
        return "Critical: Unacceptable redundancy, major revision needed"