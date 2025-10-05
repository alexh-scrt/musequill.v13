"""
Enhanced similarity detection system with three-tier analysis.

This module provides sophisticated content similarity detection to prevent
repetition in generated content. It implements a three-tier system:
- Tier 1: Identical detection (≥0.90 similarity)
- Tier 2: Semantic analysis (0.75-0.90 similarity)
- Tier 3: Information gain check (0.60-0.75 similarity)
"""

import os
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from src.storage.similarity_corpus import SimilarityCorpus
from src.common import ParagraphMatch

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    """Actions that can be taken on content"""
    SKIP = "SKIP"
    FLAG = "FLAG"
    ACCEPT = "ACCEPT"


@dataclass
class SimilarityDecision:
    """Result of similarity analysis"""
    action: DecisionAction
    reason: str
    similarity_score: float
    tier: str
    details: Dict[str, Any]
    similar_to: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None


@dataclass
class ConceptExtraction:
    """Extracted concepts from text"""
    concepts: Set[str]
    equations: List[str]
    examples: List[str]
    references: List[str]
    technical_terms: Set[str]


class RepetitionDetector:
    """
    Advanced repetition detection with three-tier analysis system.
    
    This detector analyzes content at multiple levels:
    1. Pure text similarity for near-identical content
    2. Semantic analysis for concept overlap
    3. Information gain assessment for moderate similarity
    """
    
    # Physics/cosmology concept patterns
    PHYSICS_CONCEPTS = {
        'quantum_info': [
            'von Neumann entropy', 'density matrix', 'qubit', 'entanglement',
            'unitarity', 'quantum information', 'superposition', 'decoherence',
            'quantum state', 'measurement', 'wave function collapse'
        ],
        'cosmology': [
            'FLRW', 'de Sitter', 'horizon', 'Hubble', 'inflation',
            'causal patch', 'cosmological constant', 'dark energy', 'dark matter',
            'expansion', 'redshift', 'CMB', 'cosmic microwave background'
        ],
        'quantum_gravity': [
            'Wheeler-DeWitt', 'holographic', 'Planck length', 'Planck scale',
            'covariant entropy bound', 'AdS/CFT', 'black hole', 'event horizon',
            'Hawking radiation', 'information paradox', 'string theory'
        ],
        'math_objects': [
            'Hilbert space', 'wavefunction', 'operator', 'eigenstate',
            'diffeomorphism', 'gauge invariant', 'manifold', 'metric',
            'tensor', 'Lagrangian', 'Hamiltonian', 'partition function'
        ],
        'entropy_concepts': [
            'entropy', 'thermodynamic', 'Boltzmann', 'Shannon entropy',
            'Bekenstein bound', 'holographic entropy', 'entanglement entropy',
            'coarse-graining', 'microstate', 'macrostate'
        ]
    }
    
    def __init__(self, session_id: str, corpus: Optional[SimilarityCorpus] = None):
        """
        Initialize the repetition detector.
        
        Args:
            session_id: Unique session identifier
            corpus: Optional existing corpus, creates new if not provided
        """
        self.session_id = session_id
        self.corpus = corpus or SimilarityCorpus(session_id)
        
        # Load default thresholds from environment
        self.DEFAULT_IDENTICAL_THRESHOLD = float(os.getenv("SIMILARITY_IDENTICAL_THRESHOLD", "0.90"))
        self.DEFAULT_VERY_SIMILAR_THRESHOLD = float(os.getenv("SIMILARITY_VERY_SIMILAR_THRESHOLD", "0.75"))
        self.DEFAULT_SIMILAR_THRESHOLD = float(os.getenv("SIMILARITY_SIMILAR_THRESHOLD", "0.60"))
        
        # Profile-specific thresholds
        self.PROFILE_THRESHOLDS = {
            'scientific': {
                'IDENTICAL': 0.92,  # More strict - formalism should be precise
                'VERY_SIMILAR': 0.78,
                'SIMILAR': 0.65
            },
            'popular_science': {
                'IDENTICAL': 0.88,  # Slightly more lenient - some repetition of analogies OK
                'VERY_SIMILAR': 0.75,
                'SIMILAR': 0.60
            },
            'educational': {
                'IDENTICAL': 0.90,
                'VERY_SIMILAR': 0.76,
                'SIMILAR': 0.62
            }
        }
        
        # Set default thresholds (can be overridden per analysis)
        self.IDENTICAL_THRESHOLD = self.DEFAULT_IDENTICAL_THRESHOLD
        self.VERY_SIMILAR_THRESHOLD = self.DEFAULT_VERY_SIMILAR_THRESHOLD
        self.SIMILAR_THRESHOLD = self.DEFAULT_SIMILAR_THRESHOLD
        
        # Information gain requirements
        self.MIN_NOVELTY_RATIO = float(os.getenv("MIN_NOVELTY_RATIO", "0.40"))
        
        # Behavioral flags
        self.AUTO_SKIP_IDENTICAL = os.getenv("AUTO_SKIP_IDENTICAL", "true").lower() == "true"
        self.ALWAYS_FLAG_NEVER_SKIP = os.getenv("ALWAYS_FLAG_NEVER_SKIP", "false").lower() == "true"
        
        # Storage for content history
        self.content_history = []
        self.embeddings_cache = {}
        
        logger.info(f"RepetitionDetector initialized for session {session_id}")
        logger.info(f"Default Thresholds: Identical={self.DEFAULT_IDENTICAL_THRESHOLD}, "
                   f"Very Similar={self.DEFAULT_VERY_SIMILAR_THRESHOLD}, "
                   f"Similar={self.DEFAULT_SIMILAR_THRESHOLD}")
    
    async def analyze_content(
        self,
        text: str,
        metadata: Dict[str, Any],
        profile: Optional[str] = None
    ) -> SimilarityDecision:
        """
        Analyze content for similarity and make a decision.
        
        Args:
            text: The content to analyze
            metadata: Content metadata (section, depth, type, etc.)
            profile: Optional profile name for profile-specific thresholds
            
        Returns:
            SimilarityDecision with action and reasoning
        """
        logger.debug(f"Analyzing content with metadata: {metadata}, profile: {profile}")
        
        # Update thresholds based on profile if provided
        if profile and profile in self.PROFILE_THRESHOLDS:
            self.IDENTICAL_THRESHOLD = self.PROFILE_THRESHOLDS[profile]['IDENTICAL']
            self.VERY_SIMILAR_THRESHOLD = self.PROFILE_THRESHOLDS[profile]['VERY_SIMILAR']
            self.SIMILAR_THRESHOLD = self.PROFILE_THRESHOLDS[profile]['SIMILAR']
            logger.debug(f"Using {profile} profile thresholds")
        else:
            # Use default thresholds
            self.IDENTICAL_THRESHOLD = self.DEFAULT_IDENTICAL_THRESHOLD
            self.VERY_SIMILAR_THRESHOLD = self.DEFAULT_VERY_SIMILAR_THRESHOLD
            self.SIMILAR_THRESHOLD = self.DEFAULT_SIMILAR_THRESHOLD
        
        # Search for similar content in corpus
        matches = self.corpus.search_similar_content(text)
        
        if not matches:
            logger.debug("No similar content found, accepting")
            return SimilarityDecision(
                action=DecisionAction.ACCEPT,
                reason="No similar content found",
                similarity_score=0.0,
                tier="NONE",
                details={"matches": 0, "profile": profile}
            )
        
        # Find the most similar match
        most_similar = max(matches, key=lambda m: m.similarity_score)
        similarity_score = most_similar.similarity_score
        
        # Get metadata of similar content
        # Note: ParagraphMatch doesn't have metadata, we'll use the stored_content_id
        similar_metadata = {
            'stored_content_id': most_similar.stored_content_id,
            'paragraph_index': most_similar.paragraph_index,
            'matched_index': most_similar.matched_index
        }
        
        logger.info(f"Most similar content has score: {similarity_score:.2%}")
        
        # Route to appropriate tier handler
        if similarity_score >= self.IDENTICAL_THRESHOLD:
            return await self._handle_tier1_identical(
                text, most_similar.matched_paragraph, similarity_score,
                metadata, similar_metadata, profile
            )
        elif similarity_score >= self.VERY_SIMILAR_THRESHOLD:
            return await self._handle_tier2_very_similar(
                text, most_similar.matched_paragraph, similarity_score,
                metadata, similar_metadata, profile
            )
        elif similarity_score >= self.SIMILAR_THRESHOLD:
            return await self._handle_tier3_somewhat_similar(
                text, most_similar.matched_paragraph, similarity_score,
                metadata, similar_metadata, profile
            )
        else:
            return SimilarityDecision(
                action=DecisionAction.ACCEPT,
                reason="Similarity below threshold",
                similarity_score=similarity_score,
                tier="LOW",
                details={"matches": len(matches), "profile": profile}
            )
    
    async def _handle_tier1_identical(
        self,
        text: str,
        similar_content: str,
        score: float,
        metadata: Dict[str, Any],
        similar_metadata: Dict[str, Any],
        profile: Optional[str] = None
    ) -> SimilarityDecision:
        """
        Handle near-identical content (≥0.90 similarity).
        
        This tier catches obvious duplicates like repetitive tables
        or paragraphs that are nearly word-for-word identical.
        """
        logger.info(f"TIER 1: Identical detection (score={score:.2%})")
        
        # Check if it's a structural element that's expected to be similar
        content_type = metadata.get('content_type', 'paragraph')
        
        if content_type in ['table_header', 'equation_definition']:
            # These elements naturally have similar structure
            # Check actual content similarity more strictly
            if score >= 0.95:
                if self.AUTO_SKIP_IDENTICAL and not self.ALWAYS_FLAG_NEVER_SKIP:
                    return SimilarityDecision(
                        action=DecisionAction.SKIP,
                        reason=f"Identical {content_type} content",
                        similarity_score=score,
                        tier="TIER_1",
                        details={
                            "content_type": content_type,
                            "threshold": self.IDENTICAL_THRESHOLD
                        },
                        similar_to={
                            "stored_content_id": similar_metadata.get('stored_content_id', 'unknown'),
                            "paragraph_index": similar_metadata.get('paragraph_index', 0),
                            "matched_index": similar_metadata.get('matched_index', 0),
                            "preview": similar_content[:200]
                        }
                    )
        
        # For prose/paragraphs
        if score >= 0.95:
            # Almost word-for-word identical
            if self.AUTO_SKIP_IDENTICAL and not self.ALWAYS_FLAG_NEVER_SKIP:
                return SimilarityDecision(
                    action=DecisionAction.SKIP,
                    reason=f"Near-identical content (similarity: {score:.2%})",
                    similarity_score=score,
                    tier="TIER_1",
                    details={
                        "content_type": content_type,
                        "threshold": self.IDENTICAL_THRESHOLD
                    },
                    similar_to={
                        "section": similar_metadata.get('section', 'unknown'),
                        "depth": similar_metadata.get('depth', 0),
                        "preview": similar_content[:200]
                    }
                )
        elif score >= self.IDENTICAL_THRESHOLD:
            # Very similar but not identical - check if different depth/section
            if (metadata.get('depth') != similar_metadata.get('depth') or
                metadata.get('section') != similar_metadata.get('section')):
                # Different context, downgrade to Tier 2 analysis
                return await self._handle_tier2_very_similar(
                    text, similar_content, score, metadata, similar_metadata, profile
                )
            else:
                # Same context, very similar - probably redundant
                if self.AUTO_SKIP_IDENTICAL and not self.ALWAYS_FLAG_NEVER_SKIP:
                    return SimilarityDecision(
                        action=DecisionAction.SKIP,
                        reason="Very similar content in same context",
                        similarity_score=score,
                        tier="TIER_1",
                        details={
                            "content_type": content_type,
                            "same_section": True,
                            "same_depth": True
                        },
                        similar_to={
                            "stored_content_id": similar_metadata.get('stored_content_id', 'unknown'),
                            "paragraph_index": similar_metadata.get('paragraph_index', 0),
                            "matched_index": similar_metadata.get('matched_index', 0),
                            "preview": similar_content[:200]
                        }
                    )
        
        # If we get here, flag for review
        return SimilarityDecision(
            action=DecisionAction.FLAG,
            reason="High similarity detected, review recommended",
            similarity_score=score,
            tier="TIER_1",
            details={"content_type": content_type, "profile": profile},
            similar_to={
                "section": similar_metadata.get('section', 'unknown'),
                "depth": similar_metadata.get('depth', 0),
                "preview": similar_content[:200]
            },
            recommendation="REVIEW - Content is very similar to existing material"
        )
    
    async def _handle_tier2_very_similar(
        self,
        text: str,
        similar_content: str,
        score: float,
        metadata: Dict[str, Any],
        similar_metadata: Dict[str, Any],
        profile: Optional[str] = None
    ) -> SimilarityDecision:
        """
        Semantic analysis for very similar content (0.75-0.90).
        
        This tier analyzes WHY content is similar and whether
        the similarity is justified.
        """
        logger.info(f"TIER 2: Semantic analysis (score={score:.2%})")
        
        # Extract concepts from both pieces of content
        new_concepts = self._extract_concepts(text)
        old_concepts = self._extract_concepts(similar_content)
        
        # Calculate concept overlap
        overlap = new_concepts.concepts.intersection(old_concepts.concepts)
        novel = new_concepts.concepts - old_concepts.concepts
        
        novelty_ratio = len(novel) / len(new_concepts.concepts) if new_concepts.concepts else 0
        
        # Check for different perspectives
        is_different_perspective = self._check_different_perspective(
            text, similar_content, new_concepts, old_concepts
        )
        
        # Build similarity analysis
        similarity_analysis = {
            'shared_concepts': list(overlap)[:10],  # Top 10 for readability
            'novel_concepts': list(novel)[:10],
            'novelty_ratio': novelty_ratio,
            'is_different_perspective': is_different_perspective,
            'has_new_equations': len(set(new_concepts.equations) - set(old_concepts.equations)) > 0,
            'has_new_examples': len(set(new_concepts.examples) - set(old_concepts.examples)) > 0,
            'has_new_references': len(set(new_concepts.references) - set(old_concepts.references)) > 0
        }
        
        # Decision logic
        if novelty_ratio >= self.MIN_NOVELTY_RATIO or is_different_perspective:
            # Sufficient novelty or different angle
            return SimilarityDecision(
                action=DecisionAction.FLAG,
                reason="Similar but introduces new concepts or perspective",
                similarity_score=score,
                tier="TIER_2",
                details=similarity_analysis,
                analysis=similarity_analysis,
                recommendation="REVIEW - May be acceptable if truly adding insight"
            )
        elif (similarity_analysis['has_new_equations'] or
              similarity_analysis['has_new_examples'] or
              similarity_analysis['has_new_references']):
            # Has some form of new information
            return SimilarityDecision(
                action=DecisionAction.FLAG,
                reason="Similar but contains new supporting material",
                similarity_score=score,
                tier="TIER_2",
                details=similarity_analysis,
                analysis=similarity_analysis,
                recommendation="REVIEW - Contains new examples/equations/references"
            )
        else:
            # Too similar without sufficient new information
            if not self.ALWAYS_FLAG_NEVER_SKIP:
                return SimilarityDecision(
                    action=DecisionAction.SKIP,
                    reason="Too similar without sufficient new information",
                    similarity_score=score,
                    tier="TIER_2",
                    details=similarity_analysis,
                    analysis=similarity_analysis
                )
            else:
                return SimilarityDecision(
                    action=DecisionAction.FLAG,
                    reason="Similar content flagged for review (auto-skip disabled)",
                    similarity_score=score,
                    tier="TIER_2",
                    details=similarity_analysis,
                    analysis=similarity_analysis,
                    recommendation="REVIEW - High similarity with limited new information"
                )
    
    async def _handle_tier3_somewhat_similar(
        self,
        text: str,
        similar_content: str,
        score: float,
        metadata: Dict[str, Any],
        similar_metadata: Dict[str, Any],
        profile: Optional[str] = None
    ) -> SimilarityDecision:
        """
        Information gain assessment for moderate similarity (0.60-0.75).
        
        At this level, some overlap is expected. Focus on whether
        the content adds value.
        """
        logger.info(f"TIER 3: Information gain check (score={score:.2%})")
        
        # Extract information signals
        info_gain = self._calculate_information_gain(text, similar_content)
        
        # Check various forms of information gain
        has_value = (
            info_gain['has_new_examples'] or
            info_gain['has_new_equations'] or
            info_gain['has_new_references'] or
            info_gain['deeper_explanation'] or
            info_gain['different_formalism']
        )
        
        if has_value:
            return SimilarityDecision(
                action=DecisionAction.ACCEPT,
                reason="Moderate similarity but adds new information",
                similarity_score=score,
                tier="TIER_3",
                details=info_gain
            )
        else:
            # No clear information gain
            return SimilarityDecision(
                action=DecisionAction.FLAG,
                reason="Moderate similarity without clear information gain",
                similarity_score=score,
                tier="TIER_3",
                details=info_gain,
                analysis=info_gain,
                recommendation="REVIEW - Consider if this depth of coverage is needed"
            )
    
    def _extract_concepts(self, text: str) -> ConceptExtraction:
        """
        Extract key concepts from text using pattern matching.
        
        This is a simple pattern-based approach that can be upgraded
        to LLM-based extraction if needed.
        """
        concepts = set()
        text_lower = text.lower()
        
        # Extract physics/cosmology concepts
        for category, terms in self.PHYSICS_CONCEPTS.items():
            for term in terms:
                if term.lower() in text_lower:
                    concepts.add(term)
        
        # Extract equations (LaTeX patterns)
        equation_patterns = [
            r'\$[^$]+\$',  # Inline math
            r'\\\[.*?\\\]',  # Display math
            r'\\begin\{equation\}.*?\\end\{equation\}',  # Equation environment
            r'[A-Z]\s*=\s*[^,\.\n]+',  # Simple equations like S = k log W
        ]
        
        equations = []
        for pattern in equation_patterns:
            equations.extend(re.findall(pattern, text, re.DOTALL))
        
        # Extract examples (look for indicator phrases)
        example_patterns = [
            r'[Ff]or example[,:].*?[\.!?]',
            r'[Cc]onsider.*?[\.!?]',
            r'[Ss]uppose.*?[\.!?]',
            r'[Ii]magine.*?[\.!?]',
        ]
        
        examples = []
        for pattern in example_patterns:
            examples.extend(re.findall(pattern, text))
        
        # Extract references (citations)
        reference_patterns = [
            r'[A-Z][a-z]+\s+\(\d{4}\)',  # Author (Year)
            r'[A-Z][a-z]+\s+&\s+[A-Z][a-z]+',  # Author & Author
            r'[A-Z][a-z]+\s+et\s+al\.',  # Author et al.
            r'\[\d+\]',  # Numbered references
        ]
        
        references = []
        for pattern in reference_patterns:
            references.extend(re.findall(pattern, text))
        
        # Extract technical terms (capitalized multi-word phrases)
        technical_pattern = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'
        technical_terms = set(re.findall(technical_pattern, text))
        
        return ConceptExtraction(
            concepts=concepts,
            equations=equations[:10],  # Limit for performance
            examples=examples[:5],
            references=references[:10],
            technical_terms=technical_terms
        )
    
    def _check_different_perspective(
        self,
        new_text: str,
        old_text: str,
        new_concepts: ConceptExtraction,
        old_concepts: ConceptExtraction
    ) -> bool:
        """
        Check if the new text presents a different perspective on similar concepts.
        
        Indicators of different perspective:
        - Different examples used
        - Different mathematical formulation
        - Different ordering/structure
        - Different emphasis or focus
        """
        # Different examples indicate different perspective
        if new_concepts.examples and old_concepts.examples:
            example_overlap = set(new_concepts.examples) & set(old_concepts.examples)
            if len(example_overlap) / len(new_concepts.examples) < 0.3:
                return True
        
        # Different equations/formalism
        if new_concepts.equations and old_concepts.equations:
            eq_overlap = set(new_concepts.equations) & set(old_concepts.equations)
            if len(eq_overlap) / len(new_concepts.equations) < 0.3:
                return True
        
        # Check for different emphasis (which concepts appear first)
        new_first_concepts = list(new_concepts.concepts)[:5]
        old_first_concepts = list(old_concepts.concepts)[:5]
        
        if new_first_concepts and old_first_concepts:
            if new_first_concepts != old_first_concepts:
                return True
        
        return False
    
    def _calculate_information_gain(
        self,
        new_text: str,
        old_text: str
    ) -> Dict[str, bool]:
        """
        Calculate various forms of information gain.
        
        Returns dict of signals indicating new information.
        """
        new_concepts = self._extract_concepts(new_text)
        old_concepts = self._extract_concepts(old_text)
        
        return {
            'has_new_examples': self._check_new_examples(new_concepts, old_concepts),
            'has_new_equations': self._check_new_equations(new_concepts, old_concepts),
            'has_new_references': self._check_new_references(new_concepts, old_concepts),
            'deeper_explanation': self._check_depth_increase(new_text, old_text),
            'different_formalism': self._check_different_formalism(new_concepts, old_concepts)
        }
    
    def _check_new_examples(
        self,
        new_concepts: ConceptExtraction,
        old_concepts: ConceptExtraction
    ) -> bool:
        """Check if there are new examples."""
        new_examples = set(new_concepts.examples)
        old_examples = set(old_concepts.examples)
        return len(new_examples - old_examples) > 0
    
    def _check_new_equations(
        self,
        new_concepts: ConceptExtraction,
        old_concepts: ConceptExtraction
    ) -> bool:
        """Check if there are new equations."""
        new_eqs = set(new_concepts.equations)
        old_eqs = set(old_concepts.equations)
        return len(new_eqs - old_eqs) > 0
    
    def _check_new_references(
        self,
        new_concepts: ConceptExtraction,
        old_concepts: ConceptExtraction
    ) -> bool:
        """Check if there are new references."""
        new_refs = set(new_concepts.references)
        old_refs = set(old_concepts.references)
        return len(new_refs - old_refs) > 0
    
    def _check_depth_increase(self, new_text: str, old_text: str) -> bool:
        """
        Check if the new text provides deeper explanation.
        
        Indicators:
        - Longer average sentence length
        - More technical terms
        - More mathematical content
        """
        new_sentences = re.split(r'[.!?]+', new_text)
        old_sentences = re.split(r'[.!?]+', old_text)
        
        # Check if new text is substantially longer
        if len(new_text) > len(old_text) * 1.3:
            return True
        
        # Check for more technical depth (more equations)
        new_math_count = len(re.findall(r'\$[^$]+\$', new_text))
        old_math_count = len(re.findall(r'\$[^$]+\$', old_text))
        
        if new_math_count > old_math_count * 1.5:
            return True
        
        return False
    
    def _check_different_formalism(
        self,
        new_concepts: ConceptExtraction,
        old_concepts: ConceptExtraction
    ) -> bool:
        """
        Check if content uses different formalism or notation.
        
        Different mathematical notation or terminology for same concepts
        indicates a different approach.
        """
        # Check for different equation styles
        if new_concepts.equations and old_concepts.equations:
            # Simple check: are the equations structurally different?
            new_eq_structures = {self._get_equation_structure(eq) for eq in new_concepts.equations}
            old_eq_structures = {self._get_equation_structure(eq) for eq in old_concepts.equations}
            
            overlap = new_eq_structures & old_eq_structures
            if len(overlap) / len(new_eq_structures) < 0.5:
                return True
        
        return False
    
    def _get_equation_structure(self, equation: str) -> str:
        """
        Get a simplified structure of an equation for comparison.
        
        This removes specific variables but keeps the structure.
        """
        # Replace specific numbers and variables with placeholders
        structure = re.sub(r'\b[a-z]\b', 'VAR', equation)
        structure = re.sub(r'\d+', 'NUM', structure)
        return structure
    
    async def store_content(
        self,
        text: str,
        embedding: Optional[Any],
        metadata: Dict[str, Any]
    ):
        """
        Store content for future comparison.
        
        Args:
            text: Content to store
            embedding: Optional pre-computed embedding
            metadata: Content metadata
        """
        # Store in corpus
        agent_id = metadata.get('agent_id', 'unknown')
        content_id = self.corpus.store_content(text, agent_id, metadata)
        
        # Cache locally
        self.content_history.append({
            'id': content_id,
            'text': text,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        })
        
        if embedding is not None:
            self.embeddings_cache[content_id] = embedding
        
        logger.debug(f"Stored content with ID: {content_id}")