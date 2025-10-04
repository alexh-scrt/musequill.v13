"""
Evaluator Profile Factory - Domain-specific quality assessment profiles

Provides customized evaluation criteria for different content types:
- Scientific: Rigorous academic/research content
- Popular Science: Accessible science communication
- Technology: Tech reviews, tutorials, product analysis
- Investment: Financial analysis, market insights
- General: Broad audience content
- Creative: Narrative, storytelling, artistic content
"""

from typing import Dict, Any, Literal

ProfileType = Literal[
    "scientific",
    "popular_science", 
    "technology",
    "investment",
    "general",
    "creative"
]


class EvaluatorProfileFactory:
    """
    Factory for creating domain-specific evaluator profiles.
    
    Each profile customizes:
    - Metric weights (importance of each quality dimension)
    - Thresholds (minimum acceptable scores)
    - Critical metrics (auto-reject criteria)
    - Quality tiers (score ranges for different quality levels)
    """
    
    @staticmethod
    def get(profile: ProfileType = "general") -> Dict[str, Any]:
        """
        Get evaluator configuration for a specific content domain.
        
        Args:
            profile: Content domain type
            
        Returns:
            Dictionary with metric_config, tier_thresholds, critical_metrics
        """
        profiles = {
            "scientific": EvaluatorProfileFactory._scientific_profile(),
            "popular_science": EvaluatorProfileFactory._popular_science_profile(),
            "technology": EvaluatorProfileFactory._technology_profile(),
            "investment": EvaluatorProfileFactory._investment_profile(),
            "general": EvaluatorProfileFactory._general_profile(),
            "creative": EvaluatorProfileFactory._creative_profile()
        }
        
        if profile not in profiles:
            raise ValueError(
                f"Invalid profile '{profile}'. "
                f"Must be one of: {list(profiles.keys())}"
            )
        
        return profiles[profile]
    
    @staticmethod
    def _scientific_profile() -> Dict[str, Any]:
        """
        Profile for rigorous scientific/academic content.
        
        Priorities:
        - Mathematical rigor (proofs, citations)
        - Conceptual novelty (original contributions)
        - Minimal redundancy (tight writing)
        - Empirical grounding (testable claims)
        """
        return {
            "name": "Scientific Research",
            "description": "Rigorous academic and research content",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 15,
                    'min_threshold': 6,
                    'optimal': 15,
                    'calculation': 'CNR >= 40%'
                },
                'claim_density': {
                    'weight': 10,
                    'min_threshold': 5,
                    'optimal': 10,
                    'calculation': '>= 0.8 claims/100 words'
                },
                'mathematical_rigor': {
                    'weight': 20,  # Higher weight for scientific
                    'min_threshold': 12,  # Stricter threshold
                    'optimal': 20,
                    'calculation': '90%+ proofs, 80%+ citations'
                },
                'semantic_compression': {
                    'weight': 15,
                    'min_threshold': 7,
                    'optimal': 15,
                    'calculation': 'SCR <= 5:1'
                },
                'citation_context': {
                    'weight': 15,  # Higher weight for scientific
                    'min_threshold': 8,  # Stricter threshold
                    'optimal': 15,
                    'calculation': '15-40 cites/10pg + novelty statement'
                },
                'empirical_grounding': {
                    'weight': 15,  # Higher weight for scientific
                    'min_threshold': 6,  # Stricter threshold
                    'optimal': 15,
                    'calculation': '2+ testable predictions'
                },
                'structural_coherence': {
                    'weight': 5,
                    'min_threshold': 3,
                    'optimal': 5,
                    'calculation': 'No circular reasoning'
                },
                'notation_consistency': {
                    'weight': 3,
                    'min_threshold': 2,
                    'optimal': 3,
                    'calculation': '95%+ consistency'
                },
                'figure_utility': {
                    'weight': 1,
                    'min_threshold': 0,
                    'optimal': 1,
                    'calculation': '80%+ informative'
                },
                'parsimony': {
                    'weight': 1,
                    'min_threshold': 0,
                    'optimal': 1,
                    'calculation': '70%+ essential assumptions'
                }
            },
            "tier_thresholds": {
                'excellent': 85,
                'good': 70,
                'acceptable': 55,
                'poor': 35,
                'unacceptable': 0
            },
            "critical_metrics": [
                'mathematical_rigor',
                'citation_context',
                'empirical_grounding'
            ]
        }
    
    @staticmethod
    def _popular_science_profile() -> Dict[str, Any]:
        """
        Profile for accessible science communication.
        
        Priorities:
        - Clarity and accessibility
        - Engaging examples
        - Accurate but simplified
        - Good citations (credibility)
        """
        return {
            "name": "Popular Science",
            "description": "Accessible science communication for broad audiences",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 10,
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'CNR >= 30%'
                },
                'claim_density': {
                    'weight': 15,  # Higher weight - clear assertions
                    'min_threshold': 8,
                    'optimal': 15,
                    'calculation': '>= 0.6 claims/100 words'
                },
                'mathematical_rigor': {
                    'weight': 5,  # Lower weight - accessibility > rigor
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Key claims supported'
                },
                'semantic_compression': {
                    'weight': 10,  # Some explanation needed
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'SCR <= 7:1'
                },
                'citation_context': {
                    'weight': 15,  # Important for credibility
                    'min_threshold': 6,
                    'optimal': 15,
                    'calculation': '5-20 cites/10pg + source clarity'
                },
                'empirical_grounding': {
                    'weight': 20,  # Very important - real examples
                    'min_threshold': 10,
                    'optimal': 20,
                    'calculation': 'Multiple concrete examples'
                },
                'structural_coherence': {
                    'weight': 15,  # Clear flow is critical
                    'min_threshold': 8,
                    'optimal': 15,
                    'calculation': 'Linear narrative flow'
                },
                'notation_consistency': {
                    'weight': 3,
                    'min_threshold': 1,
                    'optimal': 3,
                    'calculation': 'Minimal jargon, consistent terms'
                },
                'figure_utility': {
                    'weight': 5,  # Visuals help accessibility
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Helpful illustrations'
                },
                'parsimony': {
                    'weight': 2,
                    'min_threshold': 1,
                    'optimal': 2,
                    'calculation': 'Simplified appropriately'
                }
            },
            "tier_thresholds": {
                'excellent': 80,
                'good': 65,
                'acceptable': 50,
                'poor': 35,
                'unacceptable': 0
            },
            "critical_metrics": [
                'claim_density',
                'citation_context',
                'structural_coherence'
            ]
        }
    
    @staticmethod
    def _technology_profile() -> Dict[str, Any]:
        """
        Profile for technology reviews, tutorials, product analysis.
        
        Priorities:
        - Practical value (actionable insights)
        - Real-world examples
        - Clear structure
        - Balanced perspective
        """
        return {
            "name": "Technology Review",
            "description": "Tech reviews, tutorials, and product analysis",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 15,  # New insights valuable
                    'min_threshold': 6,
                    'optimal': 15,
                    'calculation': 'CNR >= 35%'
                },
                'claim_density': {
                    'weight': 20,  # Specific assertions critical
                    'min_threshold': 10,
                    'optimal': 20,
                    'calculation': '>= 0.7 claims/100 words'
                },
                'mathematical_rigor': {
                    'weight': 3,  # Less critical for tech content
                    'min_threshold': 1,
                    'optimal': 3,
                    'calculation': 'Technical accuracy'
                },
                'semantic_compression': {
                    'weight': 10,
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'SCR <= 6:1'
                },
                'citation_context': {
                    'weight': 10,  # Sources for specs/benchmarks
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': '3-15 cites/10pg + source links'
                },
                'empirical_grounding': {
                    'weight': 25,  # Most important - real examples
                    'min_threshold': 13,
                    'optimal': 25,
                    'calculation': 'Concrete examples, benchmarks, tests'
                },
                'structural_coherence': {
                    'weight': 10,
                    'min_threshold': 5,
                    'optimal': 10,
                    'calculation': 'Clear organization'
                },
                'notation_consistency': {
                    'weight': 2,
                    'min_threshold': 1,
                    'optimal': 2,
                    'calculation': 'Consistent terminology'
                },
                'figure_utility': {
                    'weight': 3,
                    'min_threshold': 1,
                    'optimal': 3,
                    'calculation': 'Screenshots, diagrams'
                },
                'parsimony': {
                    'weight': 2,
                    'min_threshold': 1,
                    'optimal': 2,
                    'calculation': 'Focused scope'
                }
            },
            "tier_thresholds": {
                'excellent': 80,
                'good': 65,
                'acceptable': 50,
                'poor': 35,
                'unacceptable': 0
            },
            "critical_metrics": [
                'claim_density',
                'empirical_grounding'
            ]
        }
    
    @staticmethod
    def _investment_profile() -> Dict[str, Any]:
        """
        Profile for financial analysis and investment insights.
        
        Priorities:
        - Data-driven claims
        - Risk assessment
        - Clear reasoning
        - Source credibility
        """
        return {
            "name": "Investment & Finance",
            "description": "Financial analysis, market insights, investment recommendations",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 10,
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'CNR >= 30%'
                },
                'claim_density': {
                    'weight': 25,  # Very high - specific assertions critical
                    'min_threshold': 13,
                    'optimal': 25,
                    'calculation': '>= 0.8 claims/100 words'
                },
                'mathematical_rigor': {
                    'weight': 10,  # Calculations must be sound
                    'min_threshold': 5,
                    'optimal': 10,
                    'calculation': 'Math/calculations verified'
                },
                'semantic_compression': {
                    'weight': 8,
                    'min_threshold': 3,
                    'optimal': 8,
                    'calculation': 'SCR <= 6:1'
                },
                'citation_context': {
                    'weight': 20,  # Sources absolutely critical
                    'min_threshold': 10,
                    'optimal': 20,
                    'calculation': 'Data sources cited, recent'
                },
                'empirical_grounding': {
                    'weight': 20,  # Data-driven is essential
                    'min_threshold': 10,
                    'optimal': 20,
                    'calculation': 'Historical data, metrics, ratios'
                },
                'structural_coherence': {
                    'weight': 5,
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Logical argument flow'
                },
                'notation_consistency': {
                    'weight': 1,
                    'min_threshold': 0,
                    'optimal': 1,
                    'calculation': 'Consistent metrics/terms'
                },
                'figure_utility': {
                    'weight': 1,
                    'min_threshold': 0,
                    'optimal': 1,
                    'calculation': 'Charts, tables'
                },
                'parsimony': {
                    'weight': 0,
                    'min_threshold': 0,
                    'optimal': 0,
                    'calculation': 'N/A'
                }
            },
            "tier_thresholds": {
                'excellent': 85,  # High bar for investment advice
                'good': 70,
                'acceptable': 55,
                'poor': 40,
                'unacceptable': 0
            },
            "critical_metrics": [
                'claim_density',
                'citation_context',
                'empirical_grounding'
            ]
        }
    
    @staticmethod
    def _general_profile() -> Dict[str, Any]:
        """
        Profile for general-purpose content.
        
        Balanced across all dimensions - suitable for:
        - Blog posts
        - Tutorials
        - Explanatory content
        - Mixed-domain writing
        """
        return {
            "name": "General Purpose",
            "description": "Balanced evaluation for broad-audience content",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 12,
                    'min_threshold': 5,
                    'optimal': 12,
                    'calculation': 'CNR >= 30%'
                },
                'claim_density': {
                    'weight': 15,
                    'min_threshold': 7,
                    'optimal': 15,
                    'calculation': '>= 0.6 claims/100 words'
                },
                'mathematical_rigor': {
                    'weight': 5,
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Claims reasonably supported'
                },
                'semantic_compression': {
                    'weight': 12,
                    'min_threshold': 5,
                    'optimal': 12,
                    'calculation': 'SCR <= 6:1'
                },
                'citation_context': {
                    'weight': 10,
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': '3-20 cites/10pg when applicable'
                },
                'empirical_grounding': {
                    'weight': 15,
                    'min_threshold': 7,
                    'optimal': 15,
                    'calculation': 'Examples and evidence'
                },
                'structural_coherence': {
                    'weight': 15,
                    'min_threshold': 7,
                    'optimal': 15,
                    'calculation': 'Clear flow and organization'
                },
                'notation_consistency': {
                    'weight': 5,
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Consistent terminology'
                },
                'figure_utility': {
                    'weight': 6,
                    'min_threshold': 2,
                    'optimal': 6,
                    'calculation': 'Helpful visuals'
                },
                'parsimony': {
                    'weight': 5,
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Appropriate scope'
                }
            },
            "tier_thresholds": {
                'excellent': 80,
                'good': 65,
                'acceptable': 50,
                'poor': 35,
                'unacceptable': 0
            },
            "critical_metrics": [
                'structural_coherence'
            ]
        }
    
    @staticmethod
    def _creative_profile() -> Dict[str, Any]:
        """
        Profile for creative/narrative content.
        
        Priorities:
        - Engaging storytelling
        - Minimal redundancy
        - Flow and pacing
        - Novelty in ideas/perspective
        """
        return {
            "name": "Creative Writing",
            "description": "Narrative, storytelling, and artistic content",
            "metric_config": {
                'conceptual_novelty': {
                    'weight': 25,  # Originality is paramount
                    'min_threshold': 13,
                    'optimal': 25,
                    'calculation': 'CNR >= 50%'
                },
                'claim_density': {
                    'weight': 5,  # Not claim-focused
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Meaningful moments'
                },
                'mathematical_rigor': {
                    'weight': 0,  # Not applicable
                    'min_threshold': 0,
                    'optimal': 0,
                    'calculation': 'N/A'
                },
                'semantic_compression': {
                    'weight': 20,  # Tight prose is valuable
                    'min_threshold': 10,
                    'optimal': 20,
                    'calculation': 'SCR <= 4:1'
                },
                'citation_context': {
                    'weight': 0,  # Not typically applicable
                    'min_threshold': 0,
                    'optimal': 0,
                    'calculation': 'N/A'
                },
                'empirical_grounding': {
                    'weight': 10,  # Vivid details matter
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'Concrete sensory details'
                },
                'structural_coherence': {
                    'weight': 25,  # Narrative flow critical
                    'min_threshold': 13,
                    'optimal': 25,
                    'calculation': 'Strong narrative arc'
                },
                'notation_consistency': {
                    'weight': 5,
                    'min_threshold': 2,
                    'optimal': 5,
                    'calculation': 'Voice consistency'
                },
                'figure_utility': {
                    'weight': 0,  # Not typically applicable
                    'min_threshold': 0,
                    'optimal': 0,
                    'calculation': 'N/A'
                },
                'parsimony': {
                    'weight': 10,  # Every word should matter
                    'min_threshold': 4,
                    'optimal': 10,
                    'calculation': 'No wasted prose'
                }
            },
            "tier_thresholds": {
                'excellent': 85,
                'good': 70,
                'acceptable': 55,
                'poor': 40,
                'unacceptable': 0
            },
            "critical_metrics": [
                'conceptual_novelty',
                'structural_coherence',
                'semantic_compression'
            ]
        }
    
    @staticmethod
    def list_profiles() -> Dict[str, str]:
        """Get a dictionary of available profiles and their descriptions."""
        return {
            "scientific": "Rigorous academic and research content",
            "popular_science": "Accessible science communication for broad audiences",
            "technology": "Tech reviews, tutorials, and product analysis",
            "investment": "Financial analysis, market insights, investment recommendations",
            "general": "Balanced evaluation for broad-audience content",
            "creative": "Narrative, storytelling, and artistic content"
        }
    
    @staticmethod
    def get_profile_info(profile: ProfileType) -> Dict[str, Any]:
        """Get detailed information about a specific profile."""
        config = EvaluatorProfileFactory.get(profile)
        
        # Calculate total weight
        total_weight = sum(
            m['weight'] for m in config['metric_config'].values()
        )
        
        # Get critical metrics
        critical = config.get('critical_metrics', [])
        
        return {
            "name": config['name'],
            "description": config['description'],
            "total_weight": total_weight,
            "critical_metrics": critical,
            "tier_thresholds": config['tier_thresholds'],
            "top_priorities": sorted(
                config['metric_config'].items(),
                key=lambda x: x[1]['weight'],
                reverse=True
            )[:3]
        }