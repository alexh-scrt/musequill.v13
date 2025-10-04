"""
Enhanced EvaluatorAgent with domain-specific profiles

Key changes:
- Accepts profile parameter in __init__
- Loads metric config from profile
- Dynamic metric weights and thresholds
- Profile-aware feedback generation
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from src.agents.base import BaseAgent
from src.agents.evaluator_profiles import EvaluatorProfileFactory, ProfileType
from src.agents.metrics import (
    calculate_cnr,
    calculate_claim_density,
    calculate_mathematical_rigor,
    calculate_compression_ratio,
    calculate_citation_metrics,
    calculate_empirical_grounding,
    calculate_structural_coherence,
    calculate_notation_consistency,
    calculate_figure_utility,
    calculate_parsimony
)

logger = logging.getLogger(__name__)


@dataclass
class MetricScore:
    """Individual metric score with details"""
    name: str
    score: float  # 0-max_points
    max_points: float
    percentage: float  # 0-100
    threshold_met: bool
    details: Dict[str, Any]
    suggestions: List[str]


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    total_score: float  # 0-100
    tier: str  # Unacceptable, Poor, Acceptable, Good, Excellent
    metrics: Dict[str, MetricScore]
    critical_failures: List[str]
    priority_actions: List[str]
    section_analysis: Dict[str, Dict]
    pass_threshold: bool
    detailed_feedback: str
    profile_name: str  # Which profile was used


class EvaluatorAgent(BaseAgent):
    """Agent that evaluates content quality using domain-specific profiles"""
    
    def __init__(
        self,
        agent_id: str = "evaluator",
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        profile: ProfileType = "general"
    ):
        """
        Initialize evaluator with a specific profile.
        
        Args:
            agent_id: Agent identifier
            model: LLM model to use
            session_id: Session identifier
            llm_params: LLM parameters
            profile: Content domain profile (scientific, technology, etc.)
        """
        super().__init__(
            agent_id,
            web_search=False,  # Evaluator doesn't need web search
            model=model,
            session_id=session_id,
            llm_params=llm_params
        )
        
        # Load profile configuration
        self.profile = profile
        profile_config = EvaluatorProfileFactory.get(profile)
        
        self.profile_name = profile_config['name']
        self.profile_description = profile_config['description']
        self.METRIC_CONFIG = profile_config['metric_config']
        self.TIER_THRESHOLDS = profile_config['tier_thresholds']
        self.CRITICAL_METRICS = profile_config.get('critical_metrics', [])
        
        logger.info(
            f"EvaluatorAgent initialized with model: {self.model}, "
            f"profile: {self.profile_name}"
        )
        logger.info(f"Critical metrics: {self.CRITICAL_METRICS}")
    
    async def evaluate(
        self,
        content: str,
        previous_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate content quality across all metrics using the selected profile.
        
        Args:
            content: The content to evaluate
            previous_content: Previous iteration for novelty comparison
            context: Additional context (references, figures, etc.)
            
        Returns:
            EvaluationResult with scores and feedback
        """
        logger.info(f"Starting content evaluation with profile: {self.profile_name}")
        
        # Calculate all metrics
        metrics = {}
        
        # 1. Conceptual Novelty Rate
        if self.METRIC_CONFIG['conceptual_novelty']['weight'] > 0:
            cnr_data = calculate_cnr(content, previous_content)
            metrics['conceptual_novelty'] = self._score_metric(
                'conceptual_novelty',
                cnr_data['score'],
                cnr_data
            )
        
        # 2. Claim Density
        if self.METRIC_CONFIG['claim_density']['weight'] > 0:
            cd_data = calculate_claim_density(content)
            metrics['claim_density'] = self._score_metric(
                'claim_density',
                cd_data['score'],
                cd_data
            )
        
        # 3. Mathematical Rigor
        if self.METRIC_CONFIG['mathematical_rigor']['weight'] > 0:
            mr_data = calculate_mathematical_rigor(content)
            metrics['mathematical_rigor'] = self._score_metric(
                'mathematical_rigor',
                mr_data['score'],
                mr_data
            )
        
        # 4. Semantic Compression Ratio
        if self.METRIC_CONFIG['semantic_compression']['weight'] > 0:
            scr_data = calculate_compression_ratio(content, previous_content)
            metrics['semantic_compression'] = self._score_metric(
                'semantic_compression',
                scr_data['score'],
                scr_data
            )
        
        # 5. Citation & Context
        if self.METRIC_CONFIG['citation_context']['weight'] > 0:
            citation_data = calculate_citation_metrics(content, context)
            metrics['citation_context'] = self._score_metric(
                'citation_context',
                citation_data['score'],
                citation_data
            )
        
        # 6. Empirical Grounding
        if self.METRIC_CONFIG['empirical_grounding']['weight'] > 0:
            eg_data = calculate_empirical_grounding(content)
            metrics['empirical_grounding'] = self._score_metric(
                'empirical_grounding',
                eg_data['score'],
                eg_data
            )
        
        # 7. Structural Coherence
        if self.METRIC_CONFIG['structural_coherence']['weight'] > 0:
            sc_data = calculate_structural_coherence(content)
            metrics['structural_coherence'] = self._score_metric(
                'structural_coherence',
                sc_data['score'],
                sc_data
            )
        
        # 8. Notation Consistency
        if self.METRIC_CONFIG['notation_consistency']['weight'] > 0:
            nc_data = calculate_notation_consistency(content)
            metrics['notation_consistency'] = self._score_metric(
                'notation_consistency',
                nc_data['score'],
                nc_data
            )
        
        # 9. Figure/Equation Utility
        if self.METRIC_CONFIG['figure_utility']['weight'] > 0:
            fe_data = calculate_figure_utility(content, context)
            metrics['figure_utility'] = self._score_metric(
                'figure_utility',
                fe_data['score'],
                fe_data
            )
        
        # 10. Parsimony
        if self.METRIC_CONFIG['parsimony']['weight'] > 0:
            pars_data = calculate_parsimony(content)
            metrics['parsimony'] = self._score_metric(
                'parsimony',
                pars_data['score'],
                pars_data
            )
        
        # Calculate total score
        total_score = sum(m.score for m in metrics.values())
        
        # Determine tier
        tier = self._determine_tier(total_score)
        
        # Check critical failures
        critical_failures = self._check_critical_failures(metrics)
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(metrics)
        
        # Section-level analysis
        section_analysis = self._analyze_sections(content, previous_content)
        
        # Check if passes threshold - use environment variable or profile default
        quality_threshold = float(
            os.getenv("QUALITY_THRESHOLD", str(self.TIER_THRESHOLDS['acceptable']))
        )
        pass_threshold = (
            total_score >= quality_threshold and
            len(critical_failures) == 0
        )
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(
            total_score,
            tier,
            metrics,
            critical_failures,
            priority_actions
        )
        
        result = EvaluationResult(
            total_score=total_score,
            tier=tier,
            metrics=metrics,
            critical_failures=critical_failures,
            priority_actions=priority_actions,
            section_analysis=section_analysis,
            pass_threshold=pass_threshold,
            detailed_feedback=detailed_feedback,
            profile_name=self.profile_name
        )
        
        logger.info(
            f"Evaluation complete: {total_score:.1f}/100 ({tier}) "
            f"[Profile: {self.profile_name}]"
        )
        return result
    
    def _score_metric(
        self,
        metric_name: str,
        raw_score: float,
        details: Dict[str, Any]
    ) -> MetricScore:
        """Convert raw metric score to MetricScore object"""
        config = self.METRIC_CONFIG[metric_name]
        max_points = config['weight']
        
        # Normalize score to max_points
        if 'percentage' in details:
            normalized = (details['percentage'] / 100) * max_points
        else:
            normalized = raw_score
        
        # Clamp to [0, max_points]
        normalized = max(0, min(normalized, max_points))
        
        percentage = (normalized / max_points) * 100 if max_points > 0 else 0
        threshold_met = normalized >= config['min_threshold']
        
        # Generate suggestions if below threshold
        suggestions = []
        if not threshold_met:
            suggestions = self._generate_metric_suggestions(
                metric_name,
                normalized,
                config['min_threshold'],
                details
            )
        
        return MetricScore(
            name=metric_name,
            score=normalized,
            max_points=max_points,
            percentage=percentage,
            threshold_met=threshold_met,
            details=details,
            suggestions=suggestions
        )
    
    def _determine_tier(self, total_score: float) -> str:
        """Determine quality tier from total score"""
        if total_score >= self.TIER_THRESHOLDS['excellent']:
            return "Excellent"
        elif total_score >= self.TIER_THRESHOLDS['good']:
            return "Good"
        elif total_score >= self.TIER_THRESHOLDS['acceptable']:
            return "Acceptable"
        elif total_score >= self.TIER_THRESHOLDS['poor']:
            return "Poor - Significant Revision Needed"
        else:
            return "Unacceptable - Major Revision Required"
    
    def _check_critical_failures(
        self,
        metrics: Dict[str, MetricScore]
    ) -> List[str]:
        """Check for critical failures that trigger auto-reject"""
        failures = []
        
        for metric_name in self.CRITICAL_METRICS:
            if metric_name in metrics:
                metric = metrics[metric_name]
                if not metric.threshold_met:
                    failures.append(
                        f"{metric.name}: {metric.score:.1f}/{metric.max_points} "
                        f"(need {self.METRIC_CONFIG[metric_name]['min_threshold']})"
                    )
        
        return failures
    
    def _generate_priority_actions(
        self,
        metrics: Dict[str, MetricScore]
    ) -> List[str]:
        """Generate prioritized list of actions to improve content"""
        actions = []
        
        # Sort metrics by how far below threshold they are
        issues = []
        for name, metric in metrics.items():
            if not metric.threshold_met and metric.max_points > 0:
                threshold = self.METRIC_CONFIG[name]['min_threshold']
                deficit = threshold - metric.score
                issues.append((deficit / metric.max_points, name, metric))
        
        issues.sort(reverse=True)  # Biggest deficits first
        
        # Generate actions for top issues
        for _, name, metric in issues[:5]:  # Top 5 priorities
            actions.extend(metric.suggestions)
        
        return actions
    
    def _analyze_sections(
        self,
        content: str,
        previous_content: Optional[str]
    ) -> Dict[str, Dict]:
        """Analyze content at section level"""
        # Split into sections (basic implementation)
        sections = self._split_sections(content)
        
        analysis = {}
        for i, section in enumerate(sections):
            section_id = f"section_{i+1}"
            
            # Calculate novelty for this section
            cnr_data = calculate_cnr(section, previous_content)
            
            # Determine action
            if cnr_data['percentage'] >= 40:
                action = "keep"
            elif cnr_data['percentage'] >= 20:
                action = "refine"
            elif i > 0:  # Don't merge section 0
                action = f"merge_with_{i}"
            else:
                action = "rewrite"
            
            analysis[section_id] = {
                'cnr': cnr_data['percentage'],
                'word_count': len(section.split()),
                'action': action,
                'keep': action == "keep"
            }
        
        return analysis
    
    def _split_sections(self, content: str) -> List[str]:
        """Split content into logical sections"""
        sections = content.split('\n\n')
        return [s.strip() for s in sections if s.strip()]
    
    def _generate_metric_suggestions(
        self,
        metric_name: str,
        current_score: float,
        threshold: float,
        details: Dict[str, Any]
    ) -> List[str]:
        """Generate specific suggestions for improving a metric"""
        suggestions = []
        
        if metric_name == 'conceptual_novelty':
            if details.get('percentage', 0) < 25:
                suggestions.append(
                    f"Consolidate sections: CNR {details.get('percentage', 0):.1f}% "
                    f"(need 25%+ minimum). Remove redundant explanations."
                )
        
        elif metric_name == 'semantic_compression':
            if details.get('ratio', 0) > 5:
                suggestions.append(
                    f"Reduce redundancy: SCR {details.get('ratio', 0):.1f}:1 "
                    f"(need ≤5:1). Content can be compressed by "
                    f"{(1 - 1/details.get('ratio', 1)) * 100:.0f}%."
                )
        
        elif metric_name == 'mathematical_rigor':
            unproved = details.get('unproved_theorems', [])
            if unproved:
                suggestions.append(
                    f"Add proofs for: {', '.join(unproved[:3])}"
                    f"{' and more' if len(unproved) > 3 else ''}"
                )
            
            uncited = details.get('uncited_claims', 0)
            if uncited > 0:
                suggestions.append(
                    f"Add {uncited} missing citations for established results"
                )
        
        elif metric_name == 'claim_density':
            current_density = details.get('density', 0)
            suggestions.append(
                f"Increase substantive claims: {current_density:.2f} claims/100w "
                f"(need 0.4+ minimum). Remove filler prose."
            )
        
        elif metric_name == 'citation_context':
            if details.get('citation_count', 0) < 5:
                suggestions.append(
                    "Add citations: situate work in existing literature (need 5+ citations)"
                )
            if not details.get('has_related_work', False):
                suggestions.append(
                    "Add 'Related Work' section explaining what's new vs. established"
                )
        
        elif metric_name == 'empirical_grounding':
            if details.get('predictions', 0) < 1:
                suggestions.append(
                    "Add concrete examples, test results, or real-world data"
                )
        
        elif metric_name == 'structural_coherence':
            if details.get('circular_patterns', 0) > 2:
                suggestions.append(
                    f"Fix circular reasoning: {details.get('circular_patterns')} patterns detected"
                )
        
        return suggestions
    
    def _generate_detailed_feedback(
        self,
        total_score: float,
        tier: str,
        metrics: Dict[str, MetricScore],
        critical_failures: List[str],
        priority_actions: List[str]
    ) -> str:
        """Generate human-readable detailed feedback"""
        feedback_parts = []
        
        # Header with profile info
        feedback_parts.append(f"EVALUATION PROFILE: {self.profile_name}")
        feedback_parts.append(f"OVERALL SCORE: {total_score:.1f}/100 ({tier})")
        feedback_parts.append("")
        
        # Critical failures
        if critical_failures:
            feedback_parts.append("CRITICAL FAILURES (Auto-reject):")
            for failure in critical_failures:
                feedback_parts.append(f"  ✗ {failure}")
            feedback_parts.append("")
        
        # Metric breakdown - only show metrics with weight > 0
        feedback_parts.append("METRIC BREAKDOWN:")
        active_metrics = [
            (name, metric) for name, metric in metrics.items() 
            if metric.max_points > 0
        ]
        
        # Sort by weight (descending) for better readability
        active_metrics.sort(key=lambda x: x[1].max_points, reverse=True)
        
        for name, metric in active_metrics:
            status = "✓" if metric.threshold_met else "✗"
            feedback_parts.append(
                f"  {status} {name}: {metric.score:.1f}/{metric.max_points} "
                f"({metric.percentage:.0f}%)"
            )
        feedback_parts.append("")
        
        # Priority actions
        if priority_actions:
            feedback_parts.append("PRIORITY ACTIONS:")
            for i, action in enumerate(priority_actions[:10], 1):
                feedback_parts.append(f"  {i}. {action}")
            feedback_parts.append("")
        
        # Recommendation
        if total_score >= self.TIER_THRESHOLDS['good']:
            feedback_parts.append("RECOMMENDATION: Accept with minor revisions")
        elif total_score >= self.TIER_THRESHOLDS['acceptable']:
            feedback_parts.append("RECOMMENDATION: Revise and resubmit")
        else:
            feedback_parts.append("RECOMMENDATION: Major revision or reject")
        
        return "\n".join(feedback_parts)
    
    async def process(
        self,
        prompt: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process method for BaseAgent compatibility
        Evaluates content and returns feedback string
        """
        previous_content = state.get('previous_content') if state else None
        eval_context = state.get('context', {}) if state else {}
        
        result = await self.evaluate(prompt, previous_content, eval_context)
        
        # Store evaluation in state for orchestrator
        if state:
            state['evaluation'] = result
        
        return result.detailed_feedback