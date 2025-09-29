import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from src.agents.base import BaseAgent
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


class EvaluatorAgent(BaseAgent):
    """Agent that evaluates scientific content quality"""
    
    # Metric weights and thresholds
    METRIC_CONFIG = {
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
            'weight': 15,
            'min_threshold': 9,
            'optimal': 15,
            'calculation': '80%+ proofs, 70%+ citations'
        },
        'semantic_compression': {
            'weight': 15,
            'min_threshold': 7,
            'optimal': 15,
            'calculation': 'SCR <= 5:1'
        },
        'citation_context': {
            'weight': 10,
            'min_threshold': 5,
            'optimal': 10,
            'calculation': '5-40 cites/10pg + novelty statement'
        },
        'empirical_grounding': {
            'weight': 10,
            'min_threshold': 4,
            'optimal': 10,
            'calculation': '1+ testable prediction'
        },
        'structural_coherence': {
            'weight': 10,
            'min_threshold': 7,
            'optimal': 10,
            'calculation': 'No circular reasoning'
        },
        'notation_consistency': {
            'weight': 5,
            'min_threshold': 4,
            'optimal': 5,
            'calculation': '95%+ consistency'
        },
        'figure_utility': {
            'weight': 5,
            'min_threshold': 3,
            'optimal': 5,
            'calculation': '80%+ informative'
        },
        'parsimony': {
            'weight': 5,
            'min_threshold': 3,
            'optimal': 5,
            'calculation': '70%+ essential assumptions'
        }
    }
    
    # Tier thresholds
    TIER_THRESHOLDS = {
        'excellent': 90,
        'good': 75,
        'acceptable': 60,
        'poor': 40,
        'unacceptable': 0
    }
    
    # Critical failure metrics (auto-reject if below threshold)
    CRITICAL_METRICS = [
        'mathematical_rigor',
        'semantic_compression',
        'conceptual_novelty'
    ]
    
    def __init__(
        self,
        agent_id: str = "evaluator",
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        llm_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id,
            web_search=False,  # Evaluator doesn't need web search
            model=model,
            session_id=session_id,
            llm_params=llm_params
        )
        logger.info(f"EvaluatorAgent initialized with model: {self.model}")
    
    async def evaluate(
        self,
        content: str,
        previous_content: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate content quality across all metrics
        
        Args:
            content: The content to evaluate
            previous_content: Previous iteration for novelty comparison
            context: Additional context (references, figures, etc.)
            
        Returns:
            EvaluationResult with scores and feedback
        """
        logger.info("Starting content evaluation...")
        
        # Calculate all metrics
        metrics = {}
        
        # 1. Conceptual Novelty Rate
        cnr_data = calculate_cnr(content, previous_content)
        metrics['conceptual_novelty'] = self._score_metric(
            'conceptual_novelty',
            cnr_data['score'],
            cnr_data
        )
        
        # 2. Claim Density
        cd_data = calculate_claim_density(content)
        metrics['claim_density'] = self._score_metric(
            'claim_density',
            cd_data['score'],
            cd_data
        )
        
        # 3. Mathematical Rigor
        mr_data = calculate_mathematical_rigor(content)
        metrics['mathematical_rigor'] = self._score_metric(
            'mathematical_rigor',
            mr_data['score'],
            mr_data
        )
        
        # 4. Semantic Compression Ratio
        scr_data = calculate_compression_ratio(content, previous_content)
        metrics['semantic_compression'] = self._score_metric(
            'semantic_compression',
            scr_data['score'],
            scr_data
        )
        
        # 5. Citation & Context
        citation_data = calculate_citation_metrics(content, context)
        metrics['citation_context'] = self._score_metric(
            'citation_context',
            citation_data['score'],
            citation_data
        )
        
        # 6. Empirical Grounding
        eg_data = calculate_empirical_grounding(content)
        metrics['empirical_grounding'] = self._score_metric(
            'empirical_grounding',
            eg_data['score'],
            eg_data
        )
        
        # 7. Structural Coherence
        sc_data = calculate_structural_coherence(content)
        metrics['structural_coherence'] = self._score_metric(
            'structural_coherence',
            sc_data['score'],
            sc_data
        )
        
        # 8. Notation Consistency
        nc_data = calculate_notation_consistency(content)
        metrics['notation_consistency'] = self._score_metric(
            'notation_consistency',
            nc_data['score'],
            nc_data
        )
        
        # 9. Figure/Equation Utility
        fe_data = calculate_figure_utility(content, context)
        metrics['figure_utility'] = self._score_metric(
            'figure_utility',
            fe_data['score'],
            fe_data
        )
        
        # 10. Parsimony
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
        
        # Check if passes threshold
        pass_threshold = (
            total_score >= self.TIER_THRESHOLDS['acceptable'] and
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
            detailed_feedback=detailed_feedback
        )
        
        logger.info(f"Evaluation complete: {total_score:.1f}/100 ({tier})")
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
        
        percentage = (normalized / max_points) * 100
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
            if not metric.threshold_met:
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
        # Simple split on major headers or paragraphs
        # TODO: Implement more sophisticated section detection
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
        
        # Overall assessment
        feedback_parts.append(f"OVERALL SCORE: {total_score:.1f}/100 ({tier})")
        feedback_parts.append("")
        
        # Critical failures
        if critical_failures:
            feedback_parts.append("CRITICAL FAILURES (Auto-reject):")
            for failure in critical_failures:
                feedback_parts.append(f"  ✗ {failure}")
            feedback_parts.append("")
        
        # Metric breakdown
        feedback_parts.append("METRIC BREAKDOWN:")
        for name, metric in metrics.items():
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
        if total_score >= 75:
            feedback_parts.append("RECOMMENDATION: Accept with minor revisions")
        elif total_score >= 60:
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