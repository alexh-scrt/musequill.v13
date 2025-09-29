# Implementation Instructions: Quality Evaluator Agent for Musequill v13

## Overview
Create an `EvaluatorAgent` that assesses scientific content quality using 10 quantitative metrics, provides structured feedback, and integrates with the existing generator-discriminator workflow for iterative refinement.

## File Structure

```
src/agents/evaluator.py          # Main evaluator agent
src/agents/metrics/              # Metrics calculation modules
├── __init__.py
├── novelty.py                   # CNR calculation
├── claims.py                    # Claim density
├── rigor.py                     # Mathematical rigor
├── compression.py               # Semantic compression
├── citations.py                 # Citation analysis
├── empirical.py                 # Empirical grounding
├── structure.py                 # Structural coherence
├── notation.py                  # Notation consistency
├── figures.py                   # Figure/equation utility
└── parsimony.py                 # Parsimony score
src/agents/profiles.py           # Add evaluator profile
src/workflow/orchestrator.py    # Integrate evaluation loop
```

## Core Implementation

### 1. EvaluatorAgent Class (`src/agents/evaluator.py`)

```python
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
```

### 2. Metrics Modules

Create `src/agents/metrics/__init__.py`:

```python
from .novelty import calculate_cnr
from .claims import calculate_claim_density
from .rigor import calculate_mathematical_rigor
from .compression import calculate_compression_ratio
from .citations import calculate_citation_metrics
from .empirical import calculate_empirical_grounding
from .structure import calculate_structural_coherence
from .notation import calculate_notation_consistency
from .figures import calculate_figure_utility
from .parsimony import calculate_parsimony

__all__ = [
    'calculate_cnr',
    'calculate_claim_density',
    'calculate_mathematical_rigor',
    'calculate_compression_ratio',
    'calculate_citation_metrics',
    'calculate_empirical_grounding',
    'calculate_structural_coherence',
    'calculate_notation_consistency',
    'calculate_figure_utility',
    'calculate_parsimony'
]
```

Example metric implementation (`src/agents/metrics/novelty.py`):

```python
import re
from typing import Optional, Dict, Set, Any
from collections import Counter


def calculate_cnr(
    content: str,
    previous_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate Conceptual Novelty Rate
    
    Returns dict with:
        - percentage: CNR as percentage (0-100)
        - score: Normalized score for metric system
        - new_concepts: Set of new concepts
        - total_concepts: Set of all concepts
        - details: Additional analysis
    """
    # Extract concepts from current content
    current_concepts = extract_concepts(content)
    
    # Extract concepts from previous content
    if previous_content:
        previous_concepts = extract_concepts(previous_content)
    else:
        previous_concepts = set()
    
    # Calculate novelty
    new_concepts = current_concepts - previous_concepts
    total_concepts = current_concepts
    
    if not total_concepts:
        cnr_percentage = 0.0
    else:
        cnr_percentage = (len(new_concepts) / len(total_concepts)) * 100
    
    # Calculate score (normalized to 15 points max)
    if cnr_percentage >= 40:
        score = min(cnr_percentage / 40 * 15, 15)
    elif cnr_percentage >= 25:
        score = 6 + (cnr_percentage - 25) / 15 * 6
    else:
        score = cnr_percentage / 25 * 6
    
    return {
        'percentage': cnr_percentage,
        'score': score,
        'new_concepts': list(new_concepts)[:20],  # Sample
        'total_concepts': len(total_concepts),
        'new_count': len(new_concepts),
        'details': {
            'has_previous': previous_content is not None,
            'assessment': _assess_novelty(cnr_percentage)
        }
    }


def extract_concepts(text: str) -> Set[str]:
    """
    Extract technical concepts from text
    
    Concepts include:
    - Mathematical terms
    - Technical terminology
    - Defined terms
    - Named theorems/lemmas
    - Novel compounds (e.g., "quantum-thermodynamic")
    """
    concepts = set()
    
    # Extract capitalized technical terms
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    concepts.update(capitalized)
    
    # Extract mathematical symbols/terms
    math_terms = re.findall(r'\$[^$]+\$|\\[a-zA-Z]+', text)
    concepts.update(math_terms)
    
    # Extract hyphenated compounds
    compounds = re.findall(r'\b[a-z]+-[a-z]+(?:-[a-z]+)*\b', text.lower())
    concepts.update(compounds)
    
    # Extract theorem/lemma references
    theorems = re.findall(
        r'(?:Theorem|Lemma|Proposition|Corollary)\s+\d+',
        text,
        re.IGNORECASE
    )
    concepts.update(theorems)
    
    # Extract defined terms (simple heuristic)
    definitions = re.findall(
        r'(?:define|called|termed|known as)\s+["\']?([^"\',.]+)["\']?',
        text,
        re.IGNORECASE
    )
    concepts.update(definitions)
    
    # Filter out common words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    concepts = {c for c in concepts if c.lower() not in stopwords}
    
    return concepts


def _assess_novelty(percentage: float) -> str:
    """Provide qualitative assessment of novelty"""
    if percentage >= 60:
        return "Excellent: Highly novel content"
    elif percentage >= 40:
        return "Good: Sufficient novelty"
    elif percentage >= 25:
        return "Acceptable: Minimum novelty threshold"
    elif percentage >= 15:
        return "Poor: Excessive redundancy"
    else:
        return "Critical: Unacceptable redundancy"
```

### 3. Integration with Orchestrator

Modify `src/workflow/orchestrator.py` to add evaluation loop:

```python
# Add to WorkflowOrchestrator class

from src.agents.evaluator import EvaluatorAgent, EvaluationResult

class WorkflowOrchestrator:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add evaluator
        self.evaluator = EvaluatorAgent(
            session_id=self.session_id,
            llm_params={'temperature': 0.3}  # Lower temp for consistency
        )
        
        # Evaluation settings
        self.enable_evaluation = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"
        self.max_refinement_iterations = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "3"))
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "60.0"))
    
    async def _generator_node(self, state: OrchestratorState) -> dict:
        """Enhanced generator with evaluation feedback"""
        logger.info("📋  GENERATOR")
        
        # Get evaluation feedback if available
        evaluation = state.get("evaluation")
        feedback_context = ""
        
        if evaluation and evaluation.priority_actions:
            feedback_context = "\n\nQUALITY FEEDBACK - Address these issues:\n"
            for action in evaluation.priority_actions[:5]:
                feedback_context += f"- {action}\n"
        
        # Generate content with feedback
        response = await self.generator.process(
            state.get("topic", ""),
            context=feedback_context
        )
        
        # Evaluate if enabled
        if self.enable_evaluation:
            eval_result = await self.evaluator.evaluate(
                content=response,
                previous_content=state.get("current_content"),
                context={}
            )
            
            logger.info(
                f"Quality score: {eval_result.total_score:.1f}/100 ({eval_result.tier})"
            )
            
            # Check if refinement needed
            refinement_count = state.get("refinement_iteration", 0)
            
            if (not eval_result.pass_threshold and 
                refinement_count < self.max_refinement_iterations):
                
                logger.info(f"Quality below threshold, triggering refinement {refinement_count + 1}")
                
                return {
                    "current_content": response,
                    "evaluation": eval_result,
                    "refinement_iteration": refinement_count + 1,
                    "needs_refinement": True,
                    "messages": [AIMessage(content=eval_result.detailed_feedback, name="evaluator")]
                }
            
            # Quality sufficient or max iterations reached
            return {
                "current_content": response,
                "evaluation": eval_result,
                "needs_refinement": False,
                "messages": [AIMessage(content=response, name="generator")]
            }
        
        # No evaluation - proceed normally
        return {
            "current_content": response,
            "messages": [AIMessage(content=response, name="generator")]
        }
    
    def _build_graph(self):
        """Enhanced graph with evaluation loop"""
        workflow = StateGraph(OrchestratorState)
        
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("discriminator", self._discriminator_node)
        
        # Add conditional edge for refinement
        workflow.add_edge(START, "generator")
        
        workflow.add_conditional_edges(
            "generator",
            lambda s: "generator" if s.get("needs_refinement") else "discriminator"
        )
        
        workflow.add_conditional_edges(
            "discriminator",
            lambda s: END if s.get("END") else "generator"
        )
        
        # Compile
        memory = MemorySaver()
        self.compiled_graph = workflow.compile(checkpointer=memory)
```

### 4. Configuration

Add to `.env`:

```bash
# Evaluation settings
ENABLE_EVALUATION=true
QUALITY_THRESHOLD=60.0
MAX_REFINEMENT_ITERATIONS=3

# Metric-specific thresholds (optional overrides)
CNR_THRESHOLD=25.0
SCR_THRESHOLD=5.0
MRI_THRESHOLD=80.0
```

### 5. Testing

Create `tests/test_evaluator.py`:

```python
import pytest
from src.agents.evaluator import EvaluatorAgent
from src.agents.metrics import calculate_cnr


@pytest.mark.asyncio
async def test_evaluator_initialization():
    """Test evaluator initializes correctly"""
    evaluator = EvaluatorAgent(session_id="test_session")
    assert evaluator.agent_id == "evaluator"
    assert evaluator.session_id == "test_session"


@pytest.mark.asyncio
async def test_cnr_calculation():
    """Test conceptual novelty rate calculation"""
    content = "Quantum entanglement is a phenomenon. Von Neumann entropy is conserved."
    previous = "Von Neumann entropy is conserved."
    
    result = calculate_cnr(content, previous)
    
    assert 'percentage' in result
    assert 'score' in result
    assert result['percentage'] > 0  # Should have some novelty
    assert result['new_count'] >= 1  # At least "Quantum entanglement"


@pytest.mark.asyncio
async def test_evaluation_with_poor_content():
    """Test evaluation correctly identifies poor quality"""
    evaluator = EvaluatorAgent()
    
    # Highly redundant content
    poor_content = "Information is conserved. " * 100
    
    result = await evaluator.evaluate(poor_content)
    
    assert result.total_score < 60  # Should be below threshold
    assert not result.pass_threshold
    assert len(result.critical_failures) > 0
    assert 'semantic_compression' in [f.split(':')[0] for f in result.critical_failures]


@pytest.mark.asyncio
async def test_evaluation_with_good_content():
    """Test evaluation correctly identifies good quality"""
    evaluator = EvaluatorAgent()
    
    # Diverse, well-structured content
    good_content = """
    Theorem 1: Information is conserved in unitary evolution.
    Proof: By Liouville's theorem, phase-space volume is preserved...
    
    This result generalizes previous work [1,2,3] by extending to quantum systems.
    
    We predict that finite-size environments will show correlation decay...
    
    Corollary: Mutual information I(A:B) satisfies dI/dt = 0.
    """
    
    result = await evaluator.evaluate(good_content)
    
    assert result.total_score > 40  # Should have reasonable score
    # Note: Won't score high without actual proofs, citations, etc.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 6. Logging and Monitoring

Add evaluation logging to track quality over time:

```python
# In orchestrator after evaluation
if eval_result:
    self._log_queue.put_nowait({
        "agent_id": "evaluator",
        "content": json.dumps({
            "iteration": state.get("iteration", 0),
            "score": eval_result.total_score,
            "tier": eval_result.tier,
            "metrics": {
                name: {
                    "score": m.score,
                    "threshold_met": m.threshold_met
                }
                for name, m in eval_result.metrics.items()
            }
        }, indent=2)
    })
```

## Usage Instructions

### Running with Evaluation

```bash
# Enable evaluation (default)
ENABLE_EVALUATION=true python main.py

# Disable evaluation
ENABLE_EVALUATION=false python main.py

# Adjust quality threshold
QUALITY_THRESHOLD=75.0 python main.py

# Increase refinement iterations
MAX_REFINEMENT_ITERATIONS=5 python main.py
```

### Client Usage

The evaluation happens transparently during generation:

```bash
python client.py "Explain quantum information theory"

# Output will show:
# Quality score: 45.2/100 (Poor - Significant Revision Needed)
# Quality below threshold, triggering refinement 1
# Quality score: 68.3/100 (Acceptable)
```

### Interpreting Results

The evaluator provides:
- **Total score** (0-100): Overall quality assessment
- **Tier**: Unacceptable, Poor, Acceptable, Good, Excellent
- **Metric breakdown**: Individual scores for each metric
- **Critical failures**: Auto-reject issues
- **Priority actions**: Specific improvements needed
- **Section analysis**: Which sections to keep/merge/delete

## Key Implementation Notes

1. **Start simple**: Implement basic heuristics first, refine later
2. **Log everything**: Track quality evolution across iterations
3. **Fail fast**: Critical failures should stop generation immediately
4. **Be specific**: Feedback must be actionable, not vague
5. **Iterate**: The evaluator itself should improve over time

This implementation provides a robust foundation for quality assessment while remaining modular and extensible for future enhancements.