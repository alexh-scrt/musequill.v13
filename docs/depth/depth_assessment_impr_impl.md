# Depth Assessment & Advancement: Implementation Plan

## Overview

This plan guides the implementation of intelligent depth progression in the generator-discriminator conversation system. The feature will enable conversations to systematically advance through 5 depth levels based on coverage quality and content metrics.

---

## Phase 1: Foundation - Data Structures & Configuration

### Task 1.1: Create Depth Assessment Data Structures

**File**: `src/agents/depth_assessment.py` (NEW)

**Implementation**:

```python
"""
Data structures for depth assessment and advancement decisions.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class DepthDecision(Enum):
    """Possible depth advancement decisions"""
    ADVANCE = "advance"                    # Coverage + quality met, advance
    CONTINUE_DEPTH = "continue_depth"      # Stay at current depth
    REFINE_THEN_ADVANCE = "refine_then_advance"  # Fix quality, then advance
    FORCE_ADVANCE = "force_advance"        # Saturation detected, force advance


@dataclass
class DepthCoverage:
    """Coverage breakdown for a specific depth level"""
    item_scores: Dict[str, float]  # Each coverage item (0-20 points)
    total_score: float             # Sum of item_scores (0-100)
    critical_gaps: List[str]       # Items scored < 15
    evidence_examples: List[str]   # Quotes showing coverage


@dataclass
class DepthSignals:
    """Advancement and anti-signals detected in conversation"""
    advancement_signals: List[str]      # Signals ready for next depth
    anti_signals: List[str]             # Signals need to stay
    advancement_count: int
    anti_count: int
    net_signal: int                     # advancement_count - anti_count


@dataclass
class DepthAssessment:
    """Complete assessment of depth level coverage"""
    current_depth: int
    coverage: DepthCoverage
    signals: DepthSignals
    quality_metrics: Dict[str, float]   # CNR, SCR, etc.
    advancement_ready: bool
    decision: DepthDecision
    recommendation: str                 # Human-readable explanation
    confidence: float                   # 0-1 confidence in decision
    
    @property
    def should_advance(self) -> bool:
        """Whether to advance to next depth"""
        return self.decision in [DepthDecision.ADVANCE, DepthDecision.FORCE_ADVANCE]


@dataclass
class IntegratedAssessment:
    """Combined quality + depth assessment"""
    depth_assessment: DepthAssessment
    quality_score: float
    quality_tier: str
    decision: DepthDecision
    reason: str
    recommended_action: str
    metrics_summary: Dict[str, Any]
```

**Acceptance Criteria**:
- [ ] All dataclasses defined with type hints
- [ ] Enum for depth decisions created
- [ ] Helper properties implemented (e.g., `should_advance`)
- [ ] Docstrings complete

---

### Task 1.2: Configuration System for Thresholds

**File**: `src/config/depth_thresholds.py` (NEW)

**Implementation**:

```python
"""
Configuration for depth assessment thresholds.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DepthThresholds:
    """Configurable thresholds for depth advancement"""
    
    # Coverage score thresholds (0-100)
    coverage_minimum: float = 75.0
    coverage_excellent: float = 90.0
    coverage_acceptable: float = 60.0
    
    # Signal thresholds
    net_signal_minimum: int = 1
    advancement_signals_min: int = 2
    
    # Iteration constraints
    max_iterations_per_depth: int = 6
    min_iterations_per_depth: int = 2
    
    # Quality integration
    quality_score_minimum: float = 70.0
    cnr_minimum: float = 20.0
    scr_minimum: float = 70.0
    
    # Depth-specific overrides
    depth_specific: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        1: {'coverage_minimum': 80.0, 'min_iterations': 2},
        2: {'coverage_minimum': 75.0, 'min_iterations': 2},
        3: {'coverage_minimum': 70.0, 'min_iterations': 2},
        4: {'coverage_minimum': 75.0, 'min_iterations': 2},
        5: {'coverage_minimum': 65.0, 'min_iterations': 1}
    })
    
    @classmethod
    def from_env(cls) -> 'DepthThresholds':
        """Load thresholds from environment variables"""
        return cls(
            coverage_minimum=float(os.getenv('DEPTH_COVERAGE_MINIMUM', '75.0')),
            coverage_excellent=float(os.getenv('DEPTH_COVERAGE_EXCELLENT', '90.0')),
            coverage_acceptable=float(os.getenv('DEPTH_COVERAGE_ACCEPTABLE', '60.0')),
            quality_score_minimum=float(os.getenv('DEPTH_QUALITY_MINIMUM', '70.0')),
            cnr_minimum=float(os.getenv('DEPTH_CNR_MINIMUM', '20.0')),
            scr_minimum=float(os.getenv('DEPTH_SCR_MINIMUM', '70.0')),
            max_iterations_per_depth=int(os.getenv('MAX_ITERATIONS_PER_DEPTH', '6')),
            min_iterations_per_depth=int(os.getenv('MIN_ITERATIONS_PER_DEPTH', '2'))
        )
    
    def get_threshold(self, depth: int, key: str) -> float:
        """Get depth-specific threshold or fall back to default"""
        if depth in self.depth_specific and key in self.depth_specific[depth]:
            return self.depth_specific[depth][key]
        return getattr(self, key, self.coverage_minimum)
```

**Environment Variables** (add to `.env`):
```bash
# Depth coverage thresholds
DEPTH_COVERAGE_MINIMUM=75.0
DEPTH_COVERAGE_EXCELLENT=90.0
DEPTH_COVERAGE_ACCEPTABLE=60.0

# Quality gates
DEPTH_QUALITY_MINIMUM=70.0
DEPTH_CNR_MINIMUM=20.0
DEPTH_SCR_MINIMUM=70.0

# Iteration constraints
MAX_ITERATIONS_PER_DEPTH=6
MIN_ITERATIONS_PER_DEPTH=2

# Depth-specific overrides (optional)
DEPTH_1_COVERAGE_MIN=80.0
DEPTH_3_COVERAGE_MIN=70.0
DEPTH_5_COVERAGE_MIN=65.0
```

**Acceptance Criteria**:
- [ ] Thresholds class with defaults
- [ ] Environment variable loading
- [ ] Depth-specific override system
- [ ] Getter method for depth-specific values

---

### Task 1.3: Depth Criteria Definitions

**File**: `src/config/depth_criteria.py` (NEW)

**Implementation**:

```python
"""
Criteria definitions for each depth level.
"""
from typing import Dict, List


class DepthCriteria:
    """Criteria for assessing coverage at each depth level"""
    
    @staticmethod
    def get_criteria(depth: int) -> Dict[str, any]:
        """Get assessment criteria for specific depth level"""
        
        criteria_map = {
            1: {
                'name': 'Foundations',
                'focus': 'Core definitions, basic concepts, fundamental understanding',
                'coverage_items': [
                    'core_definition',
                    'key_components', 
                    'basic_scope',
                    'fundamental_properties',
                    'terminology'
                ],
                'coverage_descriptions': {
                    'core_definition': 'Is the topic clearly defined?',
                    'key_components': 'Are main parts/elements identified?',
                    'basic_scope': 'Is it clear what falls inside vs. outside?',
                    'fundamental_properties': 'Are essential characteristics explained?',
                    'terminology': 'Are key terms introduced and defined?'
                },
                'quality_indicators': [
                    'Clarity: No ambiguity in core concepts',
                    'Consistency: Terms used consistently',
                    'Shared Understanding: Both agents demonstrate same baseline',
                    'Precision: Specific definitions, not vague generalizations'
                ],
                'advancement_signals': [
                    'Questions shift from "What is X?" to "How does X work?"',
                    'Definitions no longer questioned',
                    'Both agents reference fundamentals without re-explaining',
                    'Conversation moves toward mechanisms/processes',
                    'Repetition of basic concepts (saturation)'
                ],
                'anti_signals': [
                    'Still defining basic terms',
                    'Confusion about core concepts',
                    'Inconsistent terminology',
                    '"Wait, what exactly is X?" questions',
                    'Foundational disagreements'
                ]
            },
            2: {
                'name': 'Mechanisms',
                'focus': 'Underlying principles, mechanisms, how things work',
                'coverage_items': [
                    'process_flow',
                    'causal_relationships',
                    'internal_dynamics',
                    'underlying_principles',
                    'operational_logic'
                ],
                'coverage_descriptions': {
                    'process_flow': 'Step-by-step explanation of how it works',
                    'causal_relationships': 'What causes what? X → Y → Z chains',
                    'internal_dynamics': 'How components interact',
                    'underlying_principles': 'Why does it work this way?',
                    'operational_logic': 'Rules/constraints governing behavior'
                },
                'quality_indicators': [
                    'Logical Coherence: Clear cause-effect chains',
                    'Systematic Coverage: Connected understanding',
                    'Depth: Beyond surface "what" to deeper "why"',
                    'Precision: Specific mechanisms, not hand-waving'
                ],
                'advancement_signals': [
                    'Questions shift to "Where is this used?" or "What are examples?"',
                    'Mechanisms explained and understood',
                    'Discussion moves toward applications',
                    'Interest in real-world implementations',
                    '"I understand how, but where do we see this?"'
                ],
                'anti_signals': [
                    'Still asking "How does this work?"',
                    'Confusion about process steps',
                    'Causal relationships unclear',
                    'Requests for more mechanism explanation'
                ]
            },
            3: {
                'name': 'Applications',
                'focus': 'Real-world applications, examples, practical implications',
                'coverage_items': [
                    'concrete_examples',
                    'domain_diversity',
                    'implementation_details',
                    'practical_constraints',
                    'theory_practice_bridge'
                ],
                'coverage_descriptions': {
                    'concrete_examples': 'At least 3-5 real-world examples/use cases',
                    'domain_diversity': 'Examples from different fields/contexts',
                    'implementation_details': 'How is it actually applied?',
                    'practical_constraints': 'Real-world limitations, trade-offs',
                    'theory_practice_bridge': 'Connection between principles and applications'
                },
                'quality_indicators': [
                    'Specificity: Named examples, not generic',
                    'Concreteness: Detailed enough to visualize',
                    'Diversity: Multiple domains covered',
                    'Relevance: Examples clearly illustrate concepts'
                ],
                'advancement_signals': [
                    'Questions about limitations: "What doesn\'t work?"',
                    'Interest in edge cases: "What about scenario X?"',
                    'Critical perspective: "What are the problems?"',
                    'Boundary probing: "How far can we push this?"',
                    'Trade-off discussions emerging'
                ],
                'anti_signals': [
                    'Still asking "Where is this used?"',
                    'Requesting more examples',
                    'Examples too generic or vague',
                    'Missing application domains'
                ]
            },
            4: {
                'name': 'Edge Cases',
                'focus': 'Challenges, limitations, controversies, edge cases',
                'coverage_items': [
                    'known_problems',
                    'failure_modes',
                    'trade_offs',
                    'controversies',
                    'limitations'
                ],
                'coverage_descriptions': {
                    'known_problems': 'Documented issues and challenges',
                    'failure_modes': 'When and how it breaks down',
                    'trade_offs': 'Compromises and constraints',
                    'controversies': 'Debates and disagreements',
                    'limitations': 'Boundaries and scope restrictions'
                },
                'quality_indicators': [
                    'Balanced View: Not just benefits',
                    'Multiple Perspectives: Different viewpoints considered',
                    'Nuanced Understanding: Complexity acknowledged',
                    'Critical Thinking: Analytical depth'
                ],
                'advancement_signals': [
                    'Limitations thoroughly understood',
                    'Questions turn forward-looking',
                    'Interest in future developments',
                    '"What\'s next?" or "What\'s unsolved?" questions',
                    'Speculative discussions emerging'
                ],
                'anti_signals': [
                    'Still identifying limitations',
                    'Asking "What are the problems?"',
                    'Edge cases not yet explored',
                    'Requesting more critical analysis'
                ]
            },
            5: {
                'name': 'Future Directions',
                'focus': 'Cutting-edge developments, open questions, speculation',
                'coverage_items': [
                    'emerging_trends',
                    'research_frontiers',
                    'open_questions',
                    'speculative_possibilities',
                    'future_implications'
                ],
                'coverage_descriptions': {
                    'emerging_trends': 'Current developments and directions',
                    'research_frontiers': 'Cutting-edge work being done',
                    'open_questions': 'Unsolved problems and mysteries',
                    'speculative_possibilities': 'Informed speculation about future',
                    'future_implications': 'Potential long-term impacts'
                },
                'quality_indicators': [
                    'Forward-Looking: Future-oriented perspective',
                    'Informed Speculation: Grounded in current knowledge',
                    'Acknowledges Uncertainty: Clear about speculation',
                    'Connected: Links to established knowledge'
                ],
                'advancement_signals': [
                    'Topic feels exhausted',
                    'Circular reasoning or repetition',
                    'Both agents agree on natural closure',
                    'Summary-oriented discussion',
                    'Meta-discussion about conversation itself'
                ],
                'anti_signals': [
                    'Still exploring specific futures',
                    'New research directions emerging',
                    'Fresh speculative angles appearing'
                ]
            }
        }
        
        return criteria_map.get(depth, criteria_map[5])
    
    @staticmethod
    def get_all_depth_names() -> List[str]:
        """Get names of all depth levels"""
        return [
            'Foundations',
            'Mechanisms', 
            'Applications',
            'Edge Cases',
            'Future Directions'
        ]
```

**Acceptance Criteria**:
- [ ] Criteria defined for all 5 depth levels
- [ ] Coverage items list for each level
- [ ] Quality indicators documented
- [ ] Advancement and anti-signals specified

---

## Phase 2: Core Assessment Logic

### Task 2.1: Depth Assessment Prompts

**File**: `src/prompts/depth_prompts.py` (NEW)

**Implementation**:

```python
"""
Prompts for depth coverage assessment.
"""
from typing import List, Dict
from src.config.depth_criteria import DepthCriteria


DEPTH_ASSESSMENT_SYSTEM_PROMPT = """
You are a depth coverage analyst specializing in evaluating conversational exploration depth.

Your role is to objectively assess whether a conversation has sufficiently explored the current 
depth level before advancing to deeper analysis.

Key Principles:
1. **Evidence-Based**: Ground all assessments in specific conversation content
2. **Criteria-Driven**: Apply depth-specific criteria consistently
3. **Progressive**: Recognize natural progression signals in discourse
4. **Balanced**: Consider both coverage breadth AND quality depth
5. **Honest**: Identify gaps even if coverage seems subjectively "good"

You output structured JSON assessments that drive depth advancement decisions.
"""


def build_depth_assessment_prompt(
    conversation_text: str,
    current_depth: int,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> str:
    """
    Build depth-specific assessment prompt.
    
    Args:
        conversation_text: Recent conversation turns formatted
        current_depth: Current depth level (1-5)
        aspects_explored: List of aspects covered so far
        topic_summary: Running summary of conversation
        original_topic: Original user question/topic
    
    Returns:
        Complete assessment prompt for LLM
    """
    
    criteria = DepthCriteria.get_criteria(current_depth)
    
    # Build coverage items section
    coverage_items_text = "\n".join([
        f"{i+1}. **{item.replace('_', ' ').title()}**: {criteria['coverage_descriptions'][item]}"
        for i, item in enumerate(criteria['coverage_items'])
    ])
    
    # Build quality indicators
    quality_text = "\n".join([f"- {indicator}" for indicator in criteria['quality_indicators']])
    
    # Build signals
    advancement_text = "\n".join([f"- {signal}" for signal in criteria['advancement_signals']])
    anti_text = "\n".join([f"- {signal}" for signal in criteria['anti_signals']])
    
    prompt = f"""
DEPTH LEVEL {current_depth} ASSESSMENT: {criteria['name']}
Topic: "{original_topic}"

CONVERSATION (Last 6 turns):
{conversation_text}

ASPECTS TRACKED: {', '.join(aspects_explored) if aspects_explored else 'None'}
SUMMARY: {topic_summary if topic_summary else 'No summary yet'}

---

ASSESSMENT CRITERIA FOR LEVEL {current_depth} ({criteria['name']}):
Focus: {criteria['focus']}

**Required Coverage** (Must address ALL to score 75+):
{coverage_items_text}

**Quality Indicators**:
{quality_text}

**Advancement Signals** (Indicates readiness for Level {current_depth + 1}):
{advancement_text}

**Anti-Signals** (Must stay at Level {current_depth}):
{anti_text}

---

ANALYSIS TASK:

1. **Coverage Score** (0-100):
   Score each Required Coverage item (0-20 points each):
   - 0 points: Not addressed at all
   - 5 points: Mentioned superficially
   - 10 points: Partially explained
   - 15 points: Well explained with examples
   - 20 points: Thoroughly covered, clear understanding
   
   Total Coverage Score = Sum of {len(criteria['coverage_items'])} items (0-100)

2. **Quality Assessment**:
   - Count Quality Indicators present
   - Note any quality issues

3. **Signal Analysis**:
   - Count Advancement Signals present (0-{len(criteria['advancement_signals'])})
   - Count Anti-Signals present (0-{len(criteria['anti_signals'])})
   - Calculate Net Signal = Advancement - Anti

4. **Gap Identification**:
   - List specific Required Coverage items scored < 15
   - Identify missing Quality Indicators

5. **Advancement Decision**:
   - ADVANCE if: Coverage Score ≥ 75 AND Net Signal > 0 AND no critical gaps
   - STAY if: Coverage Score < 75 OR Net Signal ≤ 0 OR critical gaps exist

---

OUTPUT FORMAT (JSON):
{{
    "coverage_breakdown": {{
        "{criteria['coverage_items'][0]}": <0-20>,
        "{criteria['coverage_items'][1]}": <0-20>,
        "{criteria['coverage_items'][2]}": <0-20>,
        "{criteria['coverage_items'][3]}": <0-20>,
        "{criteria['coverage_items'][4]}": <0-20>
    }},
    "coverage_score": <sum of above, 0-100>,
    "quality_indicators_present": [<list of indicators found>],
    "advancement_signals_count": <number>,
    "anti_signals_count": <number>,
    "net_signal": <advancement_count - anti_count>,
    "critical_gaps": [<list of items scored < 15>],
    "advancement_ready": <true/false>,
    "recommendation": "<2-3 sentence explanation>",
    "confidence": <0.0-1.0>,
    "evidence": {{
        "coverage_examples": [<specific quotes showing coverage>],
        "signal_examples": [<specific quotes showing advancement signals>]
    }}
}}

Be precise and evidence-based. Quote specific parts of the conversation to support your assessment.
"""
    
    return prompt
```

**Acceptance Criteria**:
- [ ] System prompt defined
- [ ] Dynamic prompt builder for each depth level
- [ ] JSON output format specified
- [ ] Evidence requirements included

---

### Task 2.2: Depth Assessment Methods in EvaluatorAgent

**File**: `src/agents/evaluator.py` (MODIFY)

**Add these methods to existing `EvaluatorAgent` class**:

```python
# Add to imports at top
from src.agents.depth_assessment import (
    DepthAssessment, DepthCoverage, DepthSignals, DepthDecision
)
from src.config.depth_thresholds import DepthThresholds
from src.config.depth_criteria import DepthCriteria
from src.prompts.depth_prompts import (
    build_depth_assessment_prompt,
    DEPTH_ASSESSMENT_SYSTEM_PROMPT
)
import json
import re


# Add to __init__
def __init__(self, ...existing params...):
    # ... existing code ...
    
    # Initialize depth assessment components
    self.depth_thresholds = DepthThresholds.from_env()
    logger.info(f"Depth thresholds loaded: min={self.depth_thresholds.coverage_minimum}")


# Add new method
async def assess_depth_coverage(
    self,
    conversation_history: List[Dict],
    current_depth: int,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> DepthAssessment:
    """
    Assess whether current depth level is sufficiently explored.
    
    Args:
        conversation_history: Recent conversation turns
        current_depth: Current depth level (1-5)
        aspects_explored: List of aspects covered so far
        topic_summary: Running summary of topic
        original_topic: Original question/topic
    
    Returns:
        DepthAssessment with coverage scores and advancement decision
    """
    
    logger.info(f"🎯 Assessing depth {current_depth} coverage...")
    
    # Format conversation for assessment
    recent_turns = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
    conversation_text = self._format_conversation_for_depth_assessment(recent_turns)
    
    # Build assessment prompt
    prompt = build_depth_assessment_prompt(
        conversation_text=conversation_text,
        current_depth=current_depth,
        aspects_explored=aspects_explored,
        topic_summary=topic_summary,
        original_topic=original_topic
    )
    
    # Get LLM assessment
    response = await self.generate_with_llm(
        prompt,
        system_prompt=DEPTH_ASSESSMENT_SYSTEM_PROMPT
    )
    
    # Parse JSON response
    assessment_data = self._parse_depth_assessment_response(response)
    
    # Create DepthAssessment object
    coverage = DepthCoverage(
        item_scores=assessment_data['coverage_breakdown'],
        total_score=assessment_data['coverage_score'],
        critical_gaps=assessment_data['critical_gaps'],
        evidence_examples=assessment_data.get('evidence', {}).get('coverage_examples', [])
    )
    
    signals = DepthSignals(
        advancement_signals=assessment_data.get('evidence', {}).get('signal_examples', []),
        anti_signals=[],  # Could be extracted if needed
        advancement_count=assessment_data['advancement_signals_count'],
        anti_count=assessment_data['anti_signals_count'],
        net_signal=assessment_data['net_signal']
    )
    
    # Determine decision based on thresholds
    decision = self._determine_depth_decision(
        coverage_score=assessment_data['coverage_score'],
        net_signal=assessment_data['net_signal'],
        advancement_ready=assessment_data['advancement_ready'],
        current_depth=current_depth
    )
    
    return DepthAssessment(
        current_depth=current_depth,
        coverage=coverage,
        signals=signals,
        quality_metrics={},  # Will be filled by integrated assessment
        advancement_ready=assessment_data['advancement_ready'],
        decision=decision,
        recommendation=assessment_data['recommendation'],
        confidence=assessment_data.get('confidence', 0.8)
    )


def _format_conversation_for_depth_assessment(self, turns: List[Dict]) -> str:
    """Format conversation turns for depth assessment"""
    formatted = []
    for turn in turns:
        agent = turn.get('agent_id', 'unknown')
        content = turn.get('content', '')
        formatted.append(f"[{agent.upper()}]: {content}\n")
    return "\n".join(formatted)


def _parse_depth_assessment_response(self, response: str) -> Dict:
    """Parse JSON response from depth assessment LLM"""
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data
        else:
            logger.error("No JSON found in depth assessment response")
            return self._get_default_assessment()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse depth assessment JSON: {e}")
        return self._get_default_assessment()


def _get_default_assessment(self) -> Dict:
    """Return default assessment if parsing fails"""
    return {
        'coverage_breakdown': {},
        'coverage_score': 50.0,
        'quality_indicators_present': [],
        'advancement_signals_count': 0,
        'anti_signals_count': 0,
        'net_signal': 0,
        'critical_gaps': ['Assessment failed'],
        'advancement_ready': False,
        'recommendation': 'Continue at current depth (assessment error)',
        'confidence': 0.5,
        'evidence': {'coverage_examples': [], 'signal_examples': []}
    }


def _determine_depth_decision(
    self,
    coverage_score: float,
    net_signal: int,
    advancement_ready: bool,
    current_depth: int
) -> DepthDecision:
    """
    Determine depth decision based on assessment results.
    
    Decision logic:
    - ADVANCE: Coverage and signals meet thresholds
    - CONTINUE_DEPTH: Coverage or signals insufficient
    - FORCE_ADVANCE: Reserved for iteration-based override
    """
    
    min_coverage = self.depth_thresholds.get_threshold(current_depth, 'coverage_minimum')
    
    if advancement_ready and coverage_score >= min_coverage and net_signal > 0:
        return DepthDecision.ADVANCE
    else:
        return DepthDecision.CONTINUE_DEPTH
```

**Acceptance Criteria**:
- [ ] `assess_depth_coverage()` method implemented
- [ ] JSON parsing logic handles errors gracefully
- [ ] Decision logic uses configured thresholds
- [ ] Logging at key points
- [ ] Returns proper DepthAssessment object

---

## Phase 3: Quality-Depth Integration

### Task 3.1: Integrated Assessment Method

**File**: `src/agents/evaluator.py` (MODIFY - add to EvaluatorAgent)

```python
async def integrated_assessment(
    self,
    conversation_history: List[Dict],
    current_depth: int,
    current_content: str,
    previous_content: Optional[str],
    state: Dict[str, Any]
) -> 'IntegratedAssessment':
    """
    Perform combined quality + depth assessment.
    
    Returns unified decision that considers both:
    - Quality metrics (CNR, SCR, claim density, etc.)
    - Depth coverage (topic exploration completeness)
    
    Args:
        conversation_history: Recent turns
        current_depth: Current depth level
        current_content: Latest response to evaluate
        previous_content: Previous content for novelty comparison
        state: Conversation state with metadata
    
    Returns:
        IntegratedAssessment with unified decision
    """
    
    logger.info("📊 INTEGRATED ASSESSMENT (Quality + Depth)")
    
    # 1. Standard quality evaluation
    quality_eval = await self.evaluate(
        content=current_content,
        previous_content=previous_content,
        context=state.get('context', {})
    )
    
    logger.info(f"Quality: {quality_eval.total_score:.1f}/100 ({quality_eval.tier})")
    
    # 2. Depth coverage assessment
    depth_assessment = await self.assess_depth_coverage(
        conversation_history=conversation_history,
        current_depth=current_depth,
        aspects_explored=state.get('aspects_explored', []),
        topic_summary=state.get('topic_summary', ''),
        original_topic=state.get('original_topic', '')
    )
    
    logger.info(f"Depth Coverage: {depth_assessment.coverage.total_score:.1f}/100")
    
    # 3. Extract quality metrics
    cnr_metric = quality_eval.metrics.get('conceptual_novelty')
    scr_metric = quality_eval.metrics.get('semantic_compression')
    
    cnr_score = cnr_metric.score if cnr_metric else 0
    scr_score = scr_metric.score if scr_metric else 0
    
    # Fill in quality metrics in depth assessment
    depth_assessment.quality_metrics = {
        'cnr': cnr_score,
        'scr': scr_score,
        'quality_total': quality_eval.total_score
    }
    
    # 4. Make integrated decision
    integrated_decision = self._make_integrated_decision(
        quality_eval=quality_eval,
        depth_assessment=depth_assessment,
        current_depth=current_depth,
        cnr_score=cnr_score,
        scr_score=scr_score
    )
    
    logger.info(f"Decision: {integrated_decision.decision.value}")
    logger.info(f"Reason: {integrated_decision.reason}")
    
    return integrated_decision


def _make_integrated_decision(
    self,
    quality_eval: 'EvaluationResult',
    depth_assessment: DepthAssessment,
    current_depth: int,
    cnr_score: float,
    scr_score: float
) -> 'IntegratedAssessment':
    """
    Combine quality and depth assessments into unified decision.
    
    Decision Matrix:
    - Coverage ≥75 & Quality ≥70 & CNR ≥20 → ADVANCE
    - Coverage ≥75 & Quality <70 → REFINE_THEN_ADVANCE
    - Coverage <75 & Quality ≥70 → CONTINUE_DEPTH
    - Coverage ≥60 & CNR <15 → FORCE_ADVANCE (saturation)
    - Else → CONTINUE_DEPTH
    """
    
    from src.agents.depth_assessment import IntegratedAssessment
    
    # Thresholds
    coverage_score = depth_assessment.coverage.total_score
    quality_score = quality_eval.total_score
    
    min_coverage = self.depth_thresholds.get_threshold(current_depth, 'coverage_minimum')
    min_quality = self.depth_thresholds.quality_score_minimum
    min_cnr = self.depth_thresholds.cnr_minimum
    min_scr = self.depth_thresholds.scr_minimum
    
    # Quality gates
    quality_sufficient = (
        quality_score >= min_quality and
        cnr_score >= min_cnr and
        scr_score >= min_scr
    )
    
    # Coverage gate
    coverage_sufficient = coverage_score >= min_coverage
    
    # Saturation detection
    novelty_saturated = cnr_score < 15.0
    
    # Decision logic
    if coverage_sufficient and quality_sufficient:
        decision = DepthDecision.ADVANCE
        reason = f"Coverage ({coverage_score:.1f}) and quality ({quality_score:.1f}) both meet thresholds"
    
    elif coverage_sufficient and not quality_sufficient:
        decision = DepthDecision.REFINE_THEN_ADVANCE
        reason = f"Coverage sufficient ({coverage_score:.1f}) but quality needs improvement ({quality_score:.1f} < {min_quality})"
    
    elif not coverage_sufficient and quality_sufficient:
        decision = DepthDecision.CONTINUE_DEPTH
        reason = f"Quality good ({quality_score:.1f}) but coverage gaps remain ({coverage_score:.1f} < {min_coverage})"
    
    elif novelty_saturated and coverage_score >= self.depth_thresholds.coverage_acceptable:
        decision = DepthDecision.FORCE_ADVANCE
        reason = f"Low novelty (CNR={cnr_score:.1f}) indicates depth exhausted (diminishing returns)"
    
    else:
        decision = DepthDecision.CONTINUE_DEPTH
        reason = f"Both coverage ({coverage_score:.1f}) and quality ({quality_score:.1f}) need improvement"
    
    # Generate action recommendation
    action = self._get_action_recommendation(
        decision=decision,
        depth_assessment=depth_assessment,
        quality_eval=quality_eval,
        current_depth=current_depth
    )
    
    return IntegratedAssessment(
        depth_assessment=depth_assessment,
        quality_score=quality_score,
        quality_tier=quality_eval.tier,
        decision=decision,
        reason=reason,
        recommended_action=action,
        metrics_summary={
            'coverage_score': coverage_score,
            'quality_score': quality_score,
            'cnr': cnr_score,
            'scr': scr_score,
            'net_signal': depth_assessment.signals.net_signal,
            'critical_gaps': depth_assessment.coverage.critical_gaps
        }
    )


def _get_action_recommendation(
    self,
    decision: DepthDecision,
    depth_assessment: DepthAssessment,
    quality_eval: 'EvaluationResult',
    current_depth: int
) -> str:
    """Generate specific action recommendation based on decision"""
    
    if decision == DepthDecision.ADVANCE:
        return f"Advance to depth level {current_depth + 1}"
    
    elif decision == DepthDecision.REFINE_THEN_ADVANCE:
        # Identify specific quality issues
        issues = []
        if depth_assessment.quality_metrics.get('cnr', 100) < 20:
            issues.append("increase conceptual novelty")
        if depth_assessment.quality_metrics.get('scr', 100) < 70:
            issues.append("reduce redundancy")
        
        action = f"Refine quality"
        if issues:
            action += f" ({', '.join(issues)})"
        action += f", then advance to depth {current_depth + 1}"
        return action
    
    elif decision == DepthDecision.CONTINUE_DEPTH:
        # Address specific coverage gaps
        gaps = depth_assessment.coverage.critical_gaps
        if gaps:
            return f"Continue at depth {current_depth}: address {gaps[0]}"
        return f"Continue exploring depth {current_depth}"
    
    elif decision == DepthDecision.FORCE_ADVANCE:
        return f"Force advance to depth {current_depth + 1} (diminishing returns detected)"
    
    return f"Continue at depth {current_depth}"
```

**Acceptance Criteria**:
- [ ] Integrated assessment combines both evaluations
- [ ] Decision matrix implemented correctly
- [ ] Saturation detection using CNR
- [ ] Action recommendations are specific
- [ ] Logging shows decision reasoning

---

## Phase 4: Orchestrator Integration

### Task 4.1: State Tracking Enhancements

**File**: `src/states/topic_focused.py` (MODIFY)

```python
# Add these fields to TopicFocusedState

class TopicFocusedState(TypedDict):
    # ... existing fields ...
    
    # Depth tracking (ADD THESE)
    depth_assessment: Optional[Dict[str, Any]]      # Last depth assessment results
    depth_history: List[Dict[str, Any]]             # History of depth transitions
    iterations_at_current_depth: int                # Iterations spent at current depth
    depth_transition_log: List[str]                 # Human-readable transition log
```

**Acceptance Criteria**:
- [ ] New depth tracking fields added
- [ ] Type hints correct
- [ ] Docstrings updated

---

### Task 4.2: Modify Discriminator Evaluator for Depth

**File**: `src/workflow/orchestrator.py` (MODIFY)

**Modify the `_discriminator_evaluator` method**:

```python
async def _discriminator_evaluator(self, state: TopicFocusedState) -> dict:
    """Evaluate discriminator response with integrated quality+depth assessment"""
    logger.info("📊 DISCRIMINATOR EVALUATOR (Quality + Depth)")
    
    current_response = state.get("current_response", "")
    discriminator_iterations = state.get("discriminator_iterations", 0)
    discriminator_revisions = state.get("discriminator_revisions", [])
    iterations = state.get("iterations", 0)
    current_depth = state.get("current_depth_level", 1)
    iterations_at_depth = state.get("iterations_at_current_depth", 0)
    
    # Prepare previous content
    previous_content = None
    if discriminator_revisions:
        previous_content = discriminator_revisions[-1][0]
    
    # Standard quality evaluation
    eval_state = {
        "previous_content": previous_content,
        "context": state.get("context", {})
    }
    
    evaluation_result = await self.discriminator_evaluator.evaluate(
        content=current_response,
        previous_content=previous_content,
        context=eval_state.get("context")
    )
    
    quality_score = evaluation_result.total_score
    discriminator_revisions.append((current_response, quality_score))
    
    logger.info(f"Quality: {quality_score:.1f}/100 ({evaluation_result.tier})")
    
    # Quality thresholds
    max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))
    meets_quality = quality_score >= QUALITY_THRESHOLD
    max_refinements_reached = discriminator_iterations >= MAX_REFINEMENT_ITERATIONS - 1
    
    # If quality not met and can still refine, request revision
    if not meets_quality and not max_refinements_reached:
        logger.info(f"Quality {quality_score:.1f} < {QUALITY_THRESHOLD}, requesting revision")
        return {
            "discriminator_revisions": discriminator_revisions,
            "discriminator_iterations": discriminator_iterations + 1,
            "evaluator_feedback": evaluation_result.detailed_feedback,
            "quality_scores": {
                **state.get("quality_scores", {}),
                f"discriminator_rev_{discriminator_iterations}": quality_score
            },
            "evaluation": evaluation_result
        }
    
    # Quality sufficient or max refinements reached - select best response
    best_response, best_score = max(discriminator_revisions, key=lambda x: x[1])
    
    if meets_quality:
        logger.info(f"✅ Quality threshold met ({quality_score:.1f} >= {QUALITY_THRESHOLD})")
    else:
        logger.info(f"⚠️  Max refinements reached, using best: {best_score:.1f}")
    
    # Store best revision
    if best_response:
        await self.discriminator_evaluator.store_best_revision(
            content=best_response,
            agent_id="discriminator",
            metadata={
                "quality_score": best_score,
                "tier": evaluation_result.tier,
                "iteration": iterations,
                "depth": current_depth
            }
        )
    
    # NOW PERFORM INTEGRATED DEPTH ASSESSMENT
    logger.info("🎯 Performing integrated depth assessment...")
    
    conversation_history = await self.discriminator.get_recent_context(n=10)
    
    integrated_result = await self.discriminator_evaluator.integrated_assessment(
        conversation_history=conversation_history,
        current_depth=current_depth,
        current_content=best_response,
        previous_content=previous_content,
        state=state
    )
    
    # Log integrated assessment
    logger.info(f"""
📊 INTEGRATED ASSESSMENT RESULTS
================================
Decision: {integrated_result.decision.value}
Reason: {integrated_result.reason}
Quality Score: {integrated_result.quality_score:.1f}/100
Coverage Score: {integrated_result.metrics_summary['coverage_score']:.1f}/100
CNR: {integrated_result.metrics_summary['cnr']:.1f}
SCR: {integrated_result.metrics_summary['scr']:.1f}
Net Signal: {integrated_result.metrics_summary['net_signal']}
Action: {integrated_result.recommended_action}
Critical Gaps: {', '.join(integrated_result.metrics_summary.get('critical_gaps', [])[:3])}
    """)
    
    # Determine new depth level based on decision
    new_depth = current_depth
    depth_transitioned = False
    transition_reason = ""
    
    if integrated_result.decision == DepthDecision.ADVANCE:
        if current_depth < 5:
            new_depth = current_depth + 1
            depth_transitioned = True
            transition_reason = integrated_result.reason
            logger.info(f"✅ DEPTH ADVANCEMENT: {current_depth} → {new_depth}")
        else:
            logger.info(f"📍 Already at max depth (5)")
    
    elif integrated_result.decision == DepthDecision.FORCE_ADVANCE:
        if current_depth < 5:
            new_depth = current_depth + 1
            depth_transitioned = True
            transition_reason = integrated_result.reason + " (forced)"
            logger.info(f"⏭️  FORCED DEPTH ADVANCEMENT: {current_depth} → {new_depth}")
    
    elif integrated_result.decision == DepthDecision.REFINE_THEN_ADVANCE:
        # Quality needs work but coverage is good
        # Set flag to advance after next quality improvement
        logger.info(f"🔧 Quality refinement needed before advancing")
        return {
            "evaluator_feedback": integrated_result.recommended_action,
            "discriminator_iterations": discriminator_iterations + 1,
            "pending_depth_advance": True,
            "target_depth": current_depth + 1 if current_depth < 5 else current_depth
        }
    
    else:  # CONTINUE_DEPTH
        logger.info(f"⏸️  STAYING AT DEPTH {current_depth}")
    
    # Update depth history if transitioned
    depth_history = state.get("depth_history", [])
    if depth_transitioned:
        depth_history.append({
            'from_depth': current_depth,
            'to_depth': new_depth,
            'iteration': iterations,
            'coverage_score': integrated_result.metrics_summary['coverage_score'],
            'quality_score': integrated_result.quality_score,
            'reason': transition_reason
        })
    
    # Check if should continue conversation or go to summarizer
    if iterations < max_iterations:
        logger.info(f"Iteration {iterations + 1}/{max_iterations}: Continuing conversation")
        
        # Determine new iterations_at_depth counter
        new_iterations_at_depth = 0 if depth_transitioned else iterations_at_depth + 1
        
        # Force advance if stuck too long at one depth
        if new_iterations_at_depth >= self.discriminator_evaluator.depth_thresholds.max_iterations_per_depth:
            if new_depth == current_depth and current_depth < 5:
                logger.warning(f"⚠️  Stuck at depth {current_depth} for {new_iterations_at_depth} iterations, forcing advance")
                new_depth = current_depth + 1
                depth_transitioned = True
                depth_history.append({
                    'from_depth': current_depth,
                    'to_depth': new_depth,
                    'iteration': iterations,
                    'reason': 'Forced: max iterations per depth exceeded'
                })
                new_iterations_at_depth = 0
        
        return {
            "discriminator_revisions": discriminator_revisions,
            "discriminator_iterations": 0,
            "generator_iterations": 0,
            "iterations": iterations + 1,
            "current_depth_level": new_depth,  # ← DEPTH UPDATED HERE
            "iterations_at_current_depth": new_iterations_at_depth,
            "depth_history": depth_history,
            "depth_assessment": {
                'decision': integrated_result.decision.value,
                'coverage_score': integrated_result.metrics_summary['coverage_score'],
                'quality_score': integrated_result.quality_score,
                'metrics': integrated_result.metrics_summary,
                'recommendation': integrated_result.recommended_action
            },
            "last_followup_question": best_response,
            "evaluator_feedback": None,
            "generator_revisions": [],
            "quality_scores": {
                **state.get("quality_scores", {}),
                f"discriminator_{iterations}": best_score
            },
            "current_response": best_response,
            "evaluation": evaluation_result
        }
    else:
        # Max iterations reached, go to summarizer
        logger.info(f"Max iterations ({max_iterations}) reached, routing to summarizer")
        return {
            "discriminator_revisions": discriminator_revisions,
            "iterations": iterations,
            "current_depth_level": new_depth,
            "depth_history": depth_history,
            "depth_assessment": {
                'decision': integrated_result.decision.value,
                'coverage_score': integrated_result.metrics_summary['coverage_score'],
                'final': True
            },
            "quality_scores": {**state.get("quality_scores", {}), "discriminator_final": best_score},
            "current_response": best_response,
            "evaluation": evaluation_result,
            "evaluator_feedback": None
        }
```

**Acceptance Criteria**:
- [ ] Integrated assessment called after quality check
- [ ] Depth advancement based on IntegratedAssessment decision
- [ ] Depth history tracked
- [ ] Iterations at depth counter managed
- [ ] Force advance after max iterations per depth
- [ ] Comprehensive logging

---

### Task 4.3: Initialize Depth Tracking in run_async

**File**: `src/workflow/orchestrator.py` (MODIFY)

**Modify the `run_async` method to initialize depth tracking**:

```python
async def run_async(
    self,
    topic: str,
    max_iterations: int = 3,
    quality_threshold: float = 75.0
) -> AsyncGenerator[AgentResponse, None]:
    """Run orchestrated workflow with depth progression"""
    
    logger.info(f"🚀 Starting workflow: {topic}")
    logger.info(f"Config: max_iterations={max_iterations}, threshold={quality_threshold}")
    
    state = TopicFocusedState(
        original_topic=topic,
        current_depth_level=1,  # Start at depth 1
        aspects_explored=[],
        topic_summary="",
        last_followup_question="",
        iterations=0,
        current_response="",
        # Quality tracking
        generator_revisions=[],
        discriminator_revisions=[],
        evaluator_feedback=None,
        quality_scores={},
        generator_iterations=0,
        discriminator_iterations=0,
        best_generator_content=None,
        evaluation=None,
        context={},
        # Depth tracking (NEW)
        depth_assessment=None,
        depth_history=[],
        iterations_at_current_depth=0,
        depth_transition_log=[]
    )
    
    # ... rest of method unchanged ...
```

**Acceptance Criteria**:
- [ ] Depth tracking fields initialized
- [ ] depth_level starts at 1
- [ ] iterations_at_current_depth starts at 0

---

## Phase 5: Testing & Validation

### Task 5.1: Unit Tests for Depth Assessment

**File**: `tests/test_depth_assessment.py` (NEW)

```python
"""
Unit tests for depth assessment functionality.
"""
import pytest
import asyncio
from src.agents.evaluator import EvaluatorAgent
from src.agents.depth_assessment import DepthDecision
from src.config.depth_thresholds import DepthThresholds


@pytest.mark.asyncio
async def test_depth_1_coverage_complete():
    """Test Level 1 correctly identifies completion"""
    evaluator = EvaluatorAgent(session_id="test")
    
    # Mock conversation with clear definitions
    conversation = [
        {'agent_id': 'user', 'content': 'What is machine learning?'},
        {'agent_id': 'generator', 'content': 'Machine learning is a subset of AI...'},
        {'agent_id': 'discriminator', 'content': 'Building on that, the key components are...'}
    ]
    
    assessment = await evaluator.assess_depth_coverage(
        conversation_history=conversation,
        current_depth=1,
        aspects_explored=['definition', 'components'],
        topic_summary="ML defined as AI subset for pattern learning",
        original_topic="Machine Learning"
    )
    
    # Should have good coverage if definitions are clear
    assert assessment.coverage.total_score >= 60  # At least moderate coverage
    assert assessment.current_depth == 1


@pytest.mark.asyncio
async def test_depth_advancement_signal_detection():
    """Test that advancement signals are detected"""
    evaluator = EvaluatorAgent(session_id="test")
    
    # Conversation with clear shift from "what" to "how"
    conversation = [
        {'agent_id': 'generator', 'content': 'Now that we understand what ML is, how does it actually work?'},
        {'agent_id': 'discriminator', 'content': 'The mechanism involves...'}
    ]
    
    assessment = await evaluator.assess_depth_coverage(
        conversation_history=conversation,
        current_depth=1,
        aspects_explored=['definition', 'scope', 'terminology'],
        topic_summary="ML fundamentals covered",
        original_topic="Machine Learning"
    )
    
    # Should detect advancement signals (questions about "how")
    assert assessment.signals.advancement_count > 0


@pytest.mark.asyncio
async def test_threshold_configuration():
    """Test threshold loading from environment"""
    thresholds = DepthThresholds.from_env()
    
    assert thresholds.coverage_minimum > 0
    assert thresholds.coverage_minimum <= 100
    assert 1 in thresholds.depth_specific
    assert 'coverage_minimum' in thresholds.depth_specific[1]


@pytest.mark.asyncio
async def test_integrated_assessment_advance_decision():
    """Test integrated assessment decides to advance correctly"""
    evaluator = EvaluatorAgent(session_id="test")
    
    # Mock high-quality, high-coverage scenario
    # (This would need actual conversation data in real test)
    # Placeholder for now
    pass


@pytest.mark.asyncio
async def test_saturation_detection():
    """Test that low CNR triggers force advance"""
    evaluator = EvaluatorAgent(session_id="test")
    
    # This test would check that when CNR < 15, force advance occurs
    # Needs integration with actual evaluator
    pass
```

**Acceptance Criteria**:
- [ ] Tests for each depth level assessment
- [ ] Tests for signal detection
- [ ] Tests for threshold loading
- [ ] Tests for integrated decisions
- [ ] Tests pass locally

---

### Task 5.2: Integration Tests

**File**: `tests/test_depth_progression.py` (NEW)

```python
"""
Integration tests for depth progression through workflow.
"""
import pytest
import asyncio
from src.workflow.orchestrator import WorkflowOrchestrator


@pytest.mark.asyncio
async def test_full_depth_progression():
    """Test progression through multiple depth levels"""
    orchestrator = WorkflowOrchestrator(session_id="test_depth")
    
    depth_levels_seen = []
    
    async for response in orchestrator.run_async(
        topic="Explain neural networks",
        max_iterations=10  # Enough to potentially reach depth 3-4
    ):
        if response.metadata and 'current_depth_level' in response.metadata:
            depth = response.metadata['current_depth_level']
            if not depth_levels_seen or depth != depth_levels_seen[-1]:
                depth_levels_seen.append(depth)
    
    # Should progress through at least 2-3 depth levels
    assert len(set(depth_levels_seen)) >= 2
    assert depth_levels_seen[0] == 1  # Starts at 1
    
    # Should advance monotonically (never go backward)
    for i in range(1, len(depth_levels_seen)):
        assert depth_levels_seen[i] >= depth_levels_seen[i-1]


@pytest.mark.asyncio
async def test_depth_stuck_force_advance():
    """Test force advance when stuck at depth too long"""
    orchestrator = WorkflowOrchestrator(session_id="test_stuck")
    
    # Set max_iterations_per_depth low for testing
    orchestrator.discriminator_evaluator.depth_thresholds.max_iterations_per_depth = 3
    
    depth_transitions = []
    
    async for response in orchestrator.run_async(
        topic="Simple topic",
        max_iterations=8
    ):
        if response.metadata and 'depth_history' in response.metadata:
            transitions = response.metadata['depth_history']
            if len(transitions) > len(depth_transitions):
                depth_transitions = transitions
    
    # Should eventually force advance even if coverage incomplete
    assert len(depth_transitions) > 0


@pytest.mark.asyncio  
async def test_depth_history_tracking():
    """Test that depth transitions are properly tracked"""
    orchestrator = WorkflowOrchestrator(session_id="test_history")
    
    final_history = None
    
    async for response in orchestrator.run_async(
        topic="Quantum computing",
        max_iterations=6
    ):
        if response.metadata and 'depth_history' in response.metadata:
            final_history = response.metadata['depth_history']
    
    # Should have depth history
    assert final_history is not None
    
    # Each transition should have required fields
    for transition in final_history:
        assert 'from_depth' in transition
        assert 'to_depth' in transition
        assert 'coverage_score' in transition
        assert 'reason' in transition
```

**Acceptance Criteria**:
- [ ] Test full workflow with depth progression
- [ ] Test force advance mechanism
- [ ] Test depth history tracking
- [ ] All tests pass

---

### Task 5.3: Manual Testing Scenarios

**File**: `docs/depth_testing_guide.md` (NEW)

```markdown
# Manual Testing Guide for Depth Progression

## Test Scenario 1: Normal Progression
**Topic**: "Explain machine learning"
**Expected**: Should progress from Level 1 (definitions) → Level 2 (mechanisms) → Level 3 (applications)

**Verification**:
- Check logs for "ADVANCING TO DEPTH LEVEL X"
- Monitor coverage scores at each level
- Verify advancement signals detected

## Test Scenario 2: Complex Topic
**Topic**: "Explain quantum entanglement"
**Expected**: May take more iterations per depth, but should still advance

**Verification**:
- iterations_at_current_depth should be higher
- Eventually advances via coverage threshold or force advance

## Test Scenario 3: Simple Topic
**Topic**: "What is addition?"
**Expected**: Quick progression through depths, may reach Level 5

**Verification**:
- Faster depth transitions
- Higher coverage scores earlier

## Test Scenario 4: Force Advance
**Topic**: Any topic with max_iterations_per_depth=3
**Expected**: Should force advance after 3 iterations at same depth

**Verification**:
- Look for "Forced: max iterations per depth exceeded" in logs
- Depth should advance even if coverage < 75

## Monitoring Commands

### Watch depth progression in real-time:
```bash
tail -f logs/conversation_*.md | grep -E "DEPTH|Coverage|ADVANCING"
```

### Check final depth history:
```python
python -c "
import asyncio
from src.workflow.orchestrator import WorkflowOrchestrator

async def check():
    orch = WorkflowOrchestrator(session_id='test')
    async for response in orch.run_async('Test topic', max_iterations=6):
        if response.metadata and 'depth_history' in response.metadata:
            print('Depth History:', response.metadata['depth_history'])

asyncio.run(check())
"
```
```

**Acceptance Criteria**:
- [ ] Manual test scenarios documented
- [ ] Verification steps clear
- [ ] Monitoring commands provided

---

## Phase 6: Documentation & Deployment

### Task 6.1: Update README

**File**: `README.md` (MODIFY)

Add section:

```markdown
## Depth Progression System

The conversation system now features intelligent depth progression across 5 levels:

1. **Level 1: Foundations** - Core definitions and basic concepts
2. **Level 2: Mechanisms** - Underlying principles and how things work
3. **Level 3: Applications** - Real-world examples and use cases
4. **Level 4: Edge Cases** - Challenges, limitations, and controversies
5. **Level 5: Future Directions** - Cutting-edge developments and speculation

### How It Works

Conversations automatically advance through depth levels when:
- **Coverage threshold met**: Current level sufficiently explored (default: 75%)
- **Quality threshold met**: Content meets quality standards (default: 70%)
- **Advancement signals detected**: Questions shift to next level concerns

Depth progression is paused if:
- Coverage gaps remain at current level
- Quality needs improvement
- Anti-signals detected (confusion about current level)

### Configuration

Set thresholds in `.env`:
```bash
DEPTH_COVERAGE_MINIMUM=75.0      # Minimum coverage to advance
DEPTH_QUALITY_MINIMUM=70.0       # Minimum quality to advance
DEPTH_CNR_MINIMUM=20.0           # Minimum novelty (prevents repetition)
MAX_ITERATIONS_PER_DEPTH=6       # Force advance after N iterations
```

### Monitoring

View depth progression in conversation logs:
```bash
tail -f logs/conversation_*.md | grep "DEPTH"
```
```

**Acceptance Criteria**:
- [ ] README updated with depth system overview
- [ ] Configuration documented
- [ ] Monitoring instructions provided

---

### Task 6.2: Environment Variable Template

**File**: `.env.example` (MODIFY)

Add:

```bash
# ============================================
# Depth Progression Configuration
# ============================================

# Coverage thresholds (0-100)
DEPTH_COVERAGE_MINIMUM=75.0
DEPTH_COVERAGE_EXCELLENT=90.0
DEPTH_COVERAGE_ACCEPTABLE=60.0

# Quality gates for depth advancement
DEPTH_QUALITY_MINIMUM=70.0
DEPTH_CNR_MINIMUM=20.0           # Conceptual Novelty Rate minimum
DEPTH_SCR_MINIMUM=70.0           # Semantic Compression Ratio minimum

# Iteration constraints per depth level
MAX_ITERATIONS_PER_DEPTH=6       # Force advance after this many iterations
MIN_ITERATIONS_PER_DEPTH=2       # Minimum time at each depth

# Depth-specific overrides (optional)
DEPTH_1_COVERAGE_MIN=80.0        # Stricter for foundations
DEPTH_3_COVERAGE_MIN=70.0        # Relaxed for applications
DEPTH_5_COVERAGE_MIN=65.0        # Most relaxed for speculation
```

**Acceptance Criteria**:
- [ ] All new environment variables documented
- [ ] Reasonable defaults provided
- [ ] Comments explain each setting

---

### Task 6.3: Migration Guide

**File**: `docs/DEPTH_MIGRATION.md` (NEW)

```markdown
# Migrating to Depth Progression System

## For Existing Installations

### 1. Update Dependencies
No new dependencies required. The depth system uses existing LLM and evaluation infrastructure.

### 2. Add Environment Variables
Copy the depth configuration section from `.env.example` to your `.env`:

```bash
DEPTH_COVERAGE_MINIMUM=75.0
DEPTH_QUALITY_MINIMUM=70.0
# ... etc
```

### 3. Update Database/State (if applicable)
The `TopicFocusedState` now includes depth tracking fields. These are automatically initialized for new conversations. Existing conversations in progress will start depth tracking from their next iteration.

### 4. Review Logs
Depth progression information is now logged. You'll see:
- `🎯 Assessing depth N coverage...`
- `✅ ADVANCING TO DEPTH LEVEL N`
- `⏸️ STAYING AT DEPTH LEVEL N`

### 5. Test Your Configuration
Run a test conversation:
```bash
python client.py "Explain neural networks"
```

Check logs for depth progression:
```bash
tail -f logs/conversation_*.md | grep "DEPTH"
```

## Backward Compatibility

The depth system is fully backward compatible:
- Old conversations without depth tracking continue to work
- If depth assessment fails, conversation continues at current depth
- Can disable depth progression by setting `MAX_ITERATIONS_PER_DEPTH=999`

## Troubleshooting

### Depth never advances
- Check `DEPTH_COVERAGE_MINIMUM` - may be set too high
- Review depth assessment logs for coverage scores
- Verify LLM is responding with proper JSON format

### Advances too quickly
- Increase `DEPTH_COVERAGE_MINIMUM`
- Increase `MIN_ITERATIONS_PER_DEPTH`
- Check quality thresholds

### Gets stuck at one depth
- Reduce `DEPTH_COVERAGE_MINIMUM`
- Ensure `MAX_ITERATIONS_PER_DEPTH` is set (default: 6)
- Check for JSON parsing errors in depth assessment
```

**Acceptance Criteria**:
- [ ] Migration steps clear
- [ ] Backward compatibility explained
- [ ] Troubleshooting guide provided

---

## Implementation Timeline

### Week 1: Foundation (Phase 1)
- **Days 1-2**: Tasks 1.1, 1.2 (Data structures & config)
- **Days 3-4**: Task 1.3 (Depth criteria)
- **Day 5**: Testing and validation of Phase 1

### Week 2: Core Logic (Phase 2)
- **Days 1-2**: Task 2.1 (Depth prompts)
- **Days 3-5**: Task 2.2 (Evaluator methods)

### Week 3: Integration (Phases 3-4)
- **Days 1-2**: Task 3.1 (Integrated assessment)
- **Days 3-4**: Tasks 4.1, 4.2, 4.3 (Orchestrator integration)
- **Day 5**: End-to-end integration testing

### Week 4: Testing & Documentation (Phases 5-6)
- **Days 1-2**: Tasks 5.1, 5.2 (Unit & integration tests)
- **Days 3-4**: Task 5.3, 6.1, 6.2, 6.3 (Manual testing & docs)
- **Day 5**: Final validation and deployment

---

## Success Metrics

### Functional Metrics
- [ ] Conversations progress through at least 2-3 depth levels
- [ ] Depth advancement decisions are explainable and logged
- [ ] Coverage scores accurately reflect conversation depth
- [ ] Quality and depth gates work together correctly
- [ ] Force advance prevents infinite loops at one depth

### Quality Metrics
- [ ] 90%+ of depth assessments parse successfully (JSON)
- [ ] Average coverage score at advancement: 75-85
- [ ] Average quality score at advancement: 70-80
- [ ] Depth transitions correlate with question type shifts

### Performance Metrics
- [ ] Depth assessment adds < 2 seconds per iteration
- [ ] No memory leaks in depth history tracking
- [ ] LLM token usage increase < 20% (depth assessment prompts)

---

## Rollout Strategy

### Phase 1: Internal Testing (Week 4)
- Deploy to development environment
- Run 20+ test conversations across various topics
- Collect depth progression data
- Tune thresholds based on observations

### Phase 2: Limited Beta (Week 5)
- Enable for subset of users/sessions
- Monitor depth progression metrics
- Gather feedback on conversation quality
- Adjust thresholds if needed

### Phase 3: Full Deployment (Week 6)
- Roll out to all users
- Monitor error rates and performance
- Continue threshold tuning based on real data

---

## Monitoring & Observability

### Key Metrics to Track

```python
# Add to orchestrator.py

class DepthMetrics:
    """Track depth progression metrics for monitoring"""
    
    @staticmethod
    def log_depth_transition(
        session_id: str,
        from_depth: int,
        to_depth: int,
        coverage_score: float,
        quality_score: float,
        iterations_at_depth: int,
        decision: str
    ):
        """Log depth transition for monitoring"""
        logger.info(f"""
📊 DEPTH TRANSITION METRICS
Session: {session_id}
Transition: Depth {from_depth} → {to_depth}
Coverage Score: {coverage_score:.1f}/100
Quality Score: {quality_score:.1f}/100
Iterations at Depth: {iterations_at_depth}
Decision: {decision}
        """)
        
        # Could send to monitoring service (Prometheus, DataDog, etc.)
        # metrics.depth_transitions.labels(
        #     from_depth=from_depth,
        #     to_depth=to_depth
        # ).inc()
```

### Dashboard Queries

For monitoring dashboards, track:

1. **Average coverage score at advancement** per depth level
2. **Average iterations per depth level**
3. **Percentage of conversations reaching each depth**
4. **Force advance rate** (how often MAX_ITERATIONS_PER_DEPTH triggers)
5. **Depth assessment parsing errors**
6. **Decision distribution** (ADVANCE vs CONTINUE_DEPTH vs FORCE_ADVANCE)

### Alert Conditions

Set up alerts for:
- Depth assessment JSON parsing failure rate > 10%
- Average coverage at advancement < 60 (too lenient)
- Average coverage at advancement > 90 (too strict)
- Force advance rate > 40% (thresholds too high)
- Sessions stuck at depth 1 for > 8 iterations

---

## Risk Mitigation

### Risk 1: LLM produces malformed JSON
**Impact**: High - breaks depth assessment
**Mitigation**: 
- Robust JSON parsing with fallbacks
- Default to CONTINUE_DEPTH on parse failure
- Retry with simpler prompt format
- Log all parsing failures for analysis

**Implementation**:
```python
def _parse_depth_assessment_response(self, response: str) -> Dict:
    """Parse with multiple fallback strategies"""
    try:
        # Strategy 1: Extract JSON block
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        logger.warning("JSON parsing failed, trying cleanup...")
        
    try:
        # Strategy 2: Clean and retry
        cleaned = self._clean_json_response(response)
        return json.loads(cleaned)
    except:
        logger.error("All JSON parsing strategies failed")
        return self._get_default_assessment()
```

### Risk 2: Depth advances too quickly/slowly
**Impact**: Medium - affects conversation quality
**Mitigation**:
- Configurable thresholds per deployment
- Adaptive threshold adjustment based on outcomes
- Force advance safety valve
- A/B testing different threshold values

### Risk 3: Depth assessment is too slow
**Impact**: Medium - increases latency
**Mitigation**:
- Cache depth criteria (don't rebuild each time)
- Limit conversation history to last 6-10 turns
- Use smaller/faster LLM for assessment if available
- Parallel execution where possible

### Risk 4: Coverage criteria don't match actual depth
**Impact**: High - core feature validity
**Mitigation**:
- Iterative refinement of criteria based on testing
- Human evaluation of depth assessments
- Compare against manual depth labeling
- Collect feedback on depth appropriateness

---

## Testing Checklist

### Pre-Implementation Testing
- [ ] Review all new data structures
- [ ] Validate depth criteria for each level
- [ ] Test threshold configuration loading
- [ ] Verify JSON prompt format

### Unit Testing
- [ ] Test DepthAssessment dataclass creation
- [ ] Test DepthThresholds.from_env()
- [ ] Test depth criteria retrieval
- [ ] Test JSON parsing (valid and malformed)
- [ ] Test decision logic with various score combinations
- [ ] Test force advance conditions

### Integration Testing
- [ ] Test full workflow with depth progression
- [ ] Test depth advancement triggers correctly
- [ ] Test depth history tracking
- [ ] Test iterations_at_current_depth counter
- [ ] Test force advance after max iterations
- [ ] Test REFINE_THEN_ADVANCE path
- [ ] Test integration with quality evaluator

### End-to-End Testing
- [ ] Test with simple topic (e.g., "addition")
- [ ] Test with moderate topic (e.g., "machine learning")
- [ ] Test with complex topic (e.g., "quantum entanglement")
- [ ] Test reaching depth level 5
- [ ] Test conversation stopping at various depths
- [ ] Test with different threshold configurations

### Edge Case Testing
- [ ] Test with malformed LLM responses
- [ ] Test with very short conversations (< 3 turns)
- [ ] Test with single-word user inputs
- [ ] Test with off-topic tangents
- [ ] Test with LLM failures/timeouts

### Performance Testing
- [ ] Measure latency increase from depth assessment
- [ ] Test with 50+ iterations (long conversations)
- [ ] Check memory usage over time
- [ ] Test concurrent sessions with depth tracking

---

## Code Review Checklist

### Architecture
- [ ] Depth assessment properly separated from quality evaluation
- [ ] No circular dependencies between modules
- [ ] Clean interfaces between components
- [ ] Proper error handling at boundaries

### Code Quality
- [ ] Type hints on all functions
- [ ] Docstrings for all public methods
- [ ] Logging at appropriate levels (info, warning, error)
- [ ] No hardcoded values (use config)
- [ ] Constants properly defined

### Testing
- [ ] Unit tests cover happy path
- [ ] Unit tests cover error cases
- [ ] Integration tests cover key workflows
- [ ] Test fixtures properly isolated

### Documentation
- [ ] README updated
- [ ] Migration guide complete
- [ ] API documentation generated
- [ ] Example usage provided

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Environment variables documented
- [ ] Migration guide reviewed
- [ ] Rollback plan prepared

### Deployment Steps
1. [ ] Update `.env.example` with new variables
2. [ ] Deploy code to staging environment
3. [ ] Run smoke tests on staging
4. [ ] Update production `.env` with depth config
5. [ ] Deploy to production
6. [ ] Monitor logs for errors
7. [ ] Run validation test conversation
8. [ ] Verify depth progression in logs

### Post-Deployment
- [ ] Monitor error rates for 24 hours
- [ ] Check depth transition metrics
- [ ] Gather initial feedback
- [ ] Document any issues found
- [ ] Plan first tuning iteration

---

## Tuning Guide

### Initial Baseline (Week 1)
Use conservative thresholds:
```bash
DEPTH_COVERAGE_MINIMUM=75.0
DEPTH_QUALITY_MINIMUM=70.0
DEPTH_CNR_MINIMUM=20.0
MAX_ITERATIONS_PER_DEPTH=6
```

### Data Collection (Weeks 2-3)
Track these metrics:
- Average coverage score at advancement per depth
- Average iterations per depth
- User satisfaction scores (if available)
- Depth reached in typical conversations

### Analysis Questions
1. Are conversations advancing too quickly? → Increase coverage thresholds
2. Are conversations stuck at depth 1? → Decrease coverage thresholds
3. Is quality dropping after advancement? → Increase quality thresholds
4. Are force advances happening often? → Adjust coverage or iteration limits

### Tuning Iterations
**If coverage scores at advancement average 85-95:**
→ Thresholds might be too high, consider lowering by 5 points

**If coverage scores at advancement average 50-65:**
→ Thresholds might be too low, consider raising by 5-10 points

**If conversations rarely reach depth 3:**
→ Consider depth-specific overrides (lower threshold for depth 2→3)

**If quality drops significantly after depth advancement:**
→ Increase DEPTH_QUALITY_MINIMUM or add quality refinement step

### Adaptive Tuning (Advanced)
Implement learning system that adjusts thresholds:
```python
def tune_threshold_from_outcomes(
    depth: int,
    outcomes: List[Dict]
) -> float:
    """
    Adjust threshold based on conversation outcomes.
    
    Good outcomes: Quality remains high after advancement
    Bad outcomes: Quality drops after advancement
    """
    good_outcomes = [o for o in outcomes if o['quality_after'] >= 70]
    
    if not good_outcomes:
        return 75.0  # Default
    
    # Find coverage sweet spot
    avg_coverage = np.mean([o['coverage_at_advance'] for o in good_outcomes])
    return avg_coverage
```

---

## Maintenance Plan

### Weekly Tasks
- [ ] Review depth progression logs for anomalies
- [ ] Check JSON parsing error rate
- [ ] Monitor average coverage scores
- [ ] Review force advance frequency

### Monthly Tasks
- [ ] Analyze depth progression patterns across all conversations
- [ ] Tune thresholds based on collected data
- [ ] Update depth criteria if needed
- [ ] Review and address any recurring issues

### Quarterly Tasks
- [ ] Major threshold optimization based on data
- [ ] Evaluate depth criteria effectiveness
- [ ] User survey on conversation depth satisfaction
- [ ] Consider ML-based threshold tuning

---

## Future Enhancements

### Phase 2 Enhancements (Post-Launch)

#### 1. Dynamic Aspect Tracking
Currently `aspects_explored` is a simple list. Enhance to:
```python
aspects_explored = [
    {
        'aspect': 'neural_architecture',
        'depth_introduced': 2,
        'coverage_level': 'thorough',
        'last_mentioned': 5
    }
]
```

#### 2. Topic Summary Auto-Update
Implement automatic topic summary generation:
```python
async def update_topic_summary(
    self,
    conversation_history: List[Dict],
    current_summary: str,
    new_aspects: List[str]
) -> str:
    """Generate updated topic summary after each iteration"""
    # Use LLM to synthesize new summary
    pass
```

#### 3. Depth-Specific Quality Metrics
Different quality expectations at different depths:
- Depth 1: Clarity and precision matter most
- Depth 2: Logical coherence matters most  
- Depth 3: Concrete examples matter most
- Depth 4: Critical analysis matters most
- Depth 5: Forward-looking perspective matters most

#### 4. User Preference Learning
Learn user preferences for depth progression:
```python
user_depth_preferences = {
    'prefers_quick_overview': False,
    'prefers_deep_dive': True,
    'typical_depth_reached': 4.2,
    'avg_iterations_per_depth': 3.5
}
```

#### 5. Visual Depth Progression
In web UI, show visual indicator:
```
🔵━━━━━ Level 1: Foundations (Complete ✓)
🔵━━━━━ Level 2: Mechanisms (Complete ✓)
🔵━━━━━ Level 3: Applications (In Progress... 67%)
⚪━━━━━ Level 4: Edge Cases
⚪━━━━━ Level 5: Future Directions
```

#### 6. Depth Bookmarks
Allow users to "bookmark" specific depths:
```python
"Return to depth 2 discussion about neural network training"
```

#### 7. Multi-Branch Depth Exploration
Instead of linear 1→2→3→4→5, allow branching:
```
Depth 1: Foundations
  ├→ Depth 2a: Mechanisms
  └→ Depth 2b: History & Context
```

---

## Appendix A: Complete File Structure

```
project/
├── src/
│   ├── agents/
│   │   ├── base.py (existing)
│   │   ├── evaluator.py (MODIFY - add depth methods)
│   │   ├── generator.py (existing)
│   │   ├── discriminator.py (existing)
│   │   └── depth_assessment.py (NEW)
│   ├── config/
│   │   ├── depth_thresholds.py (NEW)
│   │   └── depth_criteria.py (NEW)
│   ├── prompts/
│   │   ├── depth_prompts.py (NEW)
│   │   ├── generator_prompts.py (existing)
│   │   └── discriminator_prompts.py (existing)
│   ├── states/
│   │   └── topic_focused.py (MODIFY - add depth fields)
│   └── workflow/
│       └── orchestrator.py (MODIFY - integrate depth)
├── tests/
│   ├── test_depth_assessment.py (NEW)
│   └── test_depth_progression.py (NEW)
├── docs/
│   ├── DEPTH_MIGRATION.md (NEW)
│   └── depth_testing_guide.md (NEW)
├── .env (MODIFY - add depth config)
├── .env.example (MODIFY - add depth config)
└── README.md (MODIFY - add depth section)
```

---

## Appendix B: Sample Conversation Log with Depth

```markdown
<user>
**Explain quantum entanglement**
</user>

<generator>
Quantum entanglement is a phenomenon where two particles become correlated...
Follow-up: Would you like to understand the mathematical basis?
</generator>

📊 DEPTH 1 ASSESSMENT
Coverage: 65/100 (Needs more work)
- Core definition: 15/20 ✓
- Key components: 10/20 (partial)
- Basic scope: 12/20 (partial)
- Fundamental properties: 15/20 ✓
- Terminology: 13/20 (partial)
Decision: CONTINUE_DEPTH (gaps in components and scope)

<discriminator>
Building on that definition, the key components are entangled particles...
Follow-up: How does measurement affect entangled states?
</discriminator>

📊 DEPTH 1 ASSESSMENT
Coverage: 82/100 (Good!)
Net Signal: +2 (advancement signals detected)
Decision: ADVANCE
✅ ADVANCING TO DEPTH LEVEL 2

<generator>
[Depth 2: Mechanisms]
The mechanism works through quantum superposition...
</generator>

📊 DEPTH 2 ASSESSMENT
Coverage: 71/100 (Getting there)
Decision: CONTINUE_DEPTH

...conversation continues...

📊 FINAL DEPTH HISTORY
1→2 at iteration 2 (coverage: 82, quality: 76)
2→3 at iteration 5 (coverage: 78, quality: 74)
3→4 at iteration 8 (coverage: 73, quality: 72)
```

---

## Appendix C: Environment Variables Reference

| Variable                    | Default | Range | Description                             |
| --------------------------- | ------- | ----- | --------------------------------------- |
| `DEPTH_COVERAGE_MINIMUM`    | 75.0    | 0-100 | Minimum coverage score to advance depth |
| `DEPTH_COVERAGE_EXCELLENT`  | 90.0    | 0-100 | Excellent coverage threshold            |
| `DEPTH_COVERAGE_ACCEPTABLE` | 60.0    | 0-100 | Acceptable coverage for edge cases      |
| `DEPTH_QUALITY_MINIMUM`     | 70.0    | 0-100 | Minimum quality score to advance        |
| `DEPTH_CNR_MINIMUM`         | 20.0    | 0-100 | Minimum conceptual novelty rate         |
| `DEPTH_SCR_MINIMUM`         | 70.0    | 0-100 | Minimum semantic compression score      |
| `MAX_ITERATIONS_PER_DEPTH`  | 6       | 1-20  | Force advance after N iterations        |
| `MIN_ITERATIONS_PER_DEPTH`  | 2       | 1-10  | Minimum iterations before can advance   |
| `DEPTH_1_COVERAGE_MIN`      | 80.0    | 0-100 | Override for depth 1 (stricter)         |
| `DEPTH_3_COVERAGE_MIN`      | 70.0    | 0-100 | Override for depth 3 (relaxed)          |
| `DEPTH_5_COVERAGE_MIN`      | 65.0    | 0-100 | Override for depth 5 (most relaxed)     |

---

## Appendix D: Troubleshooting Guide

### Issue: Depth never advances

**Symptoms**: Conversation stays at depth 1 for entire session

**Diagnostic Steps**:
1. Check logs for depth assessment scores:
   ```bash
   grep "DEPTH.*ASSESSMENT" logs/conversation_*.md
   ```

2. Look for coverage scores:
   ```bash
   grep "Coverage:" logs/conversation_*.md
   ```

3. Check for JSON parsing errors:
   ```bash
   grep "JSON parsing failed" logs/application.log
   ```

**Solutions**:
- **If coverage scores are 50-70**: Lower `DEPTH_COVERAGE_MINIMUM`
- **If JSON parsing errors**: Check LLM prompt format, add retry logic
- **If net signal is negative**: Review anti-signal detection logic
- **If quality scores low**: Address quality issues before depth can advance

### Issue: Advances too quickly

**Symptoms**: Reaches depth 3-4 in just 3-4 iterations

**Solutions**:
- Increase `DEPTH_COVERAGE_MINIMUM` by 5-10 points
- Increase `MIN_ITERATIONS_PER_DEPTH` to 3-4
- Review advancement signal detection (may be too sensitive)

### Issue: JSON parsing failures

**Symptoms**: Logs show "Failed to parse depth assessment JSON"

**Solutions**:
1. Add more robust JSON extraction:
   ```python
   # Try finding JSON between markdown code blocks
   json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
   ```

2. Simplify prompt to be more explicit about JSON format

3. Add retry with "Output ONLY valid JSON" instruction

### Issue: Force advances happening too often

**Symptoms**: > 30% of depth transitions are forced

**Solutions**:
- Reduce `MAX_ITERATIONS_PER_DEPTH` (currently too high)
- Lower `DEPTH_COVERAGE_MINIMUM` (too hard to meet)
- Review if depth criteria are too strict

---

## Sign-Off

### Implementation Team Sign-Off

- [ ] **Tech Lead**: Architecture reviewed and approved
- [ ] **Backend Engineer**: Implementation plan feasible
- [ ] **QA Engineer**: Testing strategy comprehensive
- [ ] **DevOps**: Deployment plan viable
- [ ] **Product Owner**: Feature meets requirements

### Ready for Implementation

This plan is ready for Claude Code to begin implementation.

**Estimated Total Effort**: 4 weeks (1 developer)
**Risk Level**: Medium
**Business Value**: High (enables systematically deeper conversations)

**Start Date**: _____________
**Target Completion**: _____________

---

*End of Implementation Plan*