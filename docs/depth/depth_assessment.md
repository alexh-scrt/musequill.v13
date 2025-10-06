# Depth Assessment: Prompts, Thresholds & Integration

## Part 1: Enhanced Depth Assessment Prompts

### Base Assessment Prompt Template

```python
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
```

### Depth-Specific Assessment Prompts

#### Level 1: Foundations Assessment

```python
def _build_level_1_assessment_prompt(
    self,
    conversation_text: str,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> str:
    return f"""
DEPTH LEVEL 1 ASSESSMENT: Foundations
Topic: "{original_topic}"

CONVERSATION (Last 6 turns):
{conversation_text}

ASPECTS TRACKED: {', '.join(aspects_explored) if aspects_explored else 'None'}
SUMMARY: {topic_summary if topic_summary else 'No summary yet'}

---

ASSESSMENT CRITERIA FOR LEVEL 1 (Foundations):

**Required Coverage** (Must address ALL to score 75+):
1. Core Definition: Is "{original_topic}" clearly defined?
2. Key Components: Are the main parts/elements identified?
3. Basic Scope: Is it clear what falls inside vs. outside this topic?
4. Fundamental Properties: Are essential characteristics explained?
5. Terminology: Are key terms introduced and defined?

**Quality Indicators**:
- Clarity: No ambiguity in core concepts
- Consistency: Terms used consistently by both agents
- Shared Understanding: Both agents demonstrate same baseline knowledge
- Precision: Specific definitions, not vague generalizations

**Advancement Signals** (Indicates readiness for Level 2):
- Questions shift from "What is X?" to "How does X work?"
- Definitions are no longer questioned or clarified
- Both agents reference fundamentals without re-explaining
- Conversation naturally moves toward mechanisms/processes
- Signs of saturation: repetition of basic concepts

**Anti-Signals** (Must stay at Level 1):
- Still defining basic terms
- Confusion about core concepts
- Inconsistent terminology
- "Wait, what exactly is X?" questions
- Foundational disagreements between agents

---

ANALYSIS TASK:

1. **Coverage Score** (0-100):
   - Score each Required Coverage item (0-20 points each)
   - 0 points: Not addressed
   - 5 points: Mentioned superficially
   - 10 points: Partially explained
   - 15 points: Well explained with examples
   - 20 points: Thoroughly covered, clear understanding
   
   Total Coverage Score = Sum of 5 items (0-100)

2. **Quality Assessment**:
   - Count Quality Indicators present (0-4)
   - Note any quality issues

3. **Signal Analysis**:
   - Count Advancement Signals present (0-5)
   - Count Anti-Signals present (0-5)
   - Net Signal = Advancement Signals - Anti-Signals

4. **Gap Identification**:
   - List specific Required Coverage items scored < 15
   - List missing Quality Indicators

5. **Advancement Decision**:
   - ADVANCE if: Coverage Score ≥ 75 AND Net Signal > 0 AND no critical gaps
   - STAY if: Coverage Score < 75 OR Net Signal ≤ 0 OR critical gaps exist

---

OUTPUT FORMAT (JSON):
{{
    "coverage_breakdown": {{
        "core_definition": <0-20>,
        "key_components": <0-20>,
        "basic_scope": <0-20>,
        "fundamental_properties": <0-20>,
        "terminology": <0-20>
    }},
    "coverage_score": <sum of above, 0-100>,
    "quality_indicators_present": [<list of indicators found>],
    "advancement_signals_count": <number>,
    "anti_signals_count": <number>,
    "net_signal": <advancement - anti>,
    "critical_gaps": [<list of items scored < 15>],
    "advancement_ready": <true/false>,
    "recommendation": "<2-3 sentence explanation>",
    "evidence": {{
        "coverage_examples": [<specific quotes showing coverage>],
        "signal_examples": [<specific quotes showing advancement signals>]
    }}
}}

Be precise and evidence-based. Quote specific parts of the conversation.
"""
```

#### Level 2: Mechanisms Assessment

```python
def _build_level_2_assessment_prompt(
    self,
    conversation_text: str,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> str:
    return f"""
DEPTH LEVEL 2 ASSESSMENT: Mechanisms
Topic: "{original_topic}"

CONVERSATION (Last 6 turns):
{conversation_text}

ASPECTS TRACKED: {', '.join(aspects_explored) if aspects_explored else 'None'}
SUMMARY: {topic_summary}

---

ASSESSMENT CRITERIA FOR LEVEL 2 (Mechanisms):

**Required Coverage** (Must address ALL to score 75+):
1. Process Flow: Step-by-step explanation of how {original_topic} works
2. Causal Relationships: What causes what? X → Y → Z chains explained
3. Internal Dynamics: How components interact with each other
4. Underlying Principles: Why does it work this way? Theoretical basis
5. Operational Logic: What are the rules/constraints governing behavior?

**Quality Indicators**:
- Logical Coherence: Cause-effect chains are clear and logical
- Systematic Coverage: Not just isolated facts, but connected understanding
- Depth: Goes beyond surface "what" to deeper "why" and "how"
- Precision: Specific mechanisms, not vague hand-waving

**Advancement Signals** (Indicates readiness for Level 3):
- Questions shift from "How?" to "Where is this used?" or "What are examples?"
- Mechanisms are explained and understood by both agents
- Discussion naturally moves toward applications
- Interest in real-world implementations
- "I understand how it works, but where do we see this?"

**Anti-Signals** (Must stay at Level 2):
- Still asking "How does this work?"
- Confusion about process steps
- Causal relationships unclear
- Requests for more explanation of mechanisms
- "Wait, why does X cause Y?" questions

---

ANALYSIS TASK:

1. **Coverage Score** (0-100):
   Score each Required Coverage item (0-20 points each)
   - 0: Not addressed
   - 5: Mentioned but not explained
   - 10: Basic explanation, lacks detail
   - 15: Good explanation with some detail
   - 20: Thorough explanation, clear mechanisms

2. **Mechanistic Reasoning Check**:
   - Count causal statements (X causes Y, because Z)
   - Identify process descriptions (step 1, then step 2...)
   - Note explanatory depth (surface vs. deep)

3. **Signal Analysis**:
   - Advancement Signals (0-5)
   - Anti-Signals (0-5)
   - Net Signal = Advancement - Anti

4. **Gap Identification**:
   - Unexplained mechanisms
   - Missing causal links
   - Unclear process flows

5. **Advancement Decision**:
   - ADVANCE if: Coverage ≥ 75 AND Net Signal > 0 AND mechanistic reasoning evident
   - STAY if: Coverage < 75 OR Net Signal ≤ 0 OR superficial explanations

---

OUTPUT FORMAT (JSON):
{{
    "coverage_breakdown": {{
        "process_flow": <0-20>,
        "causal_relationships": <0-20>,
        "internal_dynamics": <0-20>,
        "underlying_principles": <0-20>,
        "operational_logic": <0-20>
    }},
    "coverage_score": <0-100>,
    "mechanistic_reasoning": {{
        "causal_statements_count": <number>,
        "process_descriptions_count": <number>,
        "depth_rating": "<surface|moderate|deep>"
    }},
    "quality_indicators_present": [<list>],
    "advancement_signals_count": <number>,
    "anti_signals_count": <number>,
    "net_signal": <number>,
    "critical_gaps": [<list>],
    "advancement_ready": <true/false>,
    "recommendation": "<explanation>",
    "evidence": {{
        "mechanism_examples": [<quotes>],
        "causal_chains": [<X→Y→Z examples>]
    }}
}}
"""
```

#### Level 3: Applications Assessment

```python
def _build_level_3_assessment_prompt(
    self,
    conversation_text: str,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> str:
    return f"""
DEPTH LEVEL 3 ASSESSMENT: Applications
Topic: "{original_topic}"

CONVERSATION (Last 6 turns):
{conversation_text}

ASPECTS TRACKED: {', '.join(aspects_explored) if aspects_explored else 'None'}
SUMMARY: {topic_summary}

---

ASSESSMENT CRITERIA FOR LEVEL 3 (Applications):

**Required Coverage** (Must address ALL to score 75+):
1. Concrete Examples: At least 3-5 real-world examples/use cases
2. Domain Diversity: Examples from different fields/contexts
3. Implementation Details: How is {original_topic} actually applied?
4. Practical Constraints: Real-world limitations, trade-offs
5. Theory-Practice Bridge: Clear connection between principles (Level 2) and applications

**Quality Indicators**:
- Specificity: Named examples, not generic "it's used in industry"
- Concreteness: Detailed enough to visualize/understand
- Diversity: Multiple domains covered
- Relevance: Examples clearly illustrate core concepts

**Advancement Signals** (Indicates readiness for Level 4):
- Questions about limitations: "What doesn't work?" "Where does it fail?"
- Interest in edge cases: "What about scenario X?"
- Critical perspective: "What are the problems with this?"
- Boundary probing: "How far can we push this?"
- Trade-off discussions: "X vs Y, which is better?"

**Anti-Signals** (Must stay at Level 3):
- Still asking "Where is this used?"
- Requesting more examples
- Examples too generic or vague
- Missing application domains
- "Can you give another example?"

---

ANALYSIS TASK:

1. **Coverage Score** (0-100):
   Score each Required Coverage item (0-20 points)

2. **Example Quality Analysis**:
   - Count concrete examples mentioned
   - Assess specificity (generic=1, specific=3)
   - Count domains/fields covered
   - Rate detail level (low/medium/high)

3. **Signal Analysis**:
   - Advancement Signals (0-5)
   - Anti-Signals (0-5)
   - Net Signal = Advancement - Anti

4. **Application Coverage Map**:
   - List all application domains discussed
   - Note missing major domains
   - Identify depth per domain

5. **Advancement Decision**:
   - ADVANCE if: Coverage ≥ 75 AND ≥3 concrete examples AND Net Signal > 0
   - STAY if: Coverage < 75 OR <3 examples OR Net Signal ≤ 0

---

OUTPUT FORMAT (JSON):
{{
    "coverage_breakdown": {{ ... }},
    "coverage_score": <0-100>,
    "example_analysis": {{
        "concrete_examples_count": <number>,
        "examples_list": [<list of examples found>],
        "domains_covered": [<list of domains>],
        "specificity_avg": <1-3>,
        "detail_level": "<low|medium|high>"
    }},
    "advancement_signals_count": <number>,
    "anti_signals_count": <number>,
    "net_signal": <number>,
    "critical_gaps": [<list>],
    "advancement_ready": <true/false>,
    "recommendation": "<explanation>",
    "evidence": {{
        "application_examples": [<quotes>]
    }}
}}
"""
```

#### Levels 4 & 5 (Similar structure, adjusted criteria)

**Level 4**: Focus on critical analysis, limitations, controversies, failure modes
**Level 5**: Focus on future directions, open questions, research frontiers

---

## Part 2: Coverage Threshold Tuning

### Threshold Configuration System

```python
# src/config/depth_thresholds.py

from dataclasses import dataclass
from typing import Dict

@dataclass
class DepthThresholds:
    """Configurable thresholds for depth advancement"""
    
    # Coverage score thresholds (0-100)
    coverage_minimum: float = 75.0      # Minimum coverage to advance
    coverage_excellent: float = 90.0     # Excellent coverage marker
    coverage_acceptable: float = 60.0    # Acceptable if other conditions met
    
    # Signal thresholds
    net_signal_minimum: int = 1          # Advancement signals - Anti signals
    advancement_signals_min: int = 2     # Minimum positive signals needed
    
    # Iteration-based constraints
    max_iterations_per_depth: int = 6    # Force advance after N iterations
    min_iterations_per_depth: int = 2    # Minimum time at each depth
    
    # Quality integration
    quality_score_minimum: float = 70.0  # Must meet quality threshold too
    
    # Depth-specific overrides
    depth_specific: Dict[int, Dict[str, float]] = None
    
    def __post_init__(self):
        if self.depth_specific is None:
            self.depth_specific = {
                1: {  # Level 1: Stricter (foundations must be solid)
                    'coverage_minimum': 80.0,
                    'min_iterations': 2
                },
                2: {  # Level 2: Standard
                    'coverage_minimum': 75.0,
                    'min_iterations': 2
                },
                3: {  # Level 3: Slightly relaxed (examples can be endless)
                    'coverage_minimum': 70.0,
                    'min_iterations': 2
                },
                4: {  # Level 4: Standard
                    'coverage_minimum': 75.0,
                    'min_iterations': 2
                },
                5: {  # Level 5: Most relaxed (speculative by nature)
                    'coverage_minimum': 65.0,
                    'min_iterations': 1
                }
            }
    
    def get_threshold_for_depth(self, depth: int, key: str) -> float:
        """Get depth-specific threshold or fall back to default"""
        if depth in self.depth_specific and key in self.depth_specific[depth]:
            return self.depth_specific[depth][key]
        return getattr(self, key, self.coverage_minimum)


# Usage
thresholds = DepthThresholds()
min_coverage = thresholds.get_threshold_for_depth(depth=1, key='coverage_minimum')
```

### Adaptive Threshold System

```python
class AdaptiveThresholdManager:
    """
    Dynamically adjust thresholds based on conversation characteristics.
    
    Makes thresholds responsive to:
    - Topic complexity
    - Conversation velocity
    - Quality trends
    """
    
    def __init__(self, base_thresholds: DepthThresholds):
        self.base = base_thresholds
        self.history = []  # Track past assessments
    
    def adjust_threshold(
        self,
        base_threshold: float,
        depth: int,
        iterations_at_depth: int,
        quality_trend: str,  # 'improving', 'stable', 'declining'
        topic_complexity: str  # 'simple', 'moderate', 'complex'
    ) -> float:
        """
        Adjust threshold based on context.
        
        Logic:
        - Complex topics: Lower threshold (harder to cover fully)
        - Many iterations at depth: Lower threshold (diminishing returns)
        - Quality declining: Higher threshold (don't advance until quality improves)
        - Quality improving: Keep threshold (let it continue)
        """
        
        adjusted = base_threshold
        
        # Complexity adjustment
        if topic_complexity == 'complex':
            adjusted -= 5.0  # Relax by 5 points
        elif topic_complexity == 'simple':
            adjusted += 5.0  # Tighten by 5 points
        
        # Iteration adjustment (diminishing returns)
        if iterations_at_depth >= 4:
            adjusted -= 10.0  # Relax after many iterations
        elif iterations_at_depth >= 6:
            adjusted -= 20.0  # Strongly relax after excessive iterations
        
        # Quality trend adjustment
        if quality_trend == 'declining':
            adjusted += 10.0  # Don't advance if quality dropping
        elif quality_trend == 'improving':
            adjusted -= 5.0   # Can advance sooner if quality rising
        
        # Clamp to reasonable range
        return max(50.0, min(95.0, adjusted))
    
    def estimate_topic_complexity(
        self,
        original_topic: str,
        conversation_length: int,
        aspects_explored: List[str]
    ) -> str:
        """
        Heuristic complexity estimation.
        
        Complex topics have:
        - Many aspects to explore
        - Long conversations needed
        - Technical terminology
        """
        
        # Simple heuristics (can be enhanced with ML)
        if len(aspects_explored) > 8:
            return 'complex'
        elif len(aspects_explored) < 3 and conversation_length > 6:
            return 'complex'  # Few aspects but long conversation = complex
        elif len(aspects_explored) <= 4:
            return 'simple'
        else:
            return 'moderate'
```

### Threshold Tuning Based on Empirical Data

```python
class ThresholdTuner:
    """
    Tune thresholds based on conversation outcomes.
    
    Collects data on:
    - Conversations that advanced too early (bad quality at next level)
    - Conversations stuck too long (repetitive, no progress)
    - Optimal advancement points (good quality after advancing)
    """
    
    def __init__(self):
        self.advancement_outcomes = []  # Store (coverage_score, quality_after, satisfaction)
    
    def record_advancement_outcome(
        self,
        coverage_at_advancement: float,
        quality_after_advancement: float,
        depth_from: int,
        depth_to: int,
        user_satisfaction: Optional[float] = None
    ):
        """Record outcome of a depth advancement decision"""
        self.advancement_outcomes.append({
            'coverage': coverage_at_advancement,
            'quality_after': quality_after_advancement,
            'depth_transition': f"{depth_from}→{depth_to}",
            'satisfaction': user_satisfaction,
            'timestamp': time.time()
        })
    
    def analyze_optimal_thresholds(self) -> Dict[str, float]:
        """
        Analyze outcomes to find optimal thresholds.
        
        Good advancement: High quality after advancing
        Bad advancement: Quality drops after advancing
        """
        
        if len(self.advancement_outcomes) < 10:
            return {}  # Need more data
        
        # Find sweet spot: coverage where quality_after is maximized
        df = pd.DataFrame(self.advancement_outcomes)
        
        # Group by coverage ranges
        df['coverage_bucket'] = pd.cut(df['coverage'], bins=[0, 60, 70, 75, 80, 85, 90, 100])
        
        # Calculate average quality_after per bucket
        bucket_analysis = df.groupby('coverage_bucket')['quality_after'].agg(['mean', 'std', 'count'])
        
        # Find bucket with highest quality_after
        optimal_bucket = bucket_analysis['mean'].idxmax()
        
        return {
            'optimal_coverage_range': str(optimal_bucket),
            'avg_quality_after': bucket_analysis.loc[optimal_bucket, 'mean'],
            'sample_size': bucket_analysis.loc[optimal_bucket, 'count']
        }
```

---

## Part 3: Integration with Existing Evaluator Metrics

### Current Evaluator Metrics Recap

Your existing `EvaluatorAgent` tracks:
1. **Conceptual Novelty Rate (CNR)**: New concepts vs. repetition
2. **Semantic Compression Ratio (SCR)**: Redundancy detection
3. **Claim Density**: Substantive claims per 100 words
4. **Structural Coherence**: Logical flow and organization

### Integration Strategy

```python
class IntegratedDepthEvaluator(EvaluatorAgent):
    """
    Enhanced evaluator that combines quality metrics with depth assessment.
    
    Key insight: Depth advancement requires BOTH:
    - High depth coverage (topic well explored at current level)
    - High quality metrics (content is good, not just present)
    """
    
    async def integrated_assessment(
        self,
        conversation_history: List[Dict],
        current_depth: int,
        state: TopicFocusedState
    ) -> IntegratedAssessment:
        """
        Perform both quality evaluation AND depth assessment.
        
        Returns unified decision on depth advancement.
        """
        
        # 1. Standard quality evaluation
        current_content = conversation_history[-1]['content']
        previous_content = conversation_history[-2]['content'] if len(conversation_history) > 1 else None
        
        quality_eval = await self.evaluate(
            content=current_content,
            previous_content=previous_content,
            context=state.get('context', {})
        )
        
        # 2. Depth coverage assessment
        depth_assessment = await self.assess_depth_coverage(
            conversation_history=conversation_history,
            current_depth=current_depth,
            aspects_explored=state.get('aspects_explored', []),
            topic_summary=state.get('topic_summary', ''),
            original_topic=state.get('original_topic', '')
        )
        
        # 3. Integrate metrics into depth decision
        integrated_decision = self._make_integrated_decision(
            quality_eval=quality_eval,
            depth_assessment=depth_assessment,
            current_depth=current_depth
        )
        
        return integrated_decision
```

### Metric Integration Rules

```python
def _make_integrated_decision(
    self,
    quality_eval: EvaluationResult,
    depth_assessment: DepthAssessment,
    current_depth: int
) -> IntegratedAssessment:
    """
    Combine quality metrics with depth coverage to make advancement decision.
    
    RULE: Can only advance depth if BOTH quality AND coverage meet thresholds.
    """
    
    # Extract quality metrics
    cnr = quality_eval.metrics.get('conceptual_novelty', {})
    scr = quality_eval.metrics.get('semantic_compression', {})
    claim_density = quality_eval.metrics.get('claim_density', {})
    
    cnr_score = cnr.score if hasattr(cnr, 'score') else 0
    scr_score = scr.score if hasattr(scr, 'score') else 0
    
    # Quality gates for depth advancement
    quality_sufficient = (
        quality_eval.total_score >= 70.0 and  # Overall quality threshold
        cnr_score >= 20.0 and                  # Sufficient novelty (not repetitive)
        scr_score >= 70.0                      # Low redundancy
    )
    
    # Depth coverage gate
    coverage_sufficient = (
        depth_assessment.depth_coverage_score >= 75.0 and
        depth_assessment.readiness_for_next
    )
    
    # Special case: Novelty saturation indicates depth completion
    # If CNR is very low (<15), might indicate depth is "exhausted"
    novelty_saturated = cnr_score < 15.0
    
    # Decision matrix
    if coverage_sufficient and quality_sufficient:
        decision = "ADVANCE"
        reason = "Both coverage and quality thresholds met"
    
    elif coverage_sufficient and not quality_sufficient:
        decision = "REFINE_THEN_ADVANCE"
        reason = "Coverage sufficient but quality needs improvement"
    
    elif not coverage_sufficient and quality_sufficient:
        decision = "CONTINUE_DEPTH"
        reason = "Quality good but coverage gaps remain"
    
    elif novelty_saturated and depth_assessment.depth_coverage_score >= 60:
        decision = "FORCE_ADVANCE"
        reason = "Low novelty indicates depth exhausted (diminishing returns)"
    
    else:
        decision = "CONTINUE_DEPTH"
        reason = "Both coverage and quality need improvement"
    
    # Create integrated assessment
    return IntegratedAssessment(
        decision=decision,
        reason=reason,
        quality_score=quality_eval.total_score,
        coverage_score=depth_assessment.depth_coverage_score,
        metrics_summary={
            'cnr': cnr_score,
            'scr': scr_score,
            'claim_density': claim_density.score if hasattr(claim_density, 'score') else 0,
            'quality_tier': quality_eval.tier,
            'coverage_gaps': depth_assessment.coverage_details.get('key_gaps', [])
        },
        recommended_action=self._get_action_recommendation(decision, quality_eval, depth_assessment)
    )


def _get_action_recommendation(
    self,
    decision: str,
    quality_eval: EvaluationResult,
    depth_assessment: DepthAssessment
) -> str:
    """Provide specific action recommendation based on decision"""
    
    if decision == "ADVANCE":
        return f"Advance to depth {depth_assessment.current_depth + 1}"
    
    elif decision == "REFINE_THEN_ADVANCE":
        # Specific quality issues to fix
        issues = []
        if quality_eval.metrics.get('conceptual_novelty', {}).score < 20:
            issues.append("increase conceptual novelty")
        if quality_eval.metrics.get('semantic_compression', {}).score < 70:
            issues.append("reduce redundancy")
        
        return f"Refine quality ({', '.join(issues)}), then advance"
    
    elif decision == "CONTINUE_DEPTH":
        # Specific coverage gaps to address
        gaps = depth_assessment.coverage_details.get('key_gaps', [])
        if gaps:
            return f"Continue at depth {depth_assessment.current_depth}: address {gaps[0]}"
        return f"Continue exploring depth {depth_assessment.current_depth}"
    
    elif decision == "FORCE_ADVANCE":
        return f"Force advance to depth {depth_assessment.current_depth + 1} (diminishing returns)"
    
    return "Continue current depth"
```

### Metric-Informed Depth Scoring

```python
def _adjust_depth_score_with_quality_metrics(
    self,
    base_coverage_score: float,
    quality_metrics: Dict[str, Any],
    current_depth: int
) -> float:
    """
    Adjust depth coverage score based on quality metrics.
    
    Insight: High coverage with low quality shouldn't count as much.
    """
    
    cnr = quality_metrics.get('conceptual_novelty', 0)
    scr = quality_metrics.get('semantic_compression', 0)
    
    # Quality multiplier (0.7 to 1.3)
    quality_multiplier = 1.0
    
    # Penalize low novelty (indicates repetition, not genuine coverage)
    if cnr < 15:
        quality_multiplier -= 0.2
    elif cnr > 30:
        quality_multiplier += 0.1
    
    # Penalize high redundancy
    if scr < 60:  # High redundancy (low SCR score)
        quality_multiplier -= 0.15
    
    # Bonus for excellent structural coherence
    coherence = quality_metrics.get('structural_coherence', 0)
    if coherence > 85:
        quality_multiplier += 0.1
    
    # Clamp multiplier
    quality_multiplier = max(0.7, min(1.3, quality_multiplier))
    
    # Adjust coverage score
    adjusted_score = base_coverage_score * quality_multiplier
    
    logger.info(
        f"Depth coverage adjusted: {base_coverage_score:.1f} → {adjusted_score:.1f} "
        f"(multiplier: {quality_multiplier:.2f})"
    )
    
    return adjusted_score
```

### Orchestrator Integration

```python
# src/workflow/orchestrator.py

async def _discriminator_evaluator(self, state: TopicFocusedState) -> dict:
    """Enhanced evaluator with integrated depth+quality assessment"""
    
    logger.info("📊 INTEGRATED EVALUATOR (Quality + Depth)")
    
    # ... existing quality evaluation code ...
    
    # After quality check passes, perform integrated assessment
    if meets_quality:
        logger.info("✅ Quality threshold met, performing depth assessment...")
        
        conversation_history = await self.discriminator.get_recent_context(n=10)
        
        # Use integrated evaluator
        integrated_result = await self.discriminator_evaluator.integrated_assessment(
            conversation_history=conversation_history,
            current_depth=state.get('current_depth_level', 1),
            state=state
        )
        
        logger.info(f"""
📊 INTEGRATED ASSESSMENT
========================
Decision: {integrated_result.decision}
Reason: {integrated_result.reason}
Quality Score: {integrated_result.quality_score:.1f}/100
Coverage Score: {integrated_result.coverage_score:.1f}/100
CNR: {integrated_result.metrics_summary['cnr']:.1f}
SCR: {integrated_result.metrics_summary['scr']:.1f}
Action: {integrated_result.recommended_action}
        """)
        
        # Determine new depth based on integrated decision
        current_depth = state.get('current_depth_level', 1)
        new_depth = current_depth
        
        if integrated_result.decision in ["ADVANCE", "FORCE_ADVANCE"]:
            new_depth = min(5, current_depth + 1)
            logger.info(f"✅ ADVANCING: Depth {current_depth} → {new_depth}")
        
        elif integrated_result.decision == "REFINE_THEN_ADVANCE":
            # Set flag to refine quality before advancing
            return {
                "evaluator_feedback": integrated_result.recommended_action,
                "discriminator_iterations": state.get('discriminator_iterations', 0) + 1,
                "pending_depth_advance": True,  # Flag for next iteration
                "target_depth": current_depth + 1
            }
        
        # Return state updates with depth decision
        return {
            "current_depth_level": new_depth,
            "depth_assessment": {
                'decision': integrated_result.decision,
                'coverage_score': integrated_result.coverage_score,
                'quality_score': integrated_result.quality_score,
                'metrics': integrated_result.metrics_summary
            },
            # ... other state updates ...
        }
```

---

## Part 4: Configuration & Monitoring

### Environment Variables

```bash
# .env

# Depth coverage thresholds
DEPTH_COVERAGE_MINIMUM=75.0
DEPTH_COVERAGE_EXCELLENT=90.0
DEPTH_COVERAGE_ACCEPTABLE=60.0

# Depth-specific overrides (JSON format)
DEPTH_1_COVERAGE_MIN=80.0  # Stricter for foundations
DEPTH_3_COVERAGE_MIN=70.0  # Relaxed for applications
DEPTH_5_COVERAGE_MIN=65.0  # Most relaxed for future directions

# Quality gates for depth advancement
DEPTH_QUALITY_MINIMUM=70.0
DEPTH_CNR_MINIMUM=20.0
DEPTH_SCR_MINIMUM=70.0

# Iteration constraints
MAX_ITERATIONS_PER_DEPTH=6
MIN_ITERATIONS_PER_DEPTH=2

# Enable adaptive thresholds
ENABLE_ADAPTIVE_THRESHOLDS=true
```

### Monitoring Dashboard Data

```python
def get_depth_progression_metrics(session_id: str) -> Dict[str, Any]:
    """
    Generate metrics for monitoring depth progression.
    
    Useful for dashboard, debugging, and threshold tuning.
    """
    return {
        'session_id': session_id,
        'depth_progression': [
            {
                'depth': 1,
                'iterations_spent': 3,
                'coverage_at_exit': 82.5,
                'quality_at_exit': 76.2,
                'decision': 'ADVANCE',
                'timestamp': '2025-10-06T10:15:30Z'
            },
            # ... more depth transitions
        ],
        'average_iterations_per_depth': 2.8,
        'depths_reached': 4,
        'advancement_decisions': {
            'ADVANCE': 3,
            'CONTINUE_DEPTH': 5,
            'FORCE_ADVANCE': 1,
            'REFINE_THEN_ADVANCE': 2
        },
        'metric_correlations': {
            'coverage_vs_quality': 0.72,  # Correlation coefficient
            'cnr_vs_advancement': -0.45   # Negative: low CNR triggers advance
        }
    }
```

---

## Summary

### Key Improvements

1. **Depth-Specific Prompts**: Each depth level has tailored assessment criteria
2. **Configurable Thresholds**: Easy to tune via environment variables
3. **Adaptive System**: Thresholds adjust based on topic complexity and trends
4. **Quality Integration**: Depth advancement requires both coverage AND quality
5. **Metric Synergy**: CNR and SCR inform depth decisions (e.g., low CNR = saturation)
6. **Empirical Tuning**: System learns optimal thresholds from outcomes

### Decision Matrix

| Coverage | Quality | CNR | Decision                       |
| -------- | ------- | --- | ------------------------------ |
| ≥75      | ≥70     | ≥20 | **ADVANCE**                    |
| ≥75      | <70     | ≥20 | **REFINE_THEN_ADVANCE**        |
| <75      | ≥70     | ≥20 | **CONTINUE_DEPTH**             |
| ≥60      | Any     | <15 | **FORCE_ADVANCE** (saturation) |
| <60      | <70     | Any | **CONTINUE_DEPTH**             |

This creates an intelligent, adaptive system that advances depth based on both content coverage and conversation quality.

