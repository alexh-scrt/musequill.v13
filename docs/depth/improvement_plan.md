# Evaluator-Driven Depth Advancement: Design Document

## Overview

This approach leverages your existing `EvaluatorAgent` infrastructure to intelligently assess when a conversation has sufficiently explored the current depth level and is ready to advance to the next level.

## Core Philosophy

**Depth advancement should be earned, not automatic.** The conversation progresses deeper only when the current depth level demonstrates:
- Sufficient conceptual coverage
- Quality discourse at the current level
- Natural readiness for more advanced exploration

## Architecture

### 1. Depth Assessment Module

Add a new responsibility to the `EvaluatorAgent` class:

```python
# src/agents/evaluator.py

class DepthAssessment:
    """Assessment of depth level coverage and readiness for advancement"""
    
    def __init__(
        self,
        current_depth: int,
        depth_coverage_score: float,  # 0-100: how well current depth is explored
        readiness_for_next: bool,     # Should we advance?
        coverage_details: Dict[str, Any],
        recommendation: str           # Human-readable explanation
    ):
        self.current_depth = current_depth
        self.depth_coverage_score = depth_coverage_score
        self.readiness_for_next = readiness_for_next
        self.coverage_details = coverage_details
        self.recommendation = recommendation


class EvaluatorAgent(BaseAgent):
    # ... existing code ...
    
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
        
        Returns DepthAssessment with recommendation on depth advancement.
        """
        # Implementation details below
        pass
```

### 2. Depth Progression Criteria

Define clear criteria for each depth level. A level is "complete" when:

#### Level 1: Foundations (Core definitions and concepts)
- **Coverage**: Key terms defined, basic concepts explained
- **Quality**: Clear definitions, no ambiguity in fundamentals
- **Metrics**:
  - Conceptual Novelty Rate (CNR) > 25% for foundational concepts
  - At least 3-5 core concepts introduced and explained
  - Structural coherence > 70%
- **Signals for advancement**:
  - Both agents agree on fundamentals
  - Follow-up questions start referencing "how" rather than "what"
  - Repetition of basic definitions (indicates saturation)

#### Level 2: Mechanisms (Underlying principles and how things work)
- **Coverage**: Causal relationships explained, mechanisms detailed
- **Quality**: Logical flow, cause-effect clarity
- **Metrics**:
  - Claim density > 0.4 (substantive claims about mechanisms)
  - Evidence of causal reasoning in discourse
  - Connection between concepts established
- **Signals for advancement**:
  - Mechanisms understood and explained
  - Questions shift to "where" and "when" (applications)
  - Abstract principles grounded

#### Level 3: Applications (Real-world examples and implications)
- **Coverage**: Practical examples, use cases, implementations
- **Quality**: Concrete examples with clear connections to principles
- **Metrics**:
  - At least 3-5 distinct applications discussed
  - Real-world references or case studies mentioned
  - Bridge between theory and practice established
- **Signals for advancement**:
  - Multiple application domains covered
  - Questions probe boundaries and limitations
  - "What if" scenarios emerging

#### Level 4: Edge Cases (Challenges, limitations, controversies)
- **Coverage**: Known problems, failure modes, debates
- **Quality**: Nuanced understanding, multiple perspectives
- **Metrics**:
  - Critical analysis present (not just positive aspects)
  - Trade-offs and limitations acknowledged
  - Controversies or open questions identified
- **Signals for advancement**:
  - Edge cases thoroughly explored
  - Limitations clearly understood
  - Questions turn toward future possibilities

#### Level 5: Future Directions (Cutting-edge and open questions)
- **Coverage**: Research frontiers, emerging developments, speculation
- **Quality**: Forward-looking, informed speculation
- **Metrics**:
  - Future-oriented language increases
  - Hypothetical scenarios explored
  - Research directions or open problems identified
- **Signals for completion**:
  - Topic exhaustion detected
  - Circular reasoning (revisiting earlier points)
  - Natural conversation closure

### 3. Implementation Strategy

#### Phase 1: Depth Coverage Scoring

```python
async def assess_depth_coverage(
    self,
    conversation_history: List[Dict],
    current_depth: int,
    aspects_explored: List[str],
    topic_summary: str,
    original_topic: str
) -> DepthAssessment:
    """Main assessment logic"""
    
    # Extract conversation text
    recent_turns = conversation_history[-6:]  # Focus on recent discussion
    conversation_text = self._format_conversation(recent_turns)
    
    # Get depth-specific criteria
    depth_criteria = self._get_depth_criteria(current_depth)
    
    # Build assessment prompt
    assessment_prompt = f"""
You are evaluating depth coverage in a conversation about "{original_topic}".

CURRENT DEPTH LEVEL: {current_depth}/5
DEPTH LEVEL FOCUS: {depth_criteria['focus']}

RECENT CONVERSATION:
{conversation_text}

ASPECTS ALREADY EXPLORED: {', '.join(aspects_explored) if aspects_explored else 'None tracked'}

TOPIC SUMMARY SO FAR:
{topic_summary if topic_summary else 'No summary yet'}

---

Assess the conversation against these criteria for Depth Level {current_depth}:

**Required Coverage:**
{depth_criteria['required_coverage']}

**Quality Indicators:**
{depth_criteria['quality_indicators']}

**Advancement Signals:**
{depth_criteria['advancement_signals']}

---

Provide your assessment:

1. **Coverage Score** (0-100): How thoroughly has this depth level been explored?
   - 0-40: Superficial, needs much more exploration
   - 41-70: Partial coverage, key gaps remain
   - 71-85: Good coverage, minor gaps
   - 86-100: Comprehensive coverage, ready to advance

2. **Key Gaps**: What critical aspects of this depth level are still unexplored?

3. **Advancement Readiness**: Should we advance to depth {current_depth + 1}?
   - YES: If coverage score >= 75 AND quality indicators met AND advancement signals present
   - NO: If coverage score < 75 OR critical gaps remain OR quality insufficient

4. **Recommendation**: Brief explanation of your decision.

Format your response as JSON:
{{
    "coverage_score": <number>,
    "key_gaps": [<list of gaps>],
    "advancement_ready": <true/false>,
    "recommendation": "<explanation>"
}}
"""
    
    # Get LLM assessment
    response = await self.generate_with_llm(
        assessment_prompt,
        system_prompt="You are a depth coverage evaluator. Assess conversation depth objectively."
    )
    
    # Parse response
    assessment_data = self._parse_assessment_response(response)
    
    # Create DepthAssessment object
    return DepthAssessment(
        current_depth=current_depth,
        depth_coverage_score=assessment_data['coverage_score'],
        readiness_for_next=assessment_data['advancement_ready'],
        coverage_details={
            'key_gaps': assessment_data['key_gaps'],
            'criteria_met': assessment_data.get('criteria_met', {}),
        },
        recommendation=assessment_data['recommendation']
    )
```

#### Phase 2: Depth Criteria Definitions

```python
def _get_depth_criteria(self, depth: int) -> Dict[str, str]:
    """Get evaluation criteria for specific depth level"""
    
    criteria = {
        1: {
            'focus': 'Core definitions, basic concepts, fundamental understanding',
            'required_coverage': """
- Key terms and concepts clearly defined
- Fundamental principles explained
- Basic "what is X?" questions answered
- Foundation established for deeper exploration
            """,
            'quality_indicators': """
- Clear, unambiguous definitions
- No confusion about core concepts
- Consistent terminology usage
- Both agents demonstrate shared understanding
            """,
            'advancement_signals': """
- Definitions no longer being questioned
- Questions shift from "what" to "how" or "why"
- Repetition of basic concepts (saturation)
- Natural progression toward mechanisms
            """
        },
        2: {
            'focus': 'Underlying principles, mechanisms, how things work',
            'required_coverage': """
- Causal relationships explained
- Step-by-step processes described
- Mechanisms and dynamics detailed
- "How does X work?" questions addressed
            """,
            'quality_indicators': """
- Logical cause-effect chains established
- Clear explanation of processes
- Internal consistency in explanations
- Evidence of systematic understanding
            """,
            'advancement_signals': """
- Mechanisms thoroughly explained
- Questions shift to applications and examples
- "Where/when is this used?" questions emerge
- Interest in practical implications
            """
        },
        3: {
            'focus': 'Real-world applications, examples, practical implications',
            'required_coverage': """
- Concrete examples and use cases provided
- Real-world implementations discussed
- Practical implications explored
- Theory-practice connections established
            """,
            'quality_indicators': """
- Specific, detailed examples (not generic)
- Multiple application domains covered
- Clear links between theory and practice
- Practical constraints acknowledged
            """,
            'advancement_signals': """
- Sufficient examples provided (3-5+)
- Questions probe boundaries and edge cases
- Interest in limitations and failures
- "What doesn't work?" questions appear
            """
        },
        4: {
            'focus': 'Challenges, limitations, controversies, edge cases',
            'required_coverage': """
- Known problems and limitations discussed
- Failure modes and edge cases explored
- Trade-offs and constraints acknowledged
- Controversies or debates presented
            """,
            'quality_indicators': """
- Balanced view (not just benefits)
- Multiple perspectives considered
- Nuanced understanding demonstrated
- Critical thinking evident
            """,
            'advancement_signals': """
- Limitations thoroughly understood
- Questions turn forward-looking
- Interest in future developments
- "What's next?" or "What's unsolved?" questions
            """
        },
        5: {
            'focus': 'Future directions, cutting-edge developments, open questions',
            'required_coverage': """
- Emerging trends and developments discussed
- Research frontiers identified
- Open questions and unsolved problems noted
- Speculative possibilities explored
            """,
            'quality_indicators': """
- Forward-looking perspective
- Informed speculation (not wild guessing)
- Connection to current state of knowledge
- Acknowledgment of uncertainty
            """,
            'advancement_signals': """
- Topic feels exhausted
- Circular reasoning or repetition
- Both agents agree on natural closure
- Summary-oriented discussion
            """
        }
    }
    
    return criteria.get(depth, criteria[5])  # Default to level 5
```

#### Phase 3: Integration with Orchestrator

Modify `orchestrator.py` to call depth assessment:

```python
# src/workflow/orchestrator.py

async def _discriminator_evaluator(self, state: TopicFocusedState) -> dict:
    """Evaluate discriminator response and decide routing"""
    logger.info("📊 DISCRIMINATOR EVALUATOR")
    
    # ... existing evaluation code ...
    
    # After quality evaluation, assess depth coverage
    if not meets_quality and not max_refinements_reached:
        # Still refining quality, don't assess depth yet
        logger.info("Quality refinement in progress, skipping depth assessment")
    else:
        # Quality is acceptable, now assess depth coverage
        logger.info("🎯 Assessing depth coverage...")
        
        conversation_history = await self.discriminator.get_recent_context(n=10)
        
        depth_assessment = await self.discriminator_evaluator.assess_depth_coverage(
            conversation_history=conversation_history,
            current_depth=state.get('current_depth_level', 1),
            aspects_explored=state.get('aspects_explored', []),
            topic_summary=state.get('topic_summary', ''),
            original_topic=state.get('original_topic', '')
        )
        
        logger.info(
            f"📊 Depth {depth_assessment.current_depth} coverage: "
            f"{depth_assessment.depth_coverage_score:.1f}/100"
        )
        logger.info(f"📊 Ready for next depth: {depth_assessment.readiness_for_next}")
        logger.info(f"📊 Recommendation: {depth_assessment.recommendation}")
        
        # Determine new depth level
        current_depth = state.get('current_depth_level', 1)
        new_depth = current_depth
        
        if depth_assessment.readiness_for_next and current_depth < 5:
            new_depth = current_depth + 1
            logger.info(f"✅ ADVANCING TO DEPTH LEVEL {new_depth}")
        else:
            logger.info(f"⏸️  REMAINING AT DEPTH LEVEL {current_depth}")
    
    # Now decide routing based on conversation iterations
    if iterations < max_iterations:
        logger.info(f"Iteration {iterations + 1}/{max_iterations}: Continuing conversation")
        
        return {
            "discriminator_revisions": discriminator_revisions,
            "discriminator_iterations": 0,
            "generator_iterations": 0,
            "iterations": iterations + 1,
            "current_depth_level": new_depth,  # ← Update depth based on assessment
            "last_followup_question": best_response,
            "evaluator_feedback": None,
            "generator_revisions": [],
            "quality_scores": {
                **state.get("quality_scores", {}), 
                f"discriminator_{iterations}": best_score,
                f"depth_coverage_{iterations}": depth_assessment.depth_coverage_score
            },
            "current_response": best_response,
            "depth_assessment": {
                'coverage_score': depth_assessment.depth_coverage_score,
                'gaps': depth_assessment.coverage_details.get('key_gaps', []),
                'recommendation': depth_assessment.recommendation
            }
        }
```

### 4. Enhanced State Tracking

Update `TopicFocusedState` to include depth tracking:

```python
# src/states/topic_focused.py

class TopicFocusedState(TypedDict):
    # ... existing fields ...
    
    # Enhanced depth tracking
    depth_assessment: Optional[Dict[str, Any]]  # Last depth assessment results
    depth_history: List[Dict[str, Any]]  # History of depth transitions
    depth_coverage_scores: Dict[int, float]  # Coverage score per depth level
```

### 5. Fallback Mechanisms

Implement safety checks to prevent getting stuck:

```python
def _should_force_advance(
    self,
    current_depth: int,
    iterations_at_depth: int,
    coverage_score: float
) -> bool:
    """
    Force depth advancement if stuck too long at a level.
    
    Prevents infinite loops at a single depth level.
    """
    # Force advance after 6 iterations at same depth, even if coverage incomplete
    if iterations_at_depth >= 6:
        logger.warning(
            f"⚠️  Forcing depth advancement after {iterations_at_depth} iterations "
            f"at depth {current_depth}"
        )
        return True
    
    # Force advance if coverage is "good enough" (>60%) and we're making little progress
    if coverage_score > 60 and iterations_at_depth >= 4:
        logger.info(
            f"📈 Forcing depth advancement: coverage {coverage_score:.1f}% "
            f"after {iterations_at_depth} iterations"
        )
        return True
    
    return False
```

## Advantages of This Approach

1. **Intelligent Progression**: Depth advances based on actual conversation quality, not arbitrary iteration counts
2. **Leverages Existing Infrastructure**: Uses your sophisticated `EvaluatorAgent` framework
3. **Explainable Decisions**: Each depth transition has a clear rationale
4. **Quality-Aware**: Ensures each depth level is properly explored before advancing
5. **Flexible**: Can adjust criteria and thresholds without changing core architecture
6. **Safe**: Includes fallback mechanisms to prevent getting stuck

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Add `DepthAssessment` class to `evaluator.py`
- [ ] Implement `assess_depth_coverage()` method
- [ ] Define depth criteria for all 5 levels
- [ ] Add comprehensive logging

### Phase 2: Integration (Week 2)
- [ ] Modify `_discriminator_evaluator()` to call depth assessment
- [ ] Update state management to track depth transitions
- [ ] Implement fallback mechanisms
- [ ] Add depth advancement to state updates

### Phase 3: Refinement (Week 3)
- [ ] Test with various conversation topics
- [ ] Tune coverage thresholds
- [ ] Adjust depth criteria based on observations
- [ ] Optimize prompt engineering for assessment

### Phase 4: Enhancement (Week 4)
- [ ] Track aspects explored per depth level
- [ ] Update topic summary dynamically
- [ ] Add depth-specific quality metrics
- [ ] Implement visualization of depth progression

## Testing Strategy

### Unit Tests
```python
# tests/test_depth_assessment.py

async def test_depth_1_coverage_complete():
    """Test that Level 1 correctly identifies completion"""
    evaluator = EvaluatorAgent()
    
    # Mock conversation with clear definitions and basic concepts
    conversation = [...]
    
    assessment = await evaluator.assess_depth_coverage(
        conversation_history=conversation,
        current_depth=1,
        aspects_explored=['definition', 'basic_concept'],
        topic_summary="Core concepts defined",
        original_topic="Machine Learning"
    )
    
    assert assessment.depth_coverage_score >= 75
    assert assessment.readiness_for_next == True


async def test_depth_1_coverage_incomplete():
    """Test that Level 1 correctly identifies gaps"""
    evaluator = EvaluatorAgent()
    
    # Mock conversation with incomplete basics
    conversation = [...]
    
    assessment = await evaluator.assess_depth_coverage(
        conversation_history=conversation,
        current_depth=1,
        aspects_explored=[],
        topic_summary="",
        original_topic="Machine Learning"
    )
    
    assert assessment.depth_coverage_score < 75
    assert assessment.readiness_for_next == False
    assert len(assessment.coverage_details['key_gaps']) > 0
```

### Integration Tests
```python
async def test_full_depth_progression():
    """Test complete progression through all 5 depth levels"""
    orchestrator = WorkflowOrchestrator(session_id="test")
    
    depth_transitions = []
    
    async for response in orchestrator.run_async(
        topic="Quantum Computing",
        max_iterations=15
    ):
        if 'depth_level' in response.metadata:
            depth_transitions.append(response.metadata['depth_level'])
    
    # Should progress through multiple depth levels
    unique_depths = set(depth_transitions)
    assert len(unique_depths) >= 3  # At least 3 different depths reached
    
    # Should advance monotonically (never go backward)
    for i in range(1, len(depth_transitions)):
        assert depth_transitions[i] >= depth_transitions[i-1]
```

## Monitoring and Observability

Add logging and metrics to track depth progression:

```python
# Log each depth assessment
logger.info(f"""
📊 DEPTH ASSESSMENT
==================
Depth Level: {current_depth}/5
Coverage Score: {depth_assessment.depth_coverage_score:.1f}/100
Ready to Advance: {depth_assessment.readiness_for_next}
Key Gaps: {', '.join(depth_assessment.coverage_details['key_gaps'][:3])}
Recommendation: {depth_assessment.recommendation}
""")

# Track metrics
metrics = {
    'depth_level': current_depth,
    'coverage_score': depth_assessment.depth_coverage_score,
    'iterations_at_depth': iterations_at_current_depth,
    'total_iterations': iterations,
    'advancement_decision': depth_assessment.readiness_for_next
}
```

## Expected Outcomes

After implementation, conversations should:

1. **Start at Level 1** with foundational discussions
2. **Naturally progress** through levels as coverage improves
3. **Spend appropriate time** at each level based on complexity
4. **Reach Level 4-5** for deep, extended conversations
5. **Maintain quality** at each depth level
6. **Provide clear reasoning** for each depth transition

# Summary

This evaluator-driven approach transforms depth progression from a passive counter into an active, intelligent system that responds to conversation quality and coverage. It ensures that depth advancement is earned through substantial exploration rather than automatic iteration counting, resulting in more meaningful and genuinely deep conversations.


## Core Concept

The evaluator assesses whether the current depth level has been sufficiently explored before allowing progression to the next level. **Depth advancement is earned, not automatic.**

## Key Components

### 1. **DepthAssessment Class**
A new data structure that captures:
- Coverage score (0-100) for current depth
- Readiness signal for advancement
- Identified gaps
- Recommendation with reasoning

### 2. **Depth-Specific Criteria**
Each of the 5 depth levels has clear criteria:
- **Level 1**: Definitions and basic concepts complete
- **Level 2**: Mechanisms and principles explained
- **Level 3**: Applications and examples covered
- **Level 4**: Limitations and edge cases explored
- **Level 5**: Future directions discussed

### 3. **Assessment Logic**
The evaluator uses an LLM to analyze:
- Recent conversation turns
- Required coverage for current depth
- Quality indicators
- Advancement signals (like question types shifting)

### 4. **Integration Points**
- Called in `_discriminator_evaluator()` after quality checks pass
- Returns new depth level based on coverage assessment
- Updates state with depth tracking metadata

## Advantages

✅ **Intelligent**: Advances based on actual conversation quality
✅ **Explainable**: Each transition has clear reasoning
✅ **Safe**: Includes fallback mechanisms to prevent getting stuck
✅ **Flexible**: Easy to tune criteria and thresholds
✅ **Consistent**: Leverages your existing evaluator infrastructure

## Implementation Strategy

The document includes:
- Complete code examples for each component
- 4-phase rollout plan
- Unit and integration tests
- Monitoring and logging strategies
- Fallback mechanisms for edge cases
