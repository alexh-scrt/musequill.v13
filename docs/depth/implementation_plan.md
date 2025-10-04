# Implementation Plan: Profile-Aware and Depth-Aware Content Generation

## Context & Current State

**Problem**: The current implementation uses generic prompts for all content generation, leading to:
1. Repetitive content across different depth levels
2. Content that doesn't match the intended audience (scientific vs. popular science)
3. Lack of clear differentiation between depth levels

**User Requirements**:
- Client can specify both `--profile` and `--depth` parameters
- Example: `--depth 3 --profile scientific`
- Different profiles should generate content appropriate for their audience
- Different depths should build on each other without repetition

## Proposed Solution: Profile + Depth Aware Prompt System

### Architecture Overview

```
User Request
    ↓
--profile=scientific --depth=3
    ↓
Prompt Composition System
    ↓
Base Prompt + Profile Config + Depth Config
    ↓
Generation (with previous depth context)
    ↓
Similarity Validation (profile + depth aware)
    ↓
Output
```

---

## Step 1: Create Profile Configuration System

### Objective
Define profile-specific characteristics that influence how content is generated at each depth level.

### Files to Create

**`configs/profiles.yaml`** (or profiles.py if you prefer Python dataclasses)

```yaml
profiles:
  scientific:
    name: "Scientific"
    description: "Rigorous, technical content for researchers and scientists"
    target_audience: "PhD-level researchers, scientists, academics"
    
    depth_1:
      purpose: "Establish formal framework and definitions"
      tone: "formal, precise, technical"
      required_elements:
        - formal_definitions
        - mathematical_notation
        - key_equations
      forbidden_elements:
        - oversimplified_analogies
        - pop_culture_references
      min_citations: 2
      min_equations: 1
      complexity_level: "advanced"
      
    depth_2:
      purpose: "Present detailed derivations and proofs"
      tone: "rigorous, proof-oriented"
      required_elements:
        - mathematical_derivations
        - formal_proofs
        - technical_citations
      forbidden_elements:
        - hand_waving
        - intuition_without_formalism
      min_citations: 3
      min_equations: 3
      complexity_level: "expert"
      
    depth_3:
      purpose: "Discuss open problems and research frontiers"
      tone: "research-level, critical"
      required_elements:
        - recent_papers
        - open_problems
        - controversies
        - unknowns
      forbidden_elements:
        - basic_definitions
        - introductory_material
      min_citations: 5
      min_recent_citations: 3  # Papers from last 3 years
      complexity_level: "cutting_edge"

  popular_science:
    name: "Popular Science"
    description: "Accessible, engaging content for science enthusiasts"
    target_audience: "Educated general public, science enthusiasts"
    
    depth_1:
      purpose: "Explain the big idea and why it matters"
      tone: "engaging, accessible, conversational"
      required_elements:
        - real_world_analogies
        - concrete_examples
        - why_it_matters
      forbidden_elements:
        - unexplained_jargon
        - raw_equations
        - technical_formalism
      max_equations: 0
      require_analogies: true
      complexity_level: "accessible"
      
    depth_2:
      purpose: "Show how it works with intuitive explanations"
      tone: "clear, illustrative, engaging"
      required_elements:
        - step_by_step_explanation
        - visual_descriptions
        - simplified_mathematics
      forbidden_elements:
        - formal_proofs
        - unexplained_notation
      max_equations: 2  # Only if well-explained
      require_examples: true
      complexity_level: "intermediate"
      
    depth_3:
      purpose: "Explore what scientists are discovering"
      tone: "exciting, forward-looking"
      required_elements:
        - current_research
        - future_implications
        - open_questions
      forbidden_elements:
        - technical_derivations
        - dense_formalism
      max_equations: 1
      focus: "implications_and_discoveries"
      complexity_level: "advanced_accessible"

  educational:
    name: "Educational"
    description: "Progressive learning content with examples and exercises"
    target_audience: "Students, learners at various levels"
    
    depth_1:
      purpose: "Build intuition and foundational understanding"
      tone: "friendly, instructional, clear"
      required_elements:
        - learning_objectives
        - conceptual_explanation
        - simple_examples
      min_examples: 2
      require_learning_objectives: true
      complexity_level: "beginner"
      
    depth_2:
      purpose: "Develop problem-solving skills"
      tone: "instructional, practice-oriented"
      required_elements:
        - worked_examples
        - step_by_step_solutions
        - practice_problems
      min_worked_examples: 2
      min_practice_problems: 3
      complexity_level: "intermediate"
      
    depth_3:
      purpose: "Master advanced concepts and applications"
      tone: "challenging, comprehensive"
      required_elements:
        - advanced_problems
        - connections_to_other_topics
        - real_world_applications
      min_advanced_problems: 2
      require_synthesis: true
      complexity_level: "advanced"
```

### Why This Structure

1. **Profile-specific goals**: Each profile has different purposes for the same depth
2. **Explicit constraints**: `required_elements` and `forbidden_elements` enforce differentiation
3. **Measurable requirements**: `min_citations`, `min_equations` can be validated
4. **Tone guidance**: Helps LLM understand the writing style

---

## Step 2: Create Depth Differentiation Framework

### Objective
Define universal depth characteristics that apply across all profiles (but are modified by profile configs).

### File to Create

**`configs/depth_framework.yaml`**

```yaml
depth_framework:
  depth_1:
    universal_purpose: "Introduce and explain WHAT"
    question_pattern: "What is {topic}?"
    information_density: "low"
    abstraction_level: "concrete"
    
    constraints:
      max_paragraphs: 5
      focus: "core_concepts_introduction"
      avoid: "deep_technical_details"
    
    examples:
      good:
        - "Uses clear analogies to explain concepts"
        - "Defines key terms in simple language"
        - "Provides motivation for why topic matters"
      bad:
        - "Jumps into derivations without context"
        - "Uses jargon without explanation"
        - "Assumes prior knowledge"

  depth_2:
    universal_purpose: "Explain and formalize HOW"
    question_pattern: "How does {topic} work?"
    information_density: "medium"
    abstraction_level: "formal"
    
    constraints:
      min_paragraphs: 4
      focus: "mechanisms_and_formalization"
      avoid: "redundant_definitions_from_depth_1"
    
    context_usage:
      previous_depths: [1]
      instructions: |
        BUILD ON depth 1 concepts without restating them.
        Assume reader has already read depth 1.
        Reference depth 1 concepts but immediately go deeper.
    
    examples:
      good:
        - "Provides mathematical formulation of concepts from depth 1"
        - "Shows step-by-step how mechanisms work"
        - "Adds precision and rigor to depth 1 intuitions"
      bad:
        - "Repeats definitions from depth 1"
        - "Re-explains concepts already covered"
        - "Doesn't add new information"

  depth_3:
    universal_purpose: "Explore and critique UNKNOWNS"
    question_pattern: "What don't we know about {topic}? What are the open problems?"
    information_density: "high"
    abstraction_level: "research_level"
    
    constraints:
      min_paragraphs: 5
      focus: "open_problems_and_frontiers"
      avoid: "introductory_or_basic_material"
    
    context_usage:
      previous_depths: [1, 2]
      instructions: |
        ASSUME reader has read depths 1 and 2.
        DO NOT restate material from previous depths.
        Focus on what is NOT known, controversies, and cutting-edge research.
        Critically evaluate or extend previous depths.
    
    examples:
      good:
        - "Identifies specific open problems in the field"
        - "Discusses competing theoretical frameworks"
        - "Cites recent research papers (last 3 years)"
        - "Points out limitations of current understanding"
      bad:
        - "Repeats formalism from depth 2"
        - "Re-introduces concepts from depth 1"
        - "Provides overview rather than frontier content"
```

### Why This Structure

1. **Universal scaffolding**: All profiles follow the same depth progression (WHAT → HOW → UNKNOWNS)
2. **Clear questions**: Each depth answers a different question
3. **Context awareness**: Depths 2 and 3 explicitly reference previous depths
4. **Anti-repetition built-in**: Each depth has explicit "avoid" instructions

---

## Step 3: Implement Prompt Composition System

### Objective
Combine base prompts + profile configs + depth configs into a final, contextual prompt.

### Files to Modify/Create

**New file: `src/prompt_composer.py`**

```python
from typing import Dict, List, Optional
import yaml
from dataclasses import dataclass

@dataclass
class GenerationContext:
    """Context for generating content at a specific depth"""
    topic: str
    depth: int
    profile: str
    previous_depths: Optional[Dict[int, str]] = None
    
class PromptComposer:
    """
    Composes prompts by combining:
    1. Base system prompt
    2. Profile configuration
    3. Depth framework
    4. Context from previous depths
    """
    
    def __init__(self, profile_config_path: str, depth_framework_path: str):
        self.profiles = self._load_yaml(profile_config_path)
        self.depth_framework = self._load_yaml(depth_framework_path)
        self.base_prompt = self._load_base_prompt()
    
    def _load_yaml(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_base_prompt(self) -> str:
        """Load the base system prompt common to all generations"""
        # This would load from prompts/base_system_prompt.txt
        return """
You are an expert content generator creating depth-differentiated content.

Your task is to generate content that:
1. Matches the specified profile and audience
2. Fulfills the requirements of the specified depth level
3. Builds on previous depths without repetition
4. Adds new information and insights

CRITICAL: Each depth must be SUBSTANTIVELY DIFFERENT from other depths.
Do not simply restate or rephrase previous content.
"""
    
    def compose_prompt(self, context: GenerationContext) -> str:
        """
        Main method: Compose a complete prompt for generation
        """
        profile_config = self.profiles['profiles'][context.profile]
        depth_config_profile = profile_config[f'depth_{context.depth}']
        depth_config_universal = self.depth_framework['depth_framework'][f'depth_{context.depth}']
        
        # Build prompt sections
        sections = [
            self.base_prompt,
            self._build_topic_section(context.topic),
            self._build_profile_section(profile_config, depth_config_profile),
            self._build_depth_section(depth_config_universal, context.depth),
            self._build_constraints_section(depth_config_profile),
            self._build_context_section(context, depth_config_universal),
            self._build_examples_section(depth_config_universal),
            self._build_output_requirements(depth_config_profile)
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _build_topic_section(self, topic: str) -> str:
        return f"""
## TOPIC
{topic}
"""
    
    def _build_profile_section(self, profile_config: dict, depth_config: dict) -> str:
        return f"""
## PROFILE: {profile_config['name']}
Target Audience: {profile_config['target_audience']}

Purpose at this depth: {depth_config['purpose']}
Tone: {depth_config['tone']}
Complexity Level: {depth_config['complexity_level']}
"""
    
    def _build_depth_section(self, depth_config: dict, depth: int) -> str:
        return f"""
## DEPTH LEVEL: {depth}
Universal Purpose: {depth_config['universal_purpose']}
Key Question to Answer: {depth_config['question_pattern']}
Information Density: {depth_config['information_density']}
Abstraction Level: {depth_config['abstraction_level']}
"""
    
    def _build_constraints_section(self, depth_config: dict) -> str:
        """Build explicit constraints section"""
        constraints = []
        
        # Required elements
        if 'required_elements' in depth_config:
            req = depth_config['required_elements']
            constraints.append(f"MUST INCLUDE: {', '.join(req)}")
        
        # Forbidden elements
        if 'forbidden_elements' in depth_config:
            forb = depth_config['forbidden_elements']
            constraints.append(f"MUST NOT INCLUDE: {', '.join(forb)}")
        
        # Quantitative requirements
        if 'min_equations' in depth_config:
            constraints.append(f"Minimum equations: {depth_config['min_equations']}")
        
        if 'max_equations' in depth_config:
            constraints.append(f"Maximum equations: {depth_config['max_equations']}")
        
        if 'min_citations' in depth_config:
            constraints.append(f"Minimum citations: {depth_config['min_citations']}")
        
        if 'min_examples' in depth_config:
            constraints.append(f"Minimum examples: {depth_config['min_examples']}")
        
        if not constraints:
            return ""
        
        return f"""
## CONSTRAINTS
{chr(10).join(f"- {c}" for c in constraints)}
"""
    
    def _build_context_section(self, context: GenerationContext, depth_config: dict) -> str:
        """Build section with context from previous depths"""
        if not context.previous_depths or context.depth == 1:
            return ""
        
        # Extract concepts from previous depths
        previous_concepts = {}
        for prev_depth, prev_content in context.previous_depths.items():
            concepts = self._extract_key_concepts(prev_content)
            previous_concepts[prev_depth] = concepts
        
        context_instructions = depth_config.get('context_usage', {}).get('instructions', '')
        
        context_text = f"""
## CONTEXT FROM PREVIOUS DEPTHS

{context_instructions}

"""
        
        for prev_depth, concepts in previous_concepts.items():
            context_text += f"""
Depth {prev_depth} covered these concepts:
{', '.join(concepts)}

"""
        
        context_text += """
CRITICAL ANTI-REPETITION INSTRUCTIONS:
- DO NOT restate or re-explain concepts from previous depths
- DO NOT use similar phrasing or examples as previous depths
- ASSUME the reader has already read previous depths
- BUILD ON previous concepts, don't repeat them
- Add NEW information, NEW perspectives, or NEW insights
"""
        
        return context_text
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts from previous depth content.
        This is a simplified version - in production, you'd use the
        concept extraction from the similarity detector.
        """
        # For now, a placeholder that extracts key phrases
        # In the actual implementation, this should call the
        # concept extraction system from similarity_detector.py
        
        # Placeholder: extract capitalized phrases and technical terms
        import re
        
        # Simple heuristic: extract phrases in quotes, capitalized terms, equations
        concepts = set()
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', content)
        concepts.update(quoted)
        
        # Extract terms like "von Neumann entropy", "Wheeler-DeWitt equation"
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        concepts.update(technical_terms)
        
        return list(concepts)[:10]  # Return top 10 concepts
    
    def _build_examples_section(self, depth_config: dict) -> str:
        """Build examples of good/bad content for this depth"""
        examples = depth_config.get('examples', {})
        if not examples:
            return ""
        
        text = "## EXAMPLES\n\n"
        
        if 'good' in examples:
            text += "✓ GOOD examples for this depth:\n"
            for ex in examples['good']:
                text += f"  - {ex}\n"
            text += "\n"
        
        if 'bad' in examples:
            text += "✗ BAD examples to avoid:\n"
            for ex in examples['bad']:
                text += f"  - {ex}\n"
        
        return text
    
    def _build_output_requirements(self, depth_config: dict) -> str:
        """Final instructions for output format"""
        return """
## OUTPUT REQUIREMENTS

Generate content that:
1. Directly answers the key question for this depth level
2. Matches the specified tone and complexity
3. Meets all quantitative requirements (equations, citations, examples)
4. Includes all required elements
5. Avoids all forbidden elements
6. Does NOT repeat content from previous depths

Write in clear, well-structured prose appropriate for the target audience.
"""


# Usage in generation pipeline
def generate_content_with_profile_and_depth(
    topic: str,
    depth: int,
    profile: str,
    previous_depths: Optional[Dict[int, str]] = None
) -> str:
    """
    Generate content using profile and depth-aware prompts
    """
    
    # Initialize prompt composer
    composer = PromptComposer(
        profile_config_path='configs/profiles.yaml',
        depth_framework_path='configs/depth_framework.yaml'
    )
    
    # Create context
    context = GenerationContext(
        topic=topic,
        depth=depth,
        profile=profile,
        previous_depths=previous_depths
    )
    
    # Compose prompt
    prompt = composer.compose_prompt(context)
    
    # Generate content (using existing LLM interface)
    content = llm.generate(prompt)
    
    return content
```

### Why This Structure

1. **Separation of concerns**: Configuration (YAML) separate from logic (Python)
2. **Composable**: Easy to add new prompt sections
3. **Context-aware**: Automatically includes previous depth concepts
4. **Testable**: Each section can be unit tested
5. **Maintainable**: Change configs without touching code

---

## Step 4: Integrate with Existing Generation Pipeline

### Files to Modify

**`src/generator.py`** (or wherever your main generation logic lives)

### Changes Needed

```python
# Before (hypothetical current state)
def generate_document(topic: str):
    """Old way - generic prompt for all content"""
    generic_prompt = f"Generate content about {topic}"
    content = llm.generate(generic_prompt)
    return content


# After
from prompt_composer import PromptComposer, GenerationContext

class ProfileAwareGenerator:
    def __init__(self):
        self.prompt_composer = PromptComposer(
            profile_config_path='configs/profiles.yaml',
            depth_framework_path='configs/depth_framework.yaml'
        )
        self.similarity_detector = SimilarityDiscriminator()  # From previous plan
    
    def generate_depth_section(
        self,
        topic: str,
        depth: int,
        profile: str,
        previous_depths: Optional[Dict[int, str]] = None,
        max_retries: int = 3
    ) -> tuple[str, dict]:
        """
        Generate content for a specific depth with profile awareness
        Returns: (content, decision_metadata)
        """
        
        for attempt in range(max_retries):
            # Compose prompt
            context = GenerationContext(
                topic=topic,
                depth=depth,
                profile=profile,
                previous_depths=previous_depths
            )
            
            prompt = self.prompt_composer.compose_prompt(context)
            
            # Generate
            content = llm.generate(prompt)
            
            # Validate with similarity detector
            decision = self.similarity_detector.evaluate(
                content=content,
                depth=depth,
                profile=profile,
                previous_depths=previous_depths
            )
            
            if decision['action'] in ['ACCEPT', 'FLAG']:
                return content, decision
            
            elif decision['action'] == 'SKIP':
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} rejected for depth {depth}: "
                    f"{decision['reason']}"
                )
                
                # On retry, enhance prompt with anti-repetition feedback
                if attempt < max_retries - 1:
                    # Add rejected content to context for next attempt
                    self._add_anti_repetition_context(
                        prompt, 
                        rejected_content=content,
                        reason=decision['reason']
                    )
        
        # Failed all retries
        raise GenerationError(
            f"Could not generate acceptable content for depth {depth} "
            f"with profile {profile} after {max_retries} attempts"
        )
    
    def generate_document(
        self,
        topic: str,
        profile: str,
        max_depth: int = 3
    ) -> Document:
        """
        Generate complete document with all requested depths
        """
        document = Document(topic=topic, profile=profile)
        
        for depth in range(1, max_depth + 1):
            logger.info(f"Generating depth {depth} for profile {profile}")
            
            # Get previous depths for context
            previous_depths = document.get_previous_depths()
            
            # Generate this depth
            content, decision = self.generate_depth_section(
                topic=topic,
                depth=depth,
                profile=profile,
                previous_depths=previous_depths
            )
            
            # Add to document
            document.add_section(
                depth=depth,
                content=content,
                metadata={
                    'decision': decision,
                    'profile': profile
                }
            )
            
            logger.info(f"Depth {depth} completed with decision: {decision['action']}")
        
        return document
```

---

## Step 5: Update CLI to Accept Profile and Depth Parameters

### File to Modify

**`cli.py`** or **`main.py`** (wherever your CLI interface is)

### Changes Needed

```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Generate depth-differentiated content with profile awareness'
    )
    
    parser.add_argument(
        'topic',
        type=str,
        help='Topic to generate content about'
    )
    
    parser.add_argument(
        '--profile',
        type=str,
        choices=['scientific', 'popular_science', 'educational'],
        default='scientific',
        help='Content profile (default: scientific)'
    )
    
    parser.add_argument(
        '--depth',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='Maximum depth level to generate (default: 3)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ProfileAwareGenerator()
    
    # Generate document
    print(f"Generating content about '{args.topic}'")
    print(f"Profile: {args.profile}, Max Depth: {args.depth}")
    
    document = generator.generate_document(
        topic=args.topic,
        profile=args.profile,
        max_depth=args.depth
    )
    
    # Output
    if args.output:
        document.save(args.output)
        print(f"Saved to {args.output}")
    else:
        print(document.render())


# Example usage:
# python main.py "quantum information conservation in cosmology" --profile scientific --depth 3
# python main.py "black holes" --profile popular_science --depth 2
```

---

## Step 6: Update Similarity Discriminator to Be Profile-Aware

### File to Modify

**`src/similarity_detector.py`** (from previous implementation plan)

### Changes Needed

The discriminator needs to adjust its evaluation based on profile:

```python
class SimilarityDiscriminator:
    def __init__(self):
        self.embeddings_store = []
        self.content_store = []
        self.metadata_store = []
        
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
    
    def evaluate(
        self,
        content: str,
        depth: int,
        profile: str,
        previous_depths: Optional[Dict[int, str]] = None
    ) -> dict:
        """
        Evaluate content with profile and depth awareness
        """
        embedding = self.get_embedding(content)
        
        # Get profile-specific thresholds
        thresholds = self.PROFILE_THRESHOLDS[profile]
        
        # Find most similar previous content
        similarities = self.compute_all_similarities(embedding)
        
        if not similarities:
            return {"action": "ACCEPT", "reason": "First content"}
        
        max_sim_score, max_sim_idx = max(similarities, key=lambda x: x[0])
        most_similar_content = self.content_store[max_sim_idx]
        most_similar_metadata = self.metadata_store[max_sim_idx]
        
        # Tier 1: Near-identical
        if max_sim_score >= thresholds['IDENTICAL']:
            return self.handle_tier1_identical(
                content,
                most_similar_content,
                max_sim_score,
                depth,
                profile,
                most_similar_metadata
            )
        
        # Tier 2: Very similar
        elif max_sim_score >= thresholds['VERY_SIMILAR']:
            return self.handle_tier2_very_similar(
                content,
                most_similar_content,
                max_sim_score,
                depth,
                profile
            )
        
        # Tier 3: Somewhat similar
        elif max_sim_score >= thresholds['SIMILAR']:
            return self.handle_tier3_somewhat_similar(
                content,
                most_similar_content,
                max_sim_score,
                profile
            )
        
        # Accept
        return {"action": "ACCEPT", "reason": "Sufficiently different"}
    
    def validate_profile_requirements(
        self,
        content: str,
        depth: int,
        profile: str
    ) -> dict:
        """
        Check if content meets profile-specific requirements
        E.g., scientific depth 2 must have min 3 equations
        """
        # Load profile config
        profile_config = load_profile_config(profile)
        depth_config = profile_config[f'depth_{depth}']
        
        violations = []
        
        # Check equation count
        if 'min_equations' in depth_config:
            eq_count = count_equations(content)
            if eq_count < depth_config['min_equations']:
                violations.append(
                    f"Too few equations: {eq_count} < {depth_config['min_equations']}"
                )
        
        # Check citation count
        if 'min_citations' in depth_config:
            cite_count = count_citations(content)
            if cite_count < depth_config['min_citations']:
                violations.append(
                    f"Too few citations: {cite_count} < {depth_config['min_citations']}"
                )
        
        # Check forbidden elements
        if 'forbidden_elements' in depth_config:
            for element in depth_config['forbidden_elements']:
                if self._contains_element(content, element):
                    violations.append(f"Contains forbidden element: {element}")
        
        return {
            'valid': len(violations) == 0,
            'violations': violations
        }
```

---

## Step 7: Testing Strategy

### Unit Tests to Write

**`tests/test_prompt_composer.py`**

```python
def test_prompt_composition_scientific_depth_1():
    """Test that scientific depth 1 prompt has correct structure"""
    composer = PromptComposer(...)
    context = GenerationContext(
        topic="quantum entanglement",
        depth=1,
        profile="scientific"
    )
    
    prompt = composer.compose_prompt(context)
    
    # Assert key elements present
    assert "formal_definitions" in prompt
    assert "mathematical_notation" in prompt
    assert "MUST NOT INCLUDE: oversimplified_analogies" in prompt
    assert "Minimum equations: 1" in prompt

def test_prompt_composition_popular_science_depth_1():
    """Test that popular science depth 1 is different from scientific"""
    composer = PromptComposer(...)
    
    sci_context = GenerationContext(topic="test", depth=1, profile="scientific")
    pop_context = GenerationContext(topic="test", depth=1, profile="popular_science")
    
    sci_prompt = composer.compose_prompt(sci_context)
    pop_prompt = composer.compose_prompt(pop_context)
    
    # Should be substantially different
    assert sci_prompt != pop_prompt
    assert "analogies" in pop_prompt
    assert "formal_definitions" in sci_prompt

def test_depth_2_includes_depth_1_context():
    """Test that depth 2 receives context from depth 1"""
    composer = PromptComposer(...)
    
    depth_1_content = "Previous content about holographic principle..."
    
    context = GenerationContext(
        topic="test",
        depth=2,
        profile="scientific",
        previous_depths={1: depth_1_content}
    )
    
    prompt = composer.compose_prompt(context)
    
    assert "CONTEXT FROM PREVIOUS DEPTHS" in prompt
    assert "holographic principle" in prompt
    assert "DO NOT restate" in prompt
```

### Integration Tests

**`tests/test_generation_pipeline.py`**

```python
def test_scientific_depth_3_different_from_depth_1():
    """End-to-end test that depth 3 doesn't repeat depth 1"""
    generator = ProfileAwareGenerator()
    
    doc = generator.generate_document(
        topic="quantum information",
        profile="scientific",
        max_depth=3
    )
    
    depth_1_content = doc.get_section(1)
    depth_3_content = doc.get_section(3)
    
    # Check similarity
    similarity = compute_similarity(depth_1_content, depth_3_content)
    
    # Should be below threshold
    assert similarity < 0.75, f"Depths too similar: {similarity}"

def test_popular_science_has_no_equations():
    """Test that popular science respects forbidden elements"""
    generator = ProfileAwareGenerator()
    
    doc = generator.generate_document(
        topic="quantum mechanics",
        profile="popular_science",
        max_depth=2
    )
    
    for depth in [1, 2]:
        content = doc.get_section(depth)
        eq_count = count_equations(content)
        assert eq_count <= 2, f"Too many equations in popular science depth {depth}"
```

---

## Step 8: Documentation

### Update README with Usage Examples

```markdown
# Profile and Depth-Aware Content Generation

## Usage

### Basic Usage

Generate scientific content at depth 3:
```bash
python main.py "quantum information in cosmology" --profile scientific --depth 3
```

Generate popular science content at depth 2:
```bash
python main.py "black holes" --profile popular_science --depth 2
```

### Available Profiles

- **scientific**: Technical, rigorous content for researchers
  - Includes equations, proofs, formal citations
  - All 3 depths available
  
- **popular_science**: Accessible content for general audience
  - Uses analogies, avoids jargon
  - Typically depths 1-2
  
- **educational**: Progressive learning content
  - Includes examples, exercises, learning objectives
  - All 3 depths available

### Depth Levels

- **Depth 1**: Introduces core concepts (WHAT)
- **Depth 2**: Explains mechanisms and formalisms (HOW)
- **Depth 3**: Explores open problems and research frontiers (UNKNOWNS)

### Examples

```bash
# Generate all depths for scientific audience
python main.py "entropy" --profile scientific --depth 3

# Generate only introductory content for popular audience
python main.py "entropy" --profile popular_science --depth 1

# Generate educational content with exercises
python main.py "entropy" --profile educational --depth 2
```
```

---

## Summary of Changes for Claude Code

### New Files to Create:
1. `configs/profiles.yaml` - Profile definitions with depth-specific configs
2. `configs/depth_framework.yaml` - Universal depth characteristics
3. `src/prompt_composer.py` - Prompt composition system
4. `tests/test_prompt_composer.py` - Unit tests
5. `tests/test_generation_pipeline.py` - Integration tests

### Files to Modify:
1. `src/generator.py` - Add ProfileAwareGenerator class
2. `src/similarity_detector.py` - Add profile-aware thresholds and validation
3. `cli.py` or `main.py` - Add --profile and --depth arguments
4. `README.md` - Add usage documentation

### Key Implementation Principles:
1. **Configuration over code**: Profiles and depth configs in YAML
2. **Context passing**: Previous depths inform next depth generation
3. **Explicit constraints**: Forbidden/required elements enforced
4. **Profile-aware validation**: Discriminator adjusts standards by profile
5. **Anti-repetition by design**: Each depth answers different question

### Expected Outcomes:
- ✅ Content appropriate for target audience (scientific vs. popular)
- ✅ Clear differentiation between depth levels
- ✅ Reduction in repetition (enforced by both prompts and discriminator)
- ✅ Measurable requirements (equations, citations, etc.)
- ✅ User control via --profile and --depth CLI parameters

---

**Ready for Claude Code to implement!** Should we proceed with this plan?