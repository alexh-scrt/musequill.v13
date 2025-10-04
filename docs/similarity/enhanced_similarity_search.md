# Implementation Plan: Enhanced Similarity Detection System

## Context & Problem Statement

**Current Issue**: The document generation system produces excessive repetition, particularly visible in:
- Tables that restate the same concepts with minimal variation
- Paragraphs across different depth levels that repeat core ideas without adding new insights
- Follow-up questions answered with previously-stated content

**User Requirements**:
1. **Tolerance**: Middle ground - not too strict, not too loose
2. **Real-time checking**: Detect similarity immediately after text generation
3. **Context-aware decisions**: 
   - Pure text similarity for near-identical content (e.g., repetitive tables) → auto-skip
   - Semantic analysis for moderately similar content (0.75 similarity) → understand WHY similar
4. **Human-in-the-loop**: Flag for review now, potential post-processing agent later

## Current Implementation Analysis

**What we need to understand**:
1. Where in the codebase is text generation happening?
   - Is it generating section-by-section, paragraph-by-paragraph, or all at once?
   - Are there hooks where we can insert similarity checks?

2. What's the current data flow?
   - How is content structured (sections → paragraphs → sentences)?
   - Where is content stored before final output?

3. Are embeddings already being generated?
   - For project knowledge search, embeddings must exist
   - Can we reuse that infrastructure?

4. What's the generation loop?
   - Sequential generation (generate → check → continue)?
   - Batch generation (generate all → post-process)?

## Proposed Architecture

### Three-Tier Detection System

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1: IDENTICAL (similarity ≥ 0.90)                  │
│  Action: AUTO-SKIP with explanation                      │
│  Logic: Pure text similarity, no deep analysis needed    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Tier 2: VERY SIMILAR (0.75 ≤ similarity < 0.90)       │
│  Action: ANALYZE WHY + FLAG FOR REVIEW                   │
│  Logic: Semantic analysis - what concepts overlap?       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Tier 3: SOMEWHAT SIMILAR (0.60 ≤ similarity < 0.75)   │
│  Action: CHECK INFORMATION GAIN                          │
│  Logic: Does new content add novel concepts/insights?    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  LOW SIMILARITY (similarity < 0.60)                      │
│  Action: ACCEPT                                          │
└─────────────────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: Create Core Similarity Detection Module

**New file**: `similarity_detector.py`

**Purpose**: Central module for all similarity checking logic

**Key components**:

```
RepetitionDetector class:
├── __init__()
│   ├── Initialize empty stores for embeddings, content, metadata
│   ├── Set threshold constants (0.90, 0.75, 0.60)
│   └── Initialize embedding client (reuse existing project knowledge embedding service)
│
├── analyze_content(text, metadata) → Decision
│   ├── Compute embedding for new text
│   ├── Find most similar previous content
│   ├── Route to appropriate tier handler
│   └── Return: {action: SKIP|FLAG|ACCEPT, reason: str, details: dict}
│
├── handle_tier1_identical(text, similar_content, score) → Decision
├── handle_tier2_very_similar(text, similar_content, score) → Decision
├── handle_tier3_somewhat_similar(text, similar_content, score) → Decision
│
├── store_content(text, embedding, metadata)
│   └── Add to history for future comparisons
│
└── get_embedding(text) → vector
    └── Reuse existing embedding service from project_knowledge_search
```

**Why these components**:
- **Single responsibility**: Each tier handler focuses on one similarity level
- **Metadata tracking**: Need to know what section/depth content came from for context-aware decisions
- **Reuse embeddings**: Don't reinvent - leverage existing embedding infrastructure
- **Clear decision output**: Downstream code needs to know what action to take

### Step 2: Integrate with Content Generation Flow

**Files to modify**: (Need to identify from codebase)
- Likely: `content_generator.py` or similar main generation orchestrator
- Possibly: Individual section generators (intro, analysis, synthesis, etc.)

**Integration points**:

```
Content Generation Loop (pseudo-code):

for each section in document:
    for each content_unit in section:
        # Generate content
        generated_text = generate_content(prompt, context)
        
        # NEW: Check similarity
        decision = repetition_detector.analyze_content(
            text=generated_text,
            metadata={
                'section': section.name,
                'depth': current_depth,
                'content_type': content_unit.type  # e.g., 'table', 'paragraph', 'synthesis'
            }
        )
        
        # Act on decision
        if decision['action'] == 'SKIP':
            log_skip(decision)
            continue  # Don't add to output
            
        elif decision['action'] == 'FLAG':
            log_flag(decision)
            add_to_output_with_warning(generated_text, decision)
            repetition_detector.store_content(generated_text, ...)
            
        elif decision['action'] == 'ACCEPT':
            add_to_output(generated_text)
            repetition_detector.store_content(generated_text, ...)
```

**Why this integration**:
- **Real-time checking**: Happens immediately after generation
- **Non-blocking for FLAGS**: Content still included but marked for review
- **History building**: Each accepted/flagged content becomes part of comparison baseline

### Step 3: Implement Tier 1 - Identical Detection

**Purpose**: Catch near-duplicate content like repetitive tables

**Logic**:
```
handle_tier1_identical(text, similar_content, score, metadata, similar_metadata):
    
    # Check if it's a structural element that's expected to be similar
    if metadata['content_type'] in ['table_header', 'equation']:
        # Tables often have similar structure - that's OK
        # But check if the TABLE CONTENT is identical
        content_similarity = compare_table_content(text, similar_content)
        if content_similarity > 0.95:
            return SKIP (identical table content)
    
    # For prose/paragraphs
    if score >= 0.95:
        # Almost word-for-word identical
        return {
            'action': 'SKIP',
            'reason': f'Near-identical to previous content (similarity: {score:.2f})',
            'similar_to': {
                'section': similar_metadata['section'],
                'depth': similar_metadata['depth'],
                'preview': similar_content[:200]
            }
        }
    
    elif score >= 0.90:
        # Very similar but not identical - might be acceptable if different depth
        if metadata['depth'] != similar_metadata['depth']:
            # Downgrade to Tier 2 analysis
            return handle_tier2_very_similar(...)
        else:
            # Same depth, very similar - probably redundant
            return SKIP
```

**Why this logic**:
- **Structural awareness**: Tables/equations naturally have similar structure
- **Content vs. structure**: Separate format similarity from information similarity
- **Depth-aware**: Same concept can appear at different depths if adding new insight
- **Clear skip reason**: Downstream logging needs to know why content was rejected

### Step 4: Implement Tier 2 - Semantic Analysis

**Purpose**: Understand WHY content is similar and whether similarity is justified

**Logic**:
```
handle_tier2_very_similar(text, similar_content, score, metadata, similar_metadata):
    
    # Extract key concepts from both pieces of content
    new_concepts = extract_concepts(text)
    old_concepts = extract_concepts(similar_content)
    
    # Calculate concept overlap
    overlap = new_concepts.intersection(old_concepts)
    novel = new_concepts - old_concepts
    
    novelty_ratio = len(novel) / len(new_concepts) if new_concepts else 0
    
    # Analyze the nature of similarity
    similarity_analysis = {
        'shared_concepts': list(overlap),
        'novel_concepts': list(novel),
        'novelty_ratio': novelty_ratio,
        'is_different_perspective': check_different_perspective(text, similar_content)
    }
    
    # Decision logic
    if novelty_ratio >= 0.4 or similarity_analysis['is_different_perspective']:
        # >= 40% new concepts OR different angle on same concepts
        return {
            'action': 'FLAG',
            'reason': 'Similar but introduces new concepts or perspective',
            'similarity_score': score,
            'analysis': similarity_analysis,
            'recommendation': 'REVIEW - May be acceptable if truly adding insight'
        }
    else:
        # < 40% new concepts and same perspective
        return {
            'action': 'SKIP',
            'reason': 'Too similar without sufficient new information',
            'similarity_score': score,
            'analysis': similarity_analysis
        }
```

**Why this logic**:
- **Concept extraction**: Need to identify what ideas are being discussed
- **Novelty threshold**: 40% is a reasonable middle ground (can tune based on feedback)
- **Perspective check**: Same concepts can be valuable if discussed from different angle
- **Detailed analysis**: Human reviewer needs context to make final decision

### Step 5: Implement Tier 3 - Information Gain Check

**Purpose**: For moderately similar content, ensure it adds value

**Logic**:
```
handle_tier3_somewhat_similar(text, similar_content, score, metadata):
    
    # At this similarity level (0.60-0.75), some overlap is expected
    # Focus on: Does this add NEW information?
    
    information_gain = calculate_information_gain(text, similar_content)
    
    if information_gain['has_new_examples']:
        return ACCEPT (new examples add value)
    
    if information_gain['has_new_equations']:
        return ACCEPT (new mathematical formulation)
    
    if information_gain['has_new_references']:
        return ACCEPT (cites new sources)
    
    if information_gain['deeper_explanation']:
        return ACCEPT (provides more detail/rigor)
    
    # If no clear new information
    return {
        'action': 'FLAG',
        'reason': 'Moderate similarity without clear information gain',
        'similarity_score': score,
        'info_gain_analysis': information_gain,
        'recommendation': 'REVIEW - Consider if this depth of coverage is needed'
    }
```

**Why this logic**:
- **Permissive by default**: At 0.60-0.75 similarity, some overlap is natural
- **Multiple information gain signals**: Different ways content can add value
- **Flag for review rather than skip**: Human judgment needed at this level

### Step 6: Concept Extraction Implementation

**Purpose**: Identify key ideas in physics/cosmology text

**Approaches** (in order of implementation complexity):

**Option A: Pattern-based (easiest, implement first)**:
```python
PHYSICS_CONCEPTS = {
    'quantum_info': ['von Neumann entropy', 'density matrix', 'qubit', 'entanglement', 
                     'unitarity', 'quantum information'],
    'cosmology': ['FLRW', 'de Sitter', 'horizon', 'Hubble', 'inflation', 
                  'causal patch', 'cosmological'],
    'quantum_gravity': ['Wheeler-DeWitt', 'holographic', 'Planck length', 
                        'covariant entropy bound', 'AdS/CFT'],
    'math_objects': ['Hilbert space', 'wavefunction', 'operator', 
                     'diffeomorphism', 'gauge invariant']
}

def extract_concepts(text):
    concepts = set()
    text_lower = text.lower()
    
    for category, terms in PHYSICS_CONCEPTS.items():
        for term in terms:
            if term.lower() in text_lower:
                concepts.add(term)
    
    # Also extract equation patterns
    equation_patterns = re.findall(r'S\s*=\s*.*?|H\s*=\s*.*?|\\\[.*?\\\]', text)
    if equation_patterns:
        concepts.add('__EQUATIONS__')
    
    return concepts
```

**Option B: LLM-based extraction (more accurate, implement later)**:
```python
def extract_concepts_llm(text):
    prompt = f"""
    Extract the key physics/cosmology concepts from this text.
    List only the core technical terms and ideas (e.g., "unitarity", "holographic principle").
    
    Text: {text}
    
    Concepts (comma-separated):
    """
    
    concepts = call_llm(prompt).split(',')
    return set(c.strip() for c in concepts)
```

**Why this approach**:
- **Start simple**: Pattern matching is fast and good enough for first iteration
- **Upgrade path**: Can swap in LLM extraction without changing interface
- **Domain-specific**: Physics vocabulary is relatively constrained

### Step 7: Information Gain Calculation

**Purpose**: Determine if content adds value beyond similarity score

```python
def calculate_information_gain(new_text, old_text):
    """
    Returns dict of signals indicating new information
    """
    
    return {
        'has_new_examples': check_new_examples(new_text, old_text),
        'has_new_equations': check_new_equations(new_text, old_text),
        'has_new_references': check_new_references(new_text, old_text),
        'deeper_explanation': check_depth_increase(new_text, old_text),
        'different_formalism': check_different_formalism(new_text, old_text)
    }

def check_new_examples(new_text, old_text):
    # Look for phrases like "for example", "consider", "suppose"
    # Check if the examples are different
    new_examples = extract_examples(new_text)
    old_examples = extract_examples(old_text)
    return len(new_examples - old_examples) > 0

def check_new_equations(new_text, old_text):
    new_eqs = extract_equations(new_text)
    old_eqs = extract_equations(old_text)
    return len(new_eqs - old_eqs) > 0

def check_new_references(new_text, old_text):
    # Look for citations like "Bousso 1999", "Hartle & Hawking"
    new_refs = extract_references(new_text)
    old_refs = extract_references(old_text)
    return len(new_refs - old_refs) > 0
```

**Why these signals**:
- **Examples**: New examples indicate concrete application of concepts
- **Equations**: Mathematical formulation is distinct from conceptual explanation
- **References**: New citations suggest additional perspectives/evidence
- **Composable**: Multiple weak signals combine to strong evidence of value

### Step 8: Logging and Review Infrastructure

**Purpose**: Track decisions for human review and system improvement

**New file**: `repetition_log.py`

```python
class RepetitionLog:
    """
    Stores decisions for:
    1. Human review (flagged content)
    2. System improvement (tune thresholds)
    3. Debugging (why was content skipped?)
    """
    
    def log_decision(self, decision, original_text, metadata):
        entry = {
            'timestamp': datetime.now(),
            'action': decision['action'],
            'reason': decision['reason'],
            'similarity_score': decision.get('similarity_score'),
            'metadata': metadata,
            'text_preview': original_text[:200],
            'full_decision': decision
        }
        
        # Append to session log
        self.session_log.append(entry)
        
        # If flagged, add to review queue
        if decision['action'] == 'FLAG':
            self.review_queue.append(entry)
    
    def export_review_queue(self):
        """Export flagged content for human review"""
        return json.dumps(self.review_queue, indent=2)
    
    def get_statistics(self):
        """
        For tuning: How many SKIPs vs FLAGs vs ACCEPTs?
        Distribution of similarity scores?
        """
        return {
            'total_checks': len(self.session_log),
            'skipped': sum(1 for e in self.session_log if e['action'] == 'SKIP'),
            'flagged': sum(1 for e in self.session_log if e['action'] == 'FLAG'),
            'accepted': sum(1 for e in self.session_log if e['action'] == 'ACCEPT'),
            'avg_similarity': np.mean([e['similarity_score'] for e in self.session_log if e.get('similarity_score')])
        }
```

**Why logging**:
- **Transparency**: User can see why content was rejected
- **Tunability**: Statistics help adjust thresholds
- **Accountability**: Review queue enables human oversight

### Step 9: Integration with Existing Embedding Service

**Files to investigate**:
- How does `project_knowledge_search` currently compute embeddings?
- Is there a shared embedding service/client?
- What embedding model is being used?

**Goal**: Reuse existing infrastructure rather than duplicate

```python
# In similarity_detector.py

from existing_embedding_service import get_embedding_client

class RepetitionDetector:
    def __init__(self):
        # Reuse existing client
        self.embedding_client = get_embedding_client()
        ...
    
    def get_embedding(self, text):
        # Use same embedding model as project knowledge
        return self.embedding_client.embed(text)
```

**Why reuse**:
- **Consistency**: Same embedding space as project knowledge search
- **Efficiency**: Don't duplicate API calls or model loading
- **Maintainability**: One place to update embedding logic

### Step 10: Configuration and Tuning

**New file**: `similarity_config.py`

```python
class SimilarityConfig:
    """
    Centralized configuration for easy tuning
    """
    
    # Thresholds
    IDENTICAL_THRESHOLD = 0.90
    VERY_SIMILAR_THRESHOLD = 0.75
    SIMILAR_THRESHOLD = 0.60
    
    # Information gain requirements
    MIN_NOVELTY_RATIO = 0.40  # 40% new concepts
    
    # What content types to check
    CHECK_TABLES = True
    CHECK_PARAGRAPHS = True
    CHECK_EQUATIONS = False  # Equations often similar by nature
    
    # Behavioral flags
    AUTO_SKIP_IDENTICAL = True
    ALWAYS_FLAG_NEVER_AUTO_SKIP = False  # Override for safety
    
    @classmethod
    def from_file(cls, path):
        """Load from config file for easy experimentation"""
        ...
```

**Why configuration**:
- **Experimentation**: Easy to test different threshold values
- **User control**: Can adjust strictness without code changes
- **Per-document settings**: Different document types might need different thresholds

## Testing Strategy

### Unit Tests

**Test each tier independently**:

```python
def test_tier1_identical():
    detector = RepetitionDetector()
    
    # Store initial content
    initial = "The holographic principle states that entropy is proportional to area."
    detector.store_content(initial, ...)
    
    # Test near-duplicate
    duplicate = "The holographic principle states entropy is proportional to area."
    decision = detector.analyze_content(duplicate, ...)
    
    assert decision['action'] == 'SKIP'
    assert decision['similarity_score'] >= 0.90

def test_tier2_similar_but_novel():
    detector = RepetitionDetector()
    
    initial = "The holographic principle states that entropy is proportional to area."
    detector.store_content(initial, ...)
    
    # Similar topic but adds new concept
    extended = "The holographic principle states that entropy is proportional to area. Applying this to black holes yields the Bekenstein-Hawking formula."
    decision = detector.analyze_content(extended, ...)
    
    # Should flag for review, not skip (adds Bekenstein-Hawking)
    assert decision['action'] == 'FLAG'
    assert 'Bekenstein-Hawking' in decision['analysis']['novel_concepts']
```

### Integration Tests

**Test full generation flow**:

```python
def test_document_generation_with_similarity_check():
    """
    Generate a multi-section document and verify:
    1. Repetitive sections are skipped
    2. Novel sections are accepted
    3. Review queue contains expected flags
    """
    ...
```

### Manual Review Tests

**Create test corpus**:
- Take the problematic document from user
- Run similarity detection
- Manually verify each SKIP and FLAG decision
- Tune thresholds based on false positives/negatives

## Rollout Plan

### Phase 1: Core Implementation (Week 1)
1. Create `similarity_detector.py` with basic structure
2. Implement Tier 1 (identical detection) only
3. Integrate with one content generation point (e.g., paragraph generation)
4. Test on sample documents

### Phase 2: Semantic Analysis (Week 2)
1. Implement concept extraction (pattern-based)
2. Implement Tier 2 (semantic analysis)
3. Add logging infrastructure
4. Test on full documents

### Phase 3: Information Gain (Week 3)
1. Implement Tier 3 (information gain)
2. Add all information gain signals
3. Create configuration system
4. Comprehensive testing

### Phase 4: Refinement (Week 4)
1. Analyze logs from real usage
2. Tune thresholds based on false positive/negative rates
3. Consider LLM-based concept extraction if pattern-matching insufficient
4. Add post-processing agent for review queue

## Success Metrics

**How to know if this is working**:

1. **Quantitative**:
   - Reduction in similarity scores between consecutive sections
   - Number of SKIPs per document (target: 2-5 for a document like the example)
   - Number of FLAGs per document (target: 3-8)
   - Average novelty ratio for accepted content (target: >0.5)

2. **Qualitative**:
   - Human review of flagged content confirms decisions
   - Generated documents subjectively "feel" less repetitive
   - Tables no longer restate identical information

3. **Performance**:
   - Similarity checking adds <10% to total generation time
   - No degradation in content quality (similarity checking doesn't suppress valid reinforcement)

## Questions for Codebase Investigation

Before Claude Code starts implementation, these questions need answers:

1. **Where is the main content generation orchestrator?**
   - What file/function controls the overall document generation flow?

2. **What's the content generation granularity?**
   - Are sections generated independently or all at once?
   - Can we insert hooks between generations?

3. **Where are embeddings currently computed?**
   - File/class responsible for embedding operations
   - What embedding model is being used?

4. **What's the data structure for document content?**
   - How is the document represented in memory?
   - Is there a Document class with Section objects?

5. **Are there existing quality checks or filters?**
   - If so, where? Can we add similarity checking to existing pipeline?

6. **What's the testing framework?**
   - pytest? unittest? Need to match existing patterns

7. **Configuration management?**
   - Is there an existing config system (yaml, json, env vars)?

---

**Next Step**: Claude Code should:
1. Analyze the codebase to answer the questions above
2. Create a detailed technical specification based on this plan
3. Implement Phase 1 (Core + Tier 1) first
4. Provide sample output showing SKIPs with reasons
5. Iterate based on feedback