# Evaluator Profiles System

## Overview

The Evaluator Profiles System allows Musequill to assess content quality using domain-specific criteria. Different content types (scientific papers, tech reviews, creative writing, etc.) have different quality standards, and this system ensures evaluations are appropriate for the content domain.

## Available Profiles

### 1. Scientific (`scientific`)

**Best for:** Academic papers, research content, theoretical work

**Priorities:**
- Mathematical rigor (20 points) - Proofs, citations, formal reasoning
- Empirical grounding (15 points) - Testable predictions, data
- Citation context (15 points) - Proper attribution, related work
- Conceptual novelty (15 points) - Original contributions
- Semantic compression (15 points) - Tight, precise writing

**Critical Metrics:** Mathematical rigor, citation context, empirical grounding

**Quality Thresholds:**
- Excellent: 85+
- Good: 70+
- Acceptable: 55+
- Poor: 35+

**Example Topics:**
- "Quantum information theory foundations"
- "Novel approaches to protein folding prediction"
- "Mathematical models of neural computation"

---

### 2. Popular Science (`popular_science`)

**Best for:** Science communication, educational content for general audiences

**Priorities:**
- Empirical grounding (20 points) - Concrete examples, real-world connections
- Claim density (15 points) - Clear, specific assertions
- Citation context (15 points) - Source credibility
- Structural coherence (15 points) - Clear narrative flow
- Conceptual novelty (10 points) - Fresh perspectives

**Critical Metrics:** Claim density, citation context, structural coherence

**Quality Thresholds:**
- Excellent: 80+
- Good: 65+
- Acceptable: 50+
- Poor: 35+

**Example Topics:**
- "Review of the new M4 MacBook Pro"
- "Comprehensive guide to Kubernetes deployment"
- "Comparison of LLM inference frameworks"

---

### 4. Investment & Finance (`investment`)

**Best for:** Financial analysis, investment insights, market commentary

**Priorities:**
- Claim density (25 points) - Specific, quantifiable assertions
- Citation context (20 points) - Data sources, recent information
- Empirical grounding (20 points) - Historical data, metrics, ratios
- Mathematical rigor (10 points) - Sound calculations
- Conceptual novelty (10 points) - New perspectives

**Critical Metrics:** Claim density, citation context, empirical grounding

**Quality Thresholds:**
- Excellent: 85+
- Good: 70+
- Acceptable: 55+
- Poor: 40+

**Example Topics:**
- "Analysis of semiconductor sector valuations"
- "Tech stock growth prospects for 2025"
- "Risk assessment for emerging market bonds"

---

### 5. General (`general`)

**Best for:** Blog posts, tutorials, mixed-domain content, broad-audience writing

**Priorities:**
- Balanced across all dimensions
- Empirical grounding (15 points) - Examples and evidence
- Claim density (15 points) - Clear assertions
- Structural coherence (15 points) - Good flow
- Conceptual novelty (12 points) - Fresh ideas
- Semantic compression (12 points) - Efficient writing

**Critical Metrics:** Structural coherence

**Quality Thresholds:**
- Excellent: 80+
- Good: 65+
- Acceptable: 50+
- Poor: 35+

**Example Topics:**
- "Guide to effective remote work"
- "Understanding blockchain technology"
- "Introduction to machine learning"

---

### 6. Creative (`creative`)

**Best for:** Stories, narratives, creative non-fiction, artistic content

**Priorities:**
- Conceptual novelty (25 points) - Originality, fresh perspectives
- Structural coherence (25 points) - Narrative arc, flow
- Semantic compression (20 points) - Tight, purposeful prose
- Empirical grounding (10 points) - Vivid, concrete details
- Parsimony (10 points) - Every word matters

**Critical Metrics:** Conceptual novelty, structural coherence, semantic compression

**Quality Thresholds:**
- Excellent: 85+
- Good: 70+
- Acceptable: 55+
- Poor: 40+

**Example Topics:**
- "Write a sci-fi story about AI consciousness"
- "Personal essay on remote work culture"
- "Short story exploring themes of identity"

---

## Usage

### Command Line

```bash
# List available profiles
python client.py --list-profiles

# Use specific profile
python client.py "Your topic" --profile technology

# Examples
python client.py "Review of new GPU architecture" --profile technology
python client.py "Quantum entanglement explained" --profile popular_science
python client.py "Analysis of Tesla's financials" --profile investment
python client.py "A story about time travel" --profile creative
```

### Python API

```python
from src.workflow.orchestrator import WorkflowOrchestrator

# Create orchestrator with specific profile
orchestrator = WorkflowOrchestrator(
    evaluator_profile="technology"
)

# Run workflow
async for response in orchestrator.run_async(
    topic="Latest AI hardware developments",
    max_iterations=3,
    quality_threshold=65.0,
    evaluator_profile="technology"
):
    print(response.content)
```

### Programmatic Profile Selection

```python
from src.agents.evaluator import EvaluatorAgent
from src.agents.profiles import EvaluatorProfileFactory

# Create evaluator with profile
evaluator = EvaluatorAgent(
    profile="scientific"
)

# Get profile information
info = EvaluatorProfileFactory.get_profile_info("scientific")
print(f"Profile: {info['name']}")
print(f"Critical metrics: {info['critical_metrics']}")

# Evaluate content
result = await evaluator.evaluate(
    content="Your content here",
    previous_content=None,
    context={}
)

print(f"Score: {result.total_score}/100")
print(f"Tier: {result.tier}")
print(f"Profile used: {result.profile_name}")
```

---

## Metric Weights Comparison

| Metric                   | Scientific | Pop-Sci | Tech    | Investment | General | Creative |
| ------------------------ | ---------- | ------- | ------- | ---------- | ------- | -------- |
| **Conceptual Novelty**   | 15         | 10      | 15      | 10         | 12      | **25**   |
| **Claim Density**        | 10         | 15      | **20**  | **25**     | 15      | 5        |
| **Mathematical Rigor**   | **20**     | 5       | 3       | 10         | 5       | 0        |
| **Semantic Compression** | 15         | 10      | 10      | 8          | 12      | **20**   |
| **Citation Context**     | 15         | 15      | 10      | **20**     | 10      | 0        |
| **Empirical Grounding**  | 15         | **20**  | **25**  | **20**     | 15      | 10       |
| **Structural Coherence** | 5          | 15      | 10      | 5          | 15      | **25**   |
| **Notation Consistency** | 3          | 3       | 2       | 1          | 5       | 5        |
| **Figure Utility**       | 1          | 5       | 3       | 1          | 6       | 0        |
| **Parsimony**            | 1          | 2       | 2       | 0          | 5       | 10       |
| **TOTAL**                | **100**    | **100** | **100** | **100**    | **100** | **100**  |

**Bold** = Highest weight metrics for each profile

---

## Profile Selection Guide

### Choose **Scientific** if:
- ✅ Content requires formal proofs or derivations
- ✅ Target audience is researchers/academics
- ✅ Mathematical rigor is essential
- ✅ Original theoretical contributions expected
- ❌ Avoid for: Accessible explanations, tutorials

### Choose **Popular Science** if:
- ✅ Explaining science to general audiences
- ✅ Accessibility more important than rigor
- ✅ Real-world examples are key
- ✅ Engaging narrative matters
- ❌ Avoid for: Academic papers, technical specs

### Choose **Technology** if:
- ✅ Product reviews or comparisons
- ✅ Technical tutorials or guides
- ✅ Benchmarks and test results expected
- ✅ Practical, actionable insights needed
- ❌ Avoid for: Pure research, creative writing

### Choose **Investment** if:
- ✅ Financial analysis or recommendations
- ✅ Data-driven market insights
- ✅ Quantitative metrics are critical
- ✅ Source credibility is paramount
- ❌ Avoid for: Qualitative analysis without data

### Choose **General** if:
- ✅ Mixed domain or unclear content type
- ✅ Broad audience with varied backgrounds
- ✅ Balanced quality across dimensions
- ✅ Standard blog post or article
- ❌ Avoid for: Highly specialized content

### Choose **Creative** if:
- ✅ Storytelling or narrative content
- ✅ Originality and voice are paramount
- ✅ Artistic expression valued over facts
- ✅ Every word should contribute to story
- ❌ Avoid for: Factual, informative content

---

## Customizing Profiles

### Create a Custom Profile

```python
# Add to src/agents/profiles.py

@staticmethod
def _custom_profile() -> Dict[str, Any]:
    """
    Profile for your specific domain.
    
    Priorities:
    - Your key dimensions
    """
    return {
        "name": "Custom Domain",
        "description": "Description of your domain",
        "metric_config": {
            'conceptual_novelty': {
                'weight': 15,
                'min_threshold': 6,
                'optimal': 15,
                'calculation': 'CNR >= 40%'
            },
            # ... configure all 10 metrics
        },
        "tier_thresholds": {
            'excellent': 85,
            'good': 70,
            'acceptable': 55,
            'poor': 40,
            'unacceptable': 0
        },
        "critical_metrics": [
            'your_critical_metric_1',
            'your_critical_metric_2'
        ]
    }
```

### Add to Factory

```python
profiles = {
    # ... existing profiles ...
    "custom": EvaluatorProfileFactory._custom_profile()
}
```

### Update Type Hints

```python
ProfileType = Literal[
    "scientific",
    "popular_science", 
    "technology",
    "investment",
    "general",
    "creative",
    "custom"  # Add your profile
]
```

---

## Environment Variables

```bash
# Override quality threshold for all profiles
QUALITY_THRESHOLD=70.0

# Maximum refinement iterations per agent
MAX_REFINEMENT_ITERATIONS=3

# Maximum conversation iterations
MAX_ITERATIONS=25
```

---

## Integration with Workflow

The evaluator profile affects:

1. **Generator Evaluation**: Content quality assessment after generation
2. **Discriminator Evaluation**: Response quality assessment after analysis
3. **Revision Decisions**: Whether content needs refinement
4. **Final Assessment**: Summarizer uses profile-appropriate criteria

### Workflow Flow with Profiles

```
User Request
    ↓
Orchestrator (creates evaluators with profile)
    ↓
Generator → Generator Evaluator (profile-based scoring)
    ↓ (if quality < threshold)
Generator Revision → Generator Evaluator
    ↓ (best content selected)
Discriminator → Discriminator Evaluator (profile-based scoring)
    ↓ (if quality < threshold)
Discriminator Revision → Discriminator Evaluator
    ↓ (continue or summarize)
Summarizer (final quality assessment)
```

---

## Best Practices

### 1. Profile Selection
- Choose the profile that best matches your **content type**, not your audience
- When in doubt, use `general` profile
- For mixed content, choose the dominant content type

### 2. Quality Thresholds
- Start with profile defaults
- Adjust `QUALITY_THRESHOLD` based on your needs
- Higher thresholds = more revisions = better quality = slower

### 3. Iteration Limits
- `MAX_REFINEMENT_ITERATIONS`: Controls revision rounds per agent
- `MAX_ITERATIONS`: Controls conversation length
- Balance quality vs. speed

### 4. Profile Customization
- Create custom profiles for specialized domains
- Weight metrics based on what matters most
- Set critical metrics for must-have qualities

---

## Troubleshooting

### Issue: Content fails evaluation repeatedly
**Solution:** 
- Check if profile matches content type
- Review critical metrics - are they too strict?
- Lower `QUALITY_THRESHOLD` temporarily
- Increase `MAX_REFINEMENT_ITERATIONS`

### Issue: Wrong profile being used
**Solution:**
- Check CLI argument: `--profile technology`
- Verify server receives profile in request
- Check orchestrator initialization logs

### Issue: Profile not available
**Solution:**
- Run `python client.py --list-profiles`
- Check spelling (use underscores: `popular_science`)
- Verify profile is in `EvaluatorProfileFactory`

---

## Examples

### Scientific Paper Evaluation

```bash
python client.py "Novel quantum error correction codes" \
    --profile scientific \
    --max-iterations 5
```

**Expected Behavior:**
- High emphasis on mathematical rigor
- Requires proofs for claims
- Citations for established results
- Testable predictions valued

---

### Tech Product Review

```bash
python client.py "Comprehensive review of M4 MacBook Pro" \
    --profile technology \
    --max-iterations 3
```

**Expected Behavior:**
- Emphasis on benchmarks and tests
- Specific performance claims
- Real-world usage examples
- Practical recommendations

---

### Popular Science Article

```bash
python client.py "How does quantum entanglement work?" \
    --profile popular_science \
    --max-iterations 4
```

**Expected Behavior:**
- Accessible explanations
- Concrete analogies
- Real-world examples
- Clear narrative flow
- Less emphasis on formal rigor

---

## Future Enhancements

Potential additions to the profile system:

1. **Domain-Specific Metrics**: Add metrics tailored to specific domains
2. **Adaptive Thresholds**: Adjust thresholds based on content length
3. **Profile Mixing**: Combine aspects of multiple profiles
4. **User Profiles**: Save user-specific profile preferences
5. **A/B Testing**: Compare output quality across profiles
6. **Profile Learning**: Machine learning-based profile optimization

---

## Summary

The Evaluator Profiles System provides:

✅ Domain-appropriate quality assessment  
✅ Flexible metric weighting  
✅ Content-type specific thresholds  
✅ Critical metric enforcement  
✅ Easy CLI integration  
✅ Extensible architecture  

Choose the right profile, and Musequill will generate and evaluate content according to the standards that matter for your domain.: 35+

**Example Topics:**
- "How CRISPR gene editing works"
- "Climate change explained for everyone"
- "The science behind vaccines"

---

### 3. Technology (`technology`)

**Best for:** Tech reviews, product analysis, tutorials, technical deep-dives

**Priorities:**
- Empirical grounding (25 points) - Real examples, benchmarks, tests
- Claim density (20 points) - Specific technical assertions
- Conceptual novelty (15 points) - New insights, perspectives
- Semantic compression (10 points) - Efficient writing
- Structural coherence (10 points) - Clear organization

**Critical Metrics:** Claim density, empirical grounding

**Quality Thresholds:**
- Excellent: 80+
- Good: 65+
- Acceptable: 50+
- Poor