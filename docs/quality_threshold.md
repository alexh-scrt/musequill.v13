Here's a practical quality threshold system for your editor-generator agent loop:

## Quality Threshold Scale (0-100 points)

### Tier System

**Tier 1: Unacceptable (0-39 points)** - Reject, major revision required
**Tier 2: Poor (40-59 points)** - Significant issues, revision needed
**Tier 3: Acceptable (60-74 points)** - Publishable with minor revisions
**Tier 4: Good (75-89 points)** - Strong paper, ready for peer review
**Tier 5: Excellent (90-100 points)** - High-impact contribution

### Point Allocation by Metric

| Metric                      | Weight | Calculation                             | Min Pass Score |
| --------------------------- | ------ | --------------------------------------- | -------------- |
| **Conceptual Novelty Rate** | 15 pts | CNR × 15 (if CNR ≥ 40%, else CNR × 7.5) | 6/15           |
| **Claim Density**           | 10 pts | min(CD / 0.008, 1) × 10                 | 5/10           |
| **Mathematical Rigor**      | 15 pts | MRI × 15                                | 9/15           |
| **Semantic Compression**    | 15 pts | max(0, (5 - SCR) / 4) × 15              | 7/15           |
| **Citation & Context**      | 10 pts | Presence score + Novelty clarity        | 5/10           |
| **Empirical Grounding**     | 10 pts | (Predictions/3) × 10                    | 4/10           |
| **Structural Coherence**    | 10 pts | (1 - circular_ratio) × 10               | 7/10           |
| **Notation Consistency**    | 5 pts  | NCR × 5                                 | 4/5            |
| **Figure/Equation Utility** | 5 pts  | Information-adding ratio × 5            | 3/5            |
| **Parsimony**               | 5 pts  | PS × 5                                  | 3/5            |

### Detailed Scoring Rules

**1. Conceptual Novelty Rate (15 points)**
```
if CNR ≥ 40%: score = min(CNR/40 * 15, 15)
elif CNR ≥ 25%: score = 6 + (CNR-25)/15 * 6
else: score = CNR/25 * 6
```
- 15 pts: 40%+ new concepts per section
- 12 pts: 35-40% new concepts
- 9 pts: 30-35% new concepts
- 6 pts: 25-30% new concepts (minimum acceptable)
- <6 pts: Unacceptable redundancy

**2. Claim Density (10 points)**
```
claims_per_100 = total_claims / (word_count / 100)
score = min(claims_per_100 / 0.8, 1.0) * 10
```
- 10 pts: ≥0.8 claims/100 words
- 7 pts: 0.6-0.8 claims/100 words
- 5 pts: 0.4-0.6 claims/100 words (minimum)
- <5 pts: Too much fluff

**3. Mathematical Rigor (15 points)**
```
proved_theorems = theorems_with_proofs / total_theorems
cited_claims = properly_cited / total_known_results
score = (proved_theorems * 0.6 + cited_claims * 0.4) * 15
```
- 15 pts: 100% proofs, 100% citations
- 12 pts: 90% proofs, 85% citations
- 9 pts: 80% proofs, 70% citations (minimum)
- <9 pts: Insufficient rigor

**4. Semantic Compression Ratio (15 points)**
```
# Lower SCR is better (less redundancy)
if SCR ≤ 2.0: score = 15
elif SCR ≤ 3.0: score = 12
elif SCR ≤ 4.0: score = 9
elif SCR ≤ 5.0: score = 7 (minimum)
else: score = max(0, 15 - SCR * 2)
```
- 15 pts: SCR ≤ 2:1 (tight writing)
- 12 pts: SCR 2-3:1 (acceptable repetition)
- 7 pts: SCR 4-5:1 (maximum acceptable)
- <7 pts: Excessive redundancy

**5. Citation Density & Novelty Context (10 points)**
```
cite_density = citations_per_10_pages / 20  # normalized to 20 cites/10pg
novelty_clarity = has_related_work_section * 0.3 + 
                  has_explicit_novelty_claims * 0.3 +
                  cites_directly_relevant * 0.4
score = (cite_density * 0.5 + novelty_clarity * 0.5) * 10
```
- 10 pts: 15-40 cites/10pg + clear novelty statement + related work
- 7 pts: 10-15 cites + some context
- 5 pts: 5-10 cites + minimal context (minimum)
- 0 pts: No citations for established field claims

**6. Empirical Grounding (10 points)**
```
testable_predictions = count(falsifiable_claims)
experimental_connections = count(references_to_experiments)
philosophical_ratio = 1 - (pure_tautologies / total_arguments)
score = min(testable_predictions/3, 1) * 5 + 
        min(experimental_connections/2, 1) * 3 +
        philosophical_ratio * 2
```
- 10 pts: 3+ predictions, 2+ experimental links, no tautology
- 7 pts: 2 predictions, 1 experimental link
- 4 pts: 1 prediction OR 1 experimental link (minimum)
- <4 pts: Purely circular or unfalsifiable

**7. Structural Coherence (10 points)**
```
forward_refs = count(references_to_later_sections)
backward_refs = count(references_to_earlier_sections)
circular_deps = count(circular_reasoning_cycles)
score = max(0, 10 - circular_deps * 2 - forward_refs * 0.5)
```
- 10 pts: Linear flow, no cycles, <5% forward refs
- 8 pts: <2 circular patterns, <10% forward refs
- 7 pts: <3 circular patterns, <15% forward refs (minimum)
- <7 pts: Structural problems

**8. Notation Consistency (5 points)**
```
score = (1 - redefinitions / total_symbols) * 5
```
- 5 pts: 100% consistency
- 4 pts: 95-99% consistency (minimum)
- <4 pts: Confusing notation

**9. Figure/Equation Information Density (5 points)**
```
informative_figures = figures_adding_content / total_figures
informative_equations = equations_adding_content / total_equations
score = (informative_figures * 0.4 + informative_equations * 0.6) * 5
```
- 5 pts: 100% figures/equations add information
- 4 pts: 90%+ add information
- 3 pts: 80%+ add information (minimum)
- <3 pts: Decorative or redundant

**10. Parsimony (5 points)**
```
score = (essential_assumptions / total_assumptions) * 5
```
- 5 pts: 100% assumptions necessary
- 4 pts: 90%+ necessary
- 3 pts: 70%+ necessary (minimum)
- <3 pts: Overcomplicated

## Agent Workflow Thresholds

### Editor Agent Decision Logic

```python
def evaluate_paper(paper):
    scores = calculate_all_metrics(paper)
    total_score = sum(scores.values())
    
    # Critical failures (auto-reject regardless of total)
    if scores['Mathematical Rigor'] < 9:
        return "REJECT: Insufficient proofs"
    if scores['Semantic Compression'] < 7:
        return "REJECT: Excessive redundancy"
    if scores['Conceptual Novelty'] < 6:
        return "REJECT: Insufficient novel content"
    
    # Overall assessment
    if total_score >= 75:
        return "ACCEPT: Minor revisions"
    elif total_score >= 60:
        return "REVISE: Address specific issues"
    else:
        return "REJECT: Major revision required"
```

### Generator Agent Feedback Format

```json
{
  "overall_score": 45,
  "tier": "Poor - Significant Revision Needed",
  "critical_issues": [
    "Semantic Compression Ratio: 15:1 (need <5:1)",
    "Conceptual Novelty: 8% (need 25%+ minimum)"
  ],
  "priority_actions": [
    "Consolidate repetitive sections 3-27 into single section",
    "Remove 85% of redundant explanations",
    "Add 15-20 citations to existing literature",
    "Provide proofs for Theorems 2, 4, 7"
  ],
  "section_scores": {
    "section_1": {"CNR": 100%, "keep": true},
    "section_2": {"CNR": 12%, "action": "merge_with_1"},
    "section_3": {"CNR": 8%, "action": "delete"}
  }
}
```

## Your Document's Likely Scores

| Metric    | Your Score | Points Earned | Threshold Met?   |
| --------- | ---------- | ------------- | ---------------- |
| CNR       | 8%         | 2.4/15        | ❌ (need 6)       |
| CD        | 0.3/100w   | 3.8/10        | ❌ (need 5)       |
| MRI       | 30%        | 4.5/15        | ❌ (need 9)       |
| SCR       | 15:1       | 0/15          | ❌ (need 7)       |
| CDNC      | 0          | 0/10          | ❌ (need 5)       |
| EGS       | 2 pred     | 6.7/10        | ✅                |
| SCI       | High circ  | 4/10          | ❌ (need 7)       |
| NCR       | 95%        | 4.8/5         | ✅                |
| FEID      | N/A        | 0/5           | ❌ (need 3)       |
| PS        | 60%        | 3/5           | ✅                |
| **TOTAL** |            | **29.2/100**  | **Unacceptable** |

**Verdict**: Critical failure on 7/10 metrics. Requires complete restructuring, not incremental revision.

## Implementation Recommendation

Your agent loop should have **hard stops**:
- If SCR > 10:1 → reject immediately, don't even generate revisions
- If CNR < 15% → demand complete consolidation before refinement
- If MRI < 50% → demand proof outlines before prose

The generator shouldn't try to "fix" a 29/100 paper incrementally—it should extract the ~15% novel content and rebuild from scratch.