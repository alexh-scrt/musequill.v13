# Evaluator Profiles - Quick Reference

## Profile Selector

```
┌─────────────────────────────────────────────────────────────┐
│ What are you writing?                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 📚 Research paper / Academic work                           │
│    → Use: scientific                                        │
│                                                             │
│ 🔬 Science for general audiences                            │
│    → Use: popular_science                                   │
│                                                             │
│ 💻 Tech review / Tutorial / Product analysis                │
│    → Use: technology                                        │
│                                                             │
│ 💰 Financial analysis / Investment insights                 │
│    → Use: investment                                        │
│                                                             │
│ 📝 Blog post / General content / Mixed topics               │
│    → Use: general                                           │
│                                                             │
│ ✍️  Story / Narrative / Creative writing                    │
│    → Use: creative                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## One-Liner Commands

```bash
# List all profiles
python client.py --list-profiles

# Scientific paper
python client.py "Quantum computing theory" --profile scientific

# Tech review
python client.py "M4 MacBook Pro review" --profile technology

# Popular science
python client.py "How CRISPR works" --profile popular_science

# Investment analysis
python client.py "Tech sector valuation" --profile investment

# General content
python client.py "Remote work guide" --profile general

# Creative writing
python client.py "Sci-fi short story" --profile creative
```

## Profile Characteristics at a Glance

| Profile             | Focus              | Strictness | Best For              |
| ------------------- | ------------------ | ---------- | --------------------- |
| **scientific**      | Rigor & proofs     | ⭐⭐⭐⭐⭐      | Academic papers       |
| **popular_science** | Clarity & examples | ⭐⭐⭐        | Science communication |
| **technology**      | Specs & benchmarks | ⭐⭐⭐⭐       | Tech reviews          |
| **investment**      | Data & sources     | ⭐⭐⭐⭐⭐      | Financial analysis    |
| **general**         | Balance            | ⭐⭐⭐        | Blog posts            |
| **creative**        | Originality & flow | ⭐⭐⭐⭐       | Stories & narratives  |

## Top 3 Priorities by Profile

### Scientific
1. 🔬 Mathematical Rigor (20 pts)
2. 📊 Empirical Grounding (15 pts)
3. 📚 Citation Context (15 pts)

### Popular Science
1. 🌍 Empirical Grounding - Real examples (20 pts)
2. 💡 Claim Density - Clear assertions (15 pts)
3. 📖 Structural Coherence - Clear flow (15 pts)

### Technology
1. ⚡ Empirical Grounding - Benchmarks (25 pts)
2. 📋 Claim Density - Specific claims (20 pts)
3. 💭 Conceptual Novelty - New insights (15 pts)

### Investment
1. 📊 Claim Density - Quantifiable (25 pts)
2. 📚 Citation Context - Data sources (20 pts)
3. 💹 Empirical Grounding - Metrics (20 pts)

### General
1. 📖 Structural Coherence (15 pts)
2. 💡 Claim Density (15 pts)
3. 🌍 Empirical Grounding (15 pts)

### Creative
1. ✨ Conceptual Novelty - Originality (25 pts)
2. 📖 Structural Coherence - Narrative arc (25 pts)
3. ✂️ Semantic Compression - Tight prose (20 pts)

## Critical Metrics (Auto-Reject if Failed)

```
scientific       → Mathematical rigor, Citation context, Empirical grounding
popular_science  → Claim density, Citation context, Structural coherence
technology       → Claim density, Empirical grounding
investment       → Claim density, Citation context, Empirical grounding
general          → Structural coherence
creative         → Conceptual novelty, Structural coherence, Semantic compression
```

## Quality Thresholds

```
Profile          Excellent  Good  Acceptable  Poor
─────────────────────────────────────────────────
scientific       85+        70+   55+         35+
popular_science  80+        65+   50+         35+
technology       80+        65+   50+         35+
investment       85+        70+   55+         40+
general          80+        65+   50+         35+
creative         85+        70+   55+         40+
```

## Common Patterns

### For Academic/Research
```bash
python client.py "Your research topic" \
    --profile scientific \
    --max-iterations 5
```

### For Public Writing
```bash
python client.py "Your topic" \
    --profile popular_science \
    --max-iterations 3
```

### For Technical Content
```bash
python client.py "Your tech topic" \
    --profile technology \
    --max-iterations 4
```

### For Financial Content
```bash
python client.py "Your analysis topic" \
    --profile investment \
    --max-iterations 3
```

### For Stories
```bash
python client.py "Your story premise" \
    --profile creative \
    --max-iterations 4
```

## Files to Update

When implementing profiles, update these files:

```
src/agents/profiles.py           ← Add EvaluatorProfileFactory
src/agents/evaluator.py          ← Accept profile parameter
src/workflow/orchestrator.py    ← Pass profile to evaluators
src/server/models.py             ← Add evaluator_profile field
src/server/app.py                ← Handle profile in request
client.py                        ← Add --profile argument
```

## Environment Variables

```bash
# .env or export
QUALITY_THRESHOLD=70.0           # Override default threshold
MAX_REFINEMENT_ITERATIONS=3      # Revisions per agent
MAX_ITERATIONS=25                # Max conversation turns
```

## Python API

```python
from src.workflow.orchestrator import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator(
    evaluator_profile="technology"
)

async for response in orchestrator.run_async(
    topic="Latest GPU architecture",
    evaluator_profile="technology"
):
    print(response.content)
```

## Decision Tree

```
Start
  │
  ├─ Academic/Research? ──→ scientific
  │
  ├─ Explaining science? ──→ popular_science
  │
  ├─ Tech/Product focus? ──→ technology
  │
  ├─ Financial/Investment? ──→ investment
  │
  ├─ Story/Narrative? ──→ creative
  │
  └─ Everything else ──→ general
```

## Remember

- **Profile ≠ Audience**: Choose based on content type, not who reads it
- **Start with defaults**: Adjust only if needed
- **Critical metrics matter**: These cause auto-reject if failed
- **Higher threshold = Better quality = Slower**: Balance speed vs. quality
- **When in doubt**: Use `general` profile

## Support

```bash
# List all available profiles
python client.py --list-profiles

# Get help
python client.py --help

# Check logs
tail -f outputs/conversation_log_*.md
```

---

**Quick Tip**: Most content fits `general`, `technology`, or `popular_science`. Start there!