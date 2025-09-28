# Think Tag Stripping Implementation

## Overview

Reasoning models (like DeepSeek R1, QwQ, or other CoT models) often output their reasoning process wrapped in `<think>...</think>` tags before providing the final answer. This implementation automatically strips these tags from all agent responses.

## Implementation

### 1. Created `src/utils/parsing.py`

```python
def strip_think(text: str) -> str:
    """Remove <think>...</think> tags and their contents from text."""
```

**Features:**
- ✅ Removes single-line think tags: `<think>reasoning</think>`
- ✅ Removes multi-line think tags with newlines
- ✅ Case-insensitive: handles `<THINK>`, `<Think>`, etc.
- ✅ Multiple occurrences: removes all think tags in text
- ✅ Cleans up excessive newlines after removal
- ✅ Logs when tags are stripped (debug level)

**Regex Pattern:** `r'<think>.*?</think>'` with `re.DOTALL | re.IGNORECASE`

### 2. Integrated into Agents

Both `GeneratorAgent` and `DiscriminatorAgent` now automatically strip think tags:

```python
# In process() method, after extracting response:
response = msg.content
response = strip_think(response)  # Strips <think> tags
```

**Location in code:**
- `src/agents/generator.py:147` - Applied to generator responses
- `src/agents/discriminator.py:153` - Applied to discriminator responses

## Usage

No changes needed in your code - stripping happens automatically!

### Example

**LLM Output (with think tags):**
```
<think>
Let me analyze this question. The user is asking about AI, so I should 
provide a clear definition first, then explore the implications...
</think>

Artificial intelligence refers to computer systems that can perform tasks 
that typically require human intelligence, such as learning, reasoning, 
and problem-solving.

Follow-up: What specific aspect of AI interests you most?
```

**User Receives (think tags stripped):**
```
Artificial intelligence refers to computer systems that can perform tasks 
that typically require human intelligence, such as learning, reasoning, 
and problem-solving.

Follow-up: What specific aspect of AI interests you most?
```

## Testing

All functionality has been tested and verified:

✅ Strip single-line think tags  
✅ Strip multi-line think tags  
✅ Handle case-insensitive variants  
✅ Handle multiple think tags  
✅ Preserve text without think tags  
✅ Clean up excessive whitespace  
✅ Integrated into both agents  
✅ All imports working correctly  

## File Structure

```
src/
├── utils/
│   ├── __init__.py          # Exports strip_think
│   └── parsing.py           # Implementation
├── agents/
│   ├── generator.py         # Uses strip_think
│   └── discriminator.py     # Uses strip_think
```

## Benefits

1. **Cleaner Responses**: Users see only the final answer, not reasoning process
2. **Model Flexibility**: Can use reasoning models without exposing internal thoughts
3. **Consistent Output**: Same clean format regardless of model type
4. **Automatic**: Zero configuration - just works out of the box
5. **Debug Friendly**: Logs when tags are stripped for troubleshooting

## Reasoning Models Supported

This implementation works with any model that outputs think tags, including:
- DeepSeek R1 (and R1 variants)
- QwQ (Qwen with Questions)
- Marco-o1
- Other chain-of-thought reasoning models
- Future models that adopt this convention

## Notes

- The regex is non-greedy (`.*?`) to handle nested or multiple tags correctly
- `re.DOTALL` flag allows `.` to match newlines (multi-line tags)
- `re.IGNORECASE` flag handles any capitalization variant
- Empty responses after stripping are handled by agent fallback logic
- Memory storage preserves the cleaned response (no think tags stored)