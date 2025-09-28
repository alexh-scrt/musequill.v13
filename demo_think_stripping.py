#!/usr/bin/env python3
"""Demo: Think tag stripping in action"""

from src.utils.parsing import strip_think

print("=" * 70)
print("  THINK TAG STRIPPING DEMO")
print("=" * 70)

examples = [
    {
        "name": "Reasoning Model Response",
        "llm_output": """<think>
The user is asking about machine learning. I should:
1. Define it clearly
2. Give a practical example
3. Ask about their experience level
</think>

Machine learning is a subset of AI where systems learn from data patterns 
rather than being explicitly programmed. For example, email spam filters 
learn to recognize spam by analyzing thousands of labeled emails.

Follow-up: Have you worked with any ML frameworks before?""",
    },
    {
        "name": "DeepSeek R1 Style",
        "llm_output": """<think>Let me break this down step by step...</think>

The answer to your question is 42.

Follow-up: Does this make sense to you?""",
    },
    {
        "name": "No Think Tags",
        "llm_output": """This is a regular response without any reasoning tags.

Follow-up: What would you like to know more about?""",
    },
    {
        "name": "Multiple Think Sections",
        "llm_output": """<think>First reasoning section</think>
First answer part.

<think>Second reasoning section</think>
Second answer part.

Follow-up: Shall we continue?""",
    }
]

for i, example in enumerate(examples, 1):
    print(f"\n{'─' * 70}")
    print(f"Example {i}: {example['name']}")
    print(f"{'─' * 70}")
    
    print("\n📥 LLM Output (what the model returns):")
    print("┌" + "─" * 68 + "┐")
    for line in example['llm_output'].split('\n'):
        if '<think>' in line.lower() or '</think>' in line.lower():
            print(f"│ \033[90m{line[:66]:<66}\033[0m │")  # Gray for think tags
        else:
            print(f"│ {line[:66]:<66} │")
    print("└" + "─" * 68 + "┘")
    
    cleaned = strip_think(example['llm_output'])
    
    print("\n📤 User Receives (after stripping):")
    print("┌" + "─" * 68 + "┐")
    for line in cleaned.split('\n'):
        print(f"│ \033[92m{line[:66]:<66}\033[0m │")  # Green for clean output
    print("└" + "─" * 68 + "┘")

print("\n" + "=" * 70)
print("✅ All reasoning tags automatically removed!")
print("=" * 70)
print("\nUsers only see the final answers, not the thinking process.")
print("This works automatically in GeneratorAgent and DiscriminatorAgent.\n")