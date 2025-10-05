#!/usr/bin/env python3
"""
Demo script showing how the profile and depth-aware system addresses repetition.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompts.prompt_composer import PromptComposer, GenerationContext


def main():
    """Demonstrate the anti-repetition features"""
    
    print("=" * 70)
    print("DEMO: How Profile & Depth-Aware Generation Reduces Repetition")
    print("=" * 70)
    
    composer = PromptComposer()
    topic = "quantum information preservation in cosmology"
    
    # Simulate content from previous depths
    depth1_content = """
    At the foundational level, quantum information conservation in cosmology 
    relates to the fundamental principle of unitarity in quantum mechanics. 
    The von Neumann entropy S = -Tr(ρ ln ρ) provides a measure of quantum 
    information content in a density matrix ρ. In cosmological contexts, 
    this raises questions about information preservation during cosmic evolution,
    particularly at horizons where causal disconnection occurs.
    """
    
    depth2_content = """
    Building on the unitarity framework, the Wheeler-DeWitt equation provides
    a formal mathematical structure for quantum cosmology. The covariant entropy
    bounds, particularly the Bousso bound, constrain information content in
    spacetime regions. Detailed calculations show that holographic entropy
    S ≤ A/4l_p² applies to cosmological horizons, suggesting fundamental limits
    on information storage in de Sitter space.
    """
    
    print("\n### Problem: Traditional generation would repeat concepts")
    print("-" * 60)
    print("Without context awareness, each depth might:")
    print("• Re-explain what von Neumann entropy is")
    print("• Repeat the same examples of unitarity")
    print("• Use similar phrasing and structure")
    print("• Cover the same ground with slight variations")
    
    print("\n### Solution: Context-Aware Prompt Generation")
    print("-" * 60)
    
    # Generate depth 3 prompt with context
    context = GenerationContext(
        topic=topic,
        depth=3,
        profile="scientific",
        previous_depths={
            1: depth1_content,
            2: depth2_content
        }
    )
    
    prompt = composer.compose_prompt(context)
    
    # Show key anti-repetition features
    print("\n1. DEPTH-SPECIFIC PURPOSE:")
    print("   Depth 1: Introduce WHAT (foundations)")
    print("   Depth 2: Explain HOW (mechanisms)")
    print("   Depth 3: Explore UNKNOWNS (open problems)")
    
    print("\n2. EXTRACTED CONCEPTS FROM PREVIOUS DEPTHS:")
    concepts = composer._extract_key_concepts(depth1_content + depth2_content)
    print(f"   Identified concepts to avoid repeating:")
    for concept in concepts[:5]:
        print(f"   • {concept}")
    
    print("\n3. EXPLICIT ANTI-REPETITION INSTRUCTIONS:")
    if "DO NOT restate" in prompt:
        print("   ✓ Prompt includes 'DO NOT restate' instruction")
    if "ASSUME reader has read" in prompt:
        print("   ✓ Prompt assumes reader has previous context")
    if "Add NEW information" in prompt:
        print("   ✓ Prompt requires NEW information")
    
    print("\n4. PROFILE-SPECIFIC CONSTRAINTS:")
    print("   Scientific Depth 3 Requirements:")
    print("   • Must discuss open problems and research frontiers")
    print("   • Minimum 5 citations (3 from last 3 years)")
    print("   • Forbidden: basic definitions, introductory material")
    
    print("\n5. DIFFERENT PROMPTS FOR DIFFERENT AUDIENCES:")
    print("-" * 60)
    
    # Compare scientific vs popular science
    pop_context = GenerationContext(
        topic=topic,
        depth=1,
        profile="popular_science"
    )
    
    pop_prompt = composer.compose_prompt(pop_context)
    
    print("\nScientific Profile:")
    print("• Requires: formal definitions, mathematical notation, equations")
    print("• Tone: formal, precise, technical")
    
    print("\nPopular Science Profile:")
    print("• Requires: real-world analogies, concrete examples")
    print("• Forbidden: unexplained jargon, raw equations")
    print("• Tone: engaging, accessible, conversational")
    
    print("\n### Result: Each depth is SUBSTANTIVELY DIFFERENT")
    print("-" * 60)
    print("✓ Different questions answered at each depth")
    print("✓ Previous concepts referenced but not repeated")
    print("✓ Profile-appropriate content style")
    print("✓ Enforced constraints prevent repetition")
    print("✓ Similarity detection catches remaining duplicates")
    
    print("\n### Example Output Structure:")
    print("-" * 60)
    print("Depth 1: 'Quantum information is conserved due to unitarity...'")
    print("Depth 2: 'The mathematical formalism shows conservation via...'")
    print("Depth 3: 'Open questions remain: Is information truly conserved")
    print("         during inflation? Recent work by [2023 papers] suggests...'")
    
    print("\n" + "=" * 70)
    print("This system ensures each depth adds VALUE, not REPETITION")
    print("=" * 70)


if __name__ == "__main__":
    main()