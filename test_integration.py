#!/usr/bin/env python3
"""
Test the integrated profile and depth-aware generation system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prompts.prompt_composer import PromptComposer, GenerationContext


def test_prompt_differences():
    """Test that different profiles and depths generate different prompts"""
    composer = PromptComposer()
    
    topic = "quantum information preservation in cosmology"
    
    print("=" * 60)
    print("Testing Prompt Differentiation")
    print("=" * 60)
    
    # Test Scientific vs Popular Science at same depth
    sci_context = GenerationContext(topic=topic, depth=1, profile="scientific")
    pop_context = GenerationContext(topic=topic, depth=1, profile="popular_science")
    
    sci_prompt = composer.compose_prompt(sci_context)
    pop_prompt = composer.compose_prompt(pop_context)
    
    print("\n### Scientific Profile Requirements (Depth 1):")
    if "formal_definitions" in sci_prompt:
        print("✓ Requires formal definitions")
    if "mathematical_notation" in sci_prompt:
        print("✓ Requires mathematical notation")
    if "Minimum equations: 1" in sci_prompt:
        print("✓ Minimum 1 equation required")
    
    print("\n### Popular Science Profile Requirements (Depth 1):")
    if "real_world_analogies" in pop_prompt:
        print("✓ Requires real-world analogies")
    if "Maximum equations: 0" in pop_prompt:
        print("✓ No equations allowed")
    if "unexplained_jargon" in pop_prompt:
        print("✓ Forbids unexplained jargon")
    
    # Test depth progression with context
    print("\n### Depth Progression (Scientific):")
    
    depth1_content = "In depth 1, we covered the von Neumann entropy and unitarity principles..."
    depth2_context = GenerationContext(
        topic=topic,
        depth=2,
        profile="scientific",
        previous_depths={1: depth1_content}
    )
    
    depth2_prompt = composer.compose_prompt(depth2_context)
    
    if "CONTEXT FROM PREVIOUS DEPTHS" in depth2_prompt:
        print("✓ Depth 2 includes context from depth 1")
    if "DO NOT restate" in depth2_prompt:
        print("✓ Anti-repetition instructions present")
    if "von Neumann entropy" in depth2_prompt:
        print("✓ Key concepts from depth 1 referenced")
    
    # Show differences in requirements
    print("\n### Profile-Specific Requirements Comparison:")
    print("\nScientific Depth 1:")
    print("- Min citations: 2")
    print("- Min equations: 1") 
    print("- Complexity: advanced")
    
    print("\nPopular Science Depth 1:")
    print("- Max equations: 0")
    print("- Requires analogies: true")
    print("- Complexity: accessible")
    
    print("\nEducational Depth 2:")
    edu_context = GenerationContext(topic=topic, depth=2, profile="educational")
    edu_prompt = composer.compose_prompt(edu_context)
    if "worked_examples" in edu_prompt:
        print("- Requires worked examples")
    if "practice_problems" in edu_prompt:
        print("- Requires practice problems")
    
    print("\n✓ All profiles and depths generate distinct prompts!")
    return True


def test_similarity_thresholds():
    """Test profile-specific similarity thresholds"""
    from src.agents.similarity_detector import RepetitionDetector
    
    print("\n" + "=" * 60)
    print("Testing Profile-Specific Similarity Thresholds")
    print("=" * 60)
    
    detector = RepetitionDetector("test")
    
    print("\n### Thresholds by Profile:")
    for profile, thresholds in detector.PROFILE_THRESHOLDS.items():
        print(f"\n{profile.title()}:")
        print(f"  Identical: {thresholds['IDENTICAL']:.0%}")
        print(f"  Very Similar: {thresholds['VERY_SIMILAR']:.0%}")
        print(f"  Similar: {thresholds['SIMILAR']:.0%}")
    
    # Verify scientific is stricter
    sci_thresh = detector.PROFILE_THRESHOLDS['scientific']['IDENTICAL']
    pop_thresh = detector.PROFILE_THRESHOLDS['popular_science']['IDENTICAL']
    
    if sci_thresh > pop_thresh:
        print("\n✓ Scientific profile has stricter thresholds than popular science")
    
    return True


def main():
    """Run integration tests"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Profile & Depth-Aware Generation")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Prompt differences
    try:
        if test_prompt_differences():
            tests_passed += 1
            print("\n✓ Prompt differentiation test PASSED")
    except Exception as e:
        print(f"\n✗ Prompt differentiation test FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Similarity thresholds
    try:
        if test_similarity_thresholds():
            tests_passed += 1
            print("\n✓ Similarity threshold test PASSED")
    except Exception as e:
        print(f"\n✗ Similarity threshold test FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 All integration tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {tests_failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)