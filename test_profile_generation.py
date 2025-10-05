#!/usr/bin/env python3
"""
Test script for profile-aware content generation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.profile_aware_generator import ProfileAwareGenerator
from src.prompts.prompt_composer import PromptComposer, GenerationContext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_prompt_composition():
    """Test that prompt composition works correctly"""
    print("\n=== Testing Prompt Composition ===\n")
    
    composer = PromptComposer()
    
    # Test scientific profile at depth 1
    context = GenerationContext(
        topic="quantum entanglement",
        depth=1,
        profile="scientific",
        previous_depths=None
    )
    
    prompt = composer.compose_prompt(context)
    print("Scientific Depth 1 Prompt (first 500 chars):")
    print(prompt[:500])
    print("...")
    
    # Test popular science at depth 1
    context = GenerationContext(
        topic="quantum entanglement",
        depth=1,
        profile="popular_science",
        previous_depths=None
    )
    
    prompt = composer.compose_prompt(context)
    print("\nPopular Science Depth 1 Prompt (first 500 chars):")
    print(prompt[:500])
    print("...")
    
    # Test depth 2 with previous content
    previous_content = "Depth 1 covered quantum entanglement basics..."
    context = GenerationContext(
        topic="quantum entanglement",
        depth=2,
        profile="scientific",
        previous_depths={1: previous_content}
    )
    
    prompt = composer.compose_prompt(context)
    print("\nScientific Depth 2 with context (first 500 chars):")
    print(prompt[:500])
    print("...")
    
    return True


async def test_single_depth_generation():
    """Test generating a single depth of content"""
    print("\n=== Testing Single Depth Generation ===\n")
    
    generator = ProfileAwareGenerator(session_id="test_session")
    
    # Generate a simple test at depth 1
    topic = "test topic for entropy"
    profile = "scientific"
    depth = 1
    
    print(f"Generating {profile} content at depth {depth} for topic: {topic}")
    
    try:
        # Mock a simple response for testing without Ollama
        # In real usage, this would call the LLM
        content = """
        Entropy is a fundamental concept in thermodynamics and information theory.
        In thermodynamics, entropy S is defined by the Boltzmann equation:
        S = k_B ln(Ω)
        where k_B is Boltzmann's constant and Ω is the number of microstates.
        """
        
        # Test the similarity checking
        metadata = {
            'depth': depth,
            'profile': profile,
            'topic': topic,
            'attempt': 1
        }
        
        decision = await generator.repetition_detector.analyze_content(
            text=content,
            metadata=metadata,
            profile=profile
        )
        
        print(f"Content preview: {content[:200]}")
        print(f"Similarity Decision: {decision.action.value}")
        print(f"Similarity Score: {decision.similarity_score:.2%}")
        print(f"Reason: {decision.reason}")
        
        return True
        
    except Exception as e:
        logger.error(f"Generation test failed: {e}")
        return False


async def test_profile_thresholds():
    """Test that different profiles have different similarity thresholds"""
    print("\n=== Testing Profile-Specific Thresholds ===\n")
    
    from src.agents.similarity_detector import RepetitionDetector
    from src.storage.similarity_corpus import SimilarityCorpus
    
    corpus = SimilarityCorpus("test_thresholds")
    detector = RepetitionDetector("test_thresholds", corpus)
    
    profiles = ['scientific', 'popular_science', 'educational']
    
    for profile in profiles:
        thresholds = detector.PROFILE_THRESHOLDS.get(profile, {})
        print(f"{profile} thresholds:")
        print(f"  - IDENTICAL: {thresholds.get('IDENTICAL', 'N/A')}")
        print(f"  - VERY_SIMILAR: {thresholds.get('VERY_SIMILAR', 'N/A')}")
        print(f"  - SIMILAR: {thresholds.get('SIMILAR', 'N/A')}")
    
    return True


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Profile-Aware Content Generation")
    print("=" * 60)
    
    tests = [
        ("Prompt Composition", test_prompt_composition),
        ("Single Depth Generation", test_single_depth_generation),
        ("Profile Thresholds", test_profile_thresholds),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)