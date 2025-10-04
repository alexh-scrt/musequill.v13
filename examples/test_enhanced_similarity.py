#!/usr/bin/env python3
"""
Example script to test the enhanced similarity detection system.

This script demonstrates how the three-tier system works with sample content.
"""

import asyncio
import logging
import os
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.similarity_detector import RepetitionDetector, DecisionAction
from src.agents.repetition_log import RepetitionLog
from src.storage.similarity_corpus import SimilarityCorpus
from src.config.similarity_config import SimilarityConfig, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_similarity_detection():
    """Test the enhanced similarity detection with various content samples."""
    
    # Initialize components
    session_id = "test_demo"
    corpus = SimilarityCorpus(session_id)
    detector = RepetitionDetector(session_id, corpus)
    log = RepetitionLog(session_id)
    
    # Load and display configuration
    config = get_config()
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(config.describe())
    print()
    
    # Test samples
    samples = [
        {
            "content": """
            The holographic principle states that the entropy of a region is proportional 
            to the area of its boundary, not its volume. This fundamental insight suggests 
            that information in a volume can be encoded on its surface. The Bekenstein-Hawking 
            formula S = A/(4l_p^2) quantifies this relationship for black holes.
            """,
            "metadata": {
                "agent_id": "generator",
                "iteration": 1,
                "section": "introduction",
                "depth": 1,
                "content_type": "paragraph"
            }
        },
        {
            "content": """
            The holographic principle indicates that entropy is proportional to area rather 
            than volume. This principle implies that all information within a volume can be 
            represented on its boundary surface. For black holes, this is expressed through 
            the Bekenstein-Hawking equation S = A/(4l_p^2).
            """,
            "metadata": {
                "agent_id": "generator",
                "iteration": 2,
                "section": "overview",
                "depth": 1,
                "content_type": "paragraph"
            }
        },
        {
            "content": """
            The holographic principle has profound implications for quantum gravity. In the 
            AdS/CFT correspondence, a gravitational theory in Anti-de Sitter space is dual 
            to a conformal field theory on its boundary. This duality provides concrete 
            examples of holography where bulk physics emerges from boundary dynamics.
            """,
            "metadata": {
                "agent_id": "generator",
                "iteration": 3,
                "section": "applications",
                "depth": 2,
                "content_type": "paragraph"
            }
        },
        {
            "content": """
            Wheeler-DeWitt equation describes the quantum state of the universe without 
            reference to time. This equation, HΨ = 0, where H is the Hamiltonian constraint, 
            represents a timeless description of quantum cosmology. Unlike the holographic 
            principle, it focuses on the universe's wave function.
            """,
            "metadata": {
                "agent_id": "generator",
                "iteration": 4,
                "section": "quantum_cosmology",
                "depth": 2,
                "content_type": "paragraph"
            }
        },
        {
            "content": """
            The entropy-area relationship in holography can be understood through quantum 
            information theory. Entanglement entropy between regions follows similar scaling, 
            suggesting deep connections between geometry and quantum information. Recent work 
            on quantum error correction codes provides new perspectives on this relationship.
            """,
            "metadata": {
                "agent_id": "generator",
                "iteration": 5,
                "section": "quantum_information",
                "depth": 3,
                "content_type": "paragraph"
            }
        }
    ]
    
    print("\n" + "=" * 70)
    print("SIMILARITY DETECTION RESULTS")
    print("=" * 70)
    
    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Section: {sample['metadata']['section']}")
        print(f"Content preview: {sample['content'][:100].strip()}...")
        
        # Analyze content
        decision = await detector.analyze_content(
            sample["content"],
            sample["metadata"]
        )
        
        # Log decision
        log.log_decision(
            decision.__dict__ if hasattr(decision, '__dict__') else decision,
            sample["content"],
            sample["metadata"]
        )
        
        # Display results
        print(f"Decision: {decision.action.value}")
        print(f"Tier: {decision.tier}")
        print(f"Similarity Score: {decision.similarity_score:.1%}")
        print(f"Reason: {decision.reason}")
        
        if decision.analysis:
            print("Analysis:")
            for key, value in decision.analysis.items():
                if isinstance(value, list) and len(value) > 0:
                    print(f"  - {key}: {value[:3]}...")  # Show first 3 items
                elif isinstance(value, (int, float, bool)):
                    print(f"  - {key}: {value}")
        
        if decision.recommendation:
            print(f"Recommendation: {decision.recommendation}")
        
        # Store accepted content for future comparisons
        if decision.action == DecisionAction.ACCEPT:
            await detector.store_content(
                sample["content"],
                None,
                sample["metadata"]
            )
            print("✅ Content stored for future comparison")
        elif decision.action == DecisionAction.FLAG:
            print("⚠️ Content flagged for review")
        elif decision.action == DecisionAction.SKIP:
            print("❌ Content skipped as too similar")
    
    # Display final statistics
    print("\n" + "=" * 70)
    print("SESSION STATISTICS")
    print("=" * 70)
    print(log.generate_report())
    
    # Save logs
    log.close()
    print(f"\nLogs saved to: outputs/similarity_logs/")
    
    # Clean up
    corpus.clear_session()
    print(f"✅ Session cleaned up")


async def test_configuration_profiles():
    """Test different configuration profiles."""
    
    print("\n" + "=" * 70)
    print("CONFIGURATION PROFILES COMPARISON")
    print("=" * 70)
    
    profiles = ["strict", "balanced", "relaxed", "creative", "technical"]
    
    for profile in profiles:
        config = SimilarityConfig.get_profile_config(profile)
        print(f"\n--- {profile.upper()} Profile ---")
        print(f"Identical Threshold: {config.identical_threshold:.1%}")
        print(f"Very Similar Threshold: {config.very_similar_threshold:.1%}")
        print(f"Similar Threshold: {config.similar_threshold:.1%}")
        print(f"Min Novelty Ratio: {config.min_novelty_ratio:.1%}")
        print(f"Auto Skip: {config.auto_skip_identical}")
        print(f"Always Flag: {config.always_flag_never_skip}")


if __name__ == "__main__":
    print("Enhanced Similarity Detection System Test")
    print("=" * 70)
    
    # Run tests
    asyncio.run(test_similarity_detection())
    asyncio.run(test_configuration_profiles())
    
    print("\n✅ Test complete!")