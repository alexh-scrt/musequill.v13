#!/usr/bin/env python3
"""Test script for similarity detection system."""

import asyncio
import logging
from src.storage.similarity_corpus import SimilarityCorpus
from src.agents.similarity_checker import SimilarityChecker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_similarity_detection():
    """Test the similarity detection system."""
    
    # Create corpus and checker
    session_id = "test_session_123"
    corpus = SimilarityCorpus(session_id)
    checker = SimilarityChecker(session_id, corpus)
    
    # Test content 1 - store in corpus
    content1 = """
    Quantum computing represents a revolutionary approach to information processing.
    It leverages quantum mechanical phenomena such as superposition and entanglement.
    Unlike classical computers that use bits, quantum computers use qubits.
    This allows them to perform certain calculations exponentially faster.
    """
    
    logger.info("Storing first content in corpus...")
    content_id1 = corpus.store_content(content1, "test_agent", {"iteration": 1})
    logger.info(f"Stored content with ID: {content_id1}")
    
    # Test content 2 - very similar to content1
    content2 = """
    Quantum computing is a revolutionary method for processing information.
    It utilizes quantum mechanical effects like superposition and entanglement.
    Different from classical computers using bits, quantum computers employ qubits.
    This enables them to execute specific calculations exponentially faster.
    """
    
    logger.info("\nChecking similarity of similar content...")
    result1 = await checker.check_similarity(content2)
    logger.info(f"Similar content - Similarity: {result1.overall_similarity:.2%}, Is unique: {result1.is_unique}")
    if result1.feedback:
        logger.info(f"Feedback:\n{result1.feedback[:500]}...")
    
    # Test content 3 - different content
    content3 = """
    Machine learning has transformed how we approach data analysis.
    Neural networks can learn complex patterns from large datasets.
    Deep learning models have achieved remarkable results in vision tasks.
    Transfer learning allows us to leverage pre-trained models effectively.
    """
    
    logger.info("\nChecking similarity of different content...")
    result2 = await checker.check_similarity(content3)
    logger.info(f"Different content - Similarity: {result2.overall_similarity:.2%}, Is unique: {result2.is_unique}")
    
    # Store the different content
    content_id2 = corpus.store_content(content3, "test_agent", {"iteration": 2})
    logger.info(f"Stored different content with ID: {content_id2}")
    
    # Get corpus stats
    stats = corpus.get_corpus_stats()
    logger.info(f"\nCorpus stats: {stats}")
    
    # Clean up
    logger.info("\nCleaning up session...")
    corpus.clear_session()
    logger.info("✅ Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_similarity_detection())