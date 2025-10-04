"""
Unit tests for the enhanced similarity detection system.

Tests the three-tier detection system with various content scenarios.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.agents.similarity_detector import (
    RepetitionDetector, 
    DecisionAction, 
    SimilarityDecision,
    ConceptExtraction
)
from src.agents.repetition_log import RepetitionLog
from src.common import ParagraphMatch


class TestRepetitionDetector:
    """Test suite for RepetitionDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create a RepetitionDetector instance for testing"""
        with patch('src.agents.similarity_detector.SimilarityCorpus'):
            detector = RepetitionDetector("test_session")
            # Mock corpus methods
            detector.corpus.search_similar_content = Mock(return_value=[])
            detector.corpus.store_content = Mock(return_value="test_id")
            return detector
    
    @pytest.fixture
    def sample_physics_text(self):
        """Sample physics text for testing"""
        return """
        The holographic principle states that entropy is proportional to area,
        not volume. This has profound implications for quantum gravity and
        black hole thermodynamics. The Bekenstein-Hawking formula S = A/4
        demonstrates this relationship explicitly.
        """
    
    @pytest.fixture
    def similar_physics_text(self):
        """Similar but not identical physics text"""
        return """
        The holographic principle indicates that entropy scales with area
        rather than volume. This fundamental insight impacts our understanding
        of quantum gravity and black hole physics. The relationship is 
        expressed through S = A/4 in Planck units.
        """
    
    @pytest.fixture
    def different_physics_text(self):
        """Different physics text on another topic"""
        return """
        Wheeler-DeWitt equation describes the quantum state of the universe
        without reference to time. This timeless framework emerges from
        canonical quantum gravity and presents unique interpretational
        challenges for cosmology.
        """
    
    @pytest.mark.asyncio
    async def test_tier1_identical_detection(self, detector, sample_physics_text):
        """Test Tier 1 identical content detection"""
        # Create a match with very high similarity
        mock_match = Mock()
        mock_match.similarity_score = 0.95
        mock_match.document_text = sample_physics_text
        mock_match.metadata = {'section': 'intro', 'depth': 1}
        
        detector.corpus.search_similar_content = Mock(return_value=[mock_match])
        
        metadata = {
            'content_type': 'paragraph',
            'section': 'intro',
            'depth': 1
        }
        
        decision = await detector.analyze_content(sample_physics_text, metadata)
        
        assert decision.action == DecisionAction.SKIP
        assert decision.tier == "TIER_1"
        assert decision.similarity_score >= 0.90
        assert "Near-identical" in decision.reason or "Very similar" in decision.reason
    
    @pytest.mark.asyncio
    async def test_tier2_semantic_analysis(self, detector, sample_physics_text, similar_physics_text):
        """Test Tier 2 semantic analysis"""
        # Create a match with moderate-high similarity
        mock_match = Mock()
        mock_match.similarity_score = 0.80
        mock_match.document_text = similar_physics_text
        mock_match.metadata = {'section': 'analysis', 'depth': 2}
        
        detector.corpus.search_similar_content = Mock(return_value=[mock_match])
        
        metadata = {
            'content_type': 'paragraph',
            'section': 'synthesis',
            'depth': 2
        }
        
        decision = await detector.analyze_content(sample_physics_text, metadata)
        
        assert decision.tier == "TIER_2"
        assert decision.action in [DecisionAction.FLAG, DecisionAction.SKIP]
        assert decision.analysis is not None
        assert 'novelty_ratio' in decision.analysis
    
    @pytest.mark.asyncio
    async def test_tier3_information_gain(self, detector, sample_physics_text):
        """Test Tier 3 information gain assessment"""
        # Create a match with moderate similarity
        mock_match = Mock()
        mock_match.similarity_score = 0.65
        mock_match.document_text = """
        The holographic principle relates entropy to surface area.
        This principle has applications in AdS/CFT correspondence
        and provides insights into quantum information theory.
        """
        mock_match.metadata = {'section': 'background', 'depth': 1}
        
        detector.corpus.search_similar_content = Mock(return_value=[mock_match])
        
        metadata = {
            'content_type': 'paragraph',
            'section': 'deep_dive',
            'depth': 3
        }
        
        decision = await detector.analyze_content(sample_physics_text, metadata)
        
        assert decision.tier == "TIER_3"
        assert decision.action in [DecisionAction.ACCEPT, DecisionAction.FLAG]
        assert decision.details is not None
    
    @pytest.mark.asyncio
    async def test_no_similar_content(self, detector, different_physics_text):
        """Test handling when no similar content is found"""
        detector.corpus.search_similar_content = Mock(return_value=[])
        
        metadata = {'content_type': 'paragraph', 'section': 'new', 'depth': 1}
        
        decision = await detector.analyze_content(different_physics_text, metadata)
        
        assert decision.action == DecisionAction.ACCEPT
        assert decision.similarity_score == 0.0
        assert "No similar content" in decision.reason
    
    def test_concept_extraction(self, detector, sample_physics_text):
        """Test concept extraction from physics text"""
        concepts = detector._extract_concepts(sample_physics_text)
        
        assert isinstance(concepts, ConceptExtraction)
        assert "holographic" in concepts.concepts or "holographic principle" in concepts.concepts
        assert "entropy" in concepts.concepts
        assert "Bekenstein-Hawking" in str(concepts.equations) or "S = A/4" in str(concepts.equations)
        assert len(concepts.concepts) > 0
    
    def test_different_perspective_check(self, detector):
        """Test checking for different perspectives"""
        text1_concepts = ConceptExtraction(
            concepts={'entropy', 'area', 'holographic'},
            equations=['S = A/4'],
            examples=['black hole entropy'],
            references=['Bekenstein 1973'],
            technical_terms={'holographic principle'}
        )
        
        text2_concepts = ConceptExtraction(
            concepts={'entropy', 'area', 'holographic'},
            equations=['S = kA/(4l_p^2)'],
            examples=['de Sitter space'],
            references=['Susskind 1995'],
            technical_terms={'holographic principle'}
        )
        
        is_different = detector._check_different_perspective(
            "text1", "text2", text1_concepts, text2_concepts
        )
        
        assert is_different  # Different examples and equations indicate different perspective
    
    def test_information_gain_calculation(self, detector):
        """Test information gain calculation"""
        new_text = """
        The holographic principle, as shown by 't Hooft and Susskind,
        states that S = A/4. For example, consider a spherical region
        in de Sitter space. Recent work by Maldacena extends this.
        """
        
        old_text = """
        The holographic principle states that entropy scales with area.
        This was first proposed by 't Hooft.
        """
        
        info_gain = detector._calculate_information_gain(new_text, old_text)
        
        assert isinstance(info_gain, dict)
        assert info_gain['has_new_examples'] == True  # de Sitter example
        assert info_gain['has_new_equations'] == True  # S = A/4
        assert info_gain['has_new_references'] == True  # Maldacena, Susskind
    
    @pytest.mark.asyncio
    async def test_configuration_respect(self, detector):
        """Test that detector respects configuration settings"""
        # Set to always flag, never skip
        detector.ALWAYS_FLAG_NEVER_SKIP = True
        detector.AUTO_SKIP_IDENTICAL = False
        
        # Create near-identical match
        mock_match = Mock()
        mock_match.similarity_score = 0.98
        mock_match.document_text = "identical content"
        mock_match.metadata = {}
        
        detector.corpus.search_similar_content = Mock(return_value=[mock_match])
        
        decision = await detector.analyze_content("identical content", {})
        
        # Should flag instead of skip due to configuration
        assert decision.action == DecisionAction.FLAG
    
    @pytest.mark.asyncio
    async def test_store_content(self, detector):
        """Test content storage for future comparison"""
        metadata = {
            'agent_id': 'test_agent',
            'iteration': 1,
            'section': 'test'
        }
        
        await detector.store_content("test content", None, metadata)
        
        detector.corpus.store_content.assert_called_once()
        assert len(detector.content_history) == 1
        assert detector.content_history[0]['text'] == "test content"


class TestRepetitionLog:
    """Test suite for RepetitionLog"""
    
    @pytest.fixture
    def logger(self, tmp_path):
        """Create a RepetitionLog instance for testing"""
        return RepetitionLog("test_session", str(tmp_path))
    
    def test_log_decision(self, logger):
        """Test logging a similarity decision"""
        decision = {
            'action': 'FLAG',
            'reason': 'Test reason',
            'similarity_score': 0.85,
            'tier': 'TIER_2'
        }
        
        metadata = {
            'agent_id': 'generator',
            'iteration': 1
        }
        
        logger.log_decision(decision, "Test content", metadata)
        
        assert len(logger.session_log) == 1
        assert len(logger.review_queue) == 1
        assert logger.session_log[0].action == 'FLAG'
    
    def test_statistics_calculation(self, logger):
        """Test statistics calculation"""
        # Log some decisions
        decisions = [
            {'action': 'SKIP', 'reason': 'Too similar', 'similarity_score': 0.95, 'tier': 'TIER_1'},
            {'action': 'FLAG', 'reason': 'Review needed', 'similarity_score': 0.80, 'tier': 'TIER_2'},
            {'action': 'ACCEPT', 'reason': 'Unique', 'similarity_score': 0.40, 'tier': 'LOW'},
            {'action': 'FLAG', 'reason': 'Check gain', 'similarity_score': 0.70, 'tier': 'TIER_3'},
        ]
        
        for decision in decisions:
            logger.log_decision(decision, "content", {})
        
        stats = logger.get_statistics()
        
        assert stats['total_checks'] == 4
        assert stats['action_counts']['skipped'] == 1
        assert stats['action_counts']['flagged'] == 2
        assert stats['action_counts']['accepted'] == 1
        assert 'similarity_scores' in stats
        assert stats['similarity_scores']['mean'] > 0
    
    def test_review_queue_export(self, logger):
        """Test exporting the review queue"""
        # Add flagged items
        for i in range(3):
            decision = {
                'action': 'FLAG',
                'reason': f'Reason {i}',
                'similarity_score': 0.75 + i * 0.05,
                'tier': 'TIER_2',
                'recommendation': f'Review item {i}'
            }
            logger.log_decision(decision, f"Content {i}", {'iteration': i})
        
        export = logger.export_review_queue()
        
        import json
        review_data = json.loads(export)
        
        assert len(review_data) == 3
        assert all('recommendation' in item for item in review_data)
        assert all('similarity_score' in item for item in review_data)
    
    def test_report_generation(self, logger):
        """Test human-readable report generation"""
        # Add various decisions
        decisions = [
            {'action': 'SKIP', 'reason': 'Identical', 'similarity_score': 0.95, 'tier': 'TIER_1'},
            {'action': 'FLAG', 'reason': 'Similar', 'similarity_score': 0.80, 'tier': 'TIER_2'},
            {'action': 'ACCEPT', 'reason': 'Novel', 'similarity_score': 0.30, 'tier': 'LOW'},
        ]
        
        for d in decisions:
            logger.log_decision(d, "content", {})
        
        report = logger.generate_report()
        
        assert "SIMILARITY DETECTION REPORT" in report
        assert "Total Checks: 3" in report
        assert "DECISIONS" in report
        assert "TIER DISTRIBUTION" in report


@pytest.mark.asyncio
async def test_integration_with_generator():
    """Test integration with GeneratorAgent"""
    from src.agents.generator import GeneratorAgent
    
    with patch('src.agents.generator.SimilarityCorpus'), \
         patch('src.agents.generator.SimilarityChecker'), \
         patch('src.agents.generator.RepetitionDetector') as MockDetector, \
         patch('src.agents.generator.RepetitionLog'):
        
        # Setup mock detector
        mock_detector = MockDetector.return_value
        mock_decision = SimilarityDecision(
            action=DecisionAction.ACCEPT,
            reason="Content is unique",
            similarity_score=0.3,
            tier="LOW",
            details={}
        )
        mock_detector.analyze_content = asyncio.coroutine(lambda *args: mock_decision)
        mock_detector.store_content = asyncio.coroutine(lambda *args: None)
        
        # Create generator with mocked LLM
        with patch.object(GeneratorAgent, '_create_agent'), \
             patch.object(GeneratorAgent, '_process_without_similarity') as mock_process:
            
            mock_process.return_value = asyncio.coroutine(lambda *args: "Generated content")()
            
            generator = GeneratorAgent(
                agent_id="test_gen",
                session_id="test_session"
            )
            generator.use_enhanced_similarity = True
            
            # Test generation
            content, state_updates = await generator.process("Test prompt", {})
            
            assert content is not None
            assert 'similarity_attempts' in state_updates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])