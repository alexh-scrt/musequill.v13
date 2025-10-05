"""
Profile-aware content generator with depth differentiation.
"""

import logging
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from src.agents.base import BaseAgent
from src.agents.generator import GeneratorAgent
from src.prompts.prompt_composer import PromptComposer, GenerationContext
from src.agents.similarity_detector import RepetitionDetector, DecisionAction
from src.agents.repetition_log import RepetitionLog
from src.storage.similarity_corpus import SimilarityCorpus
from src.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Container for multi-depth document"""
    topic: str
    profile: str
    sections: Dict[int, str]
    metadata: Dict[int, Dict[str, Any]]
    
    def get_previous_depths(self) -> Dict[int, str]:
        """Get all previous depth sections"""
        return {depth: content for depth, content in self.sections.items()}
    
    def add_section(self, depth: int, content: str, metadata: Dict[str, Any]):
        """Add a new section at specified depth"""
        self.sections[depth] = content
        self.metadata[depth] = metadata
        
    def get_section(self, depth: int) -> Optional[str]:
        """Get content for a specific depth"""
        return self.sections.get(depth)
    
    def render(self) -> str:
        """Render complete document as markdown"""
        output = [f"# {self.topic}\n"]
        output.append(f"**Profile**: {self.profile}\n\n")
        
        for depth in sorted(self.sections.keys()):
            output.append(f"## Depth {depth}\n\n")
            output.append(self.sections[depth])
            output.append("\n\n")
            
        return "".join(output)
    
    def save(self, filepath: str):
        """Save document to file"""
        with open(filepath, 'w') as f:
            f.write(self.render())


class ProfileAwareGenerator:
    """
    Generator that uses profile and depth-aware prompts for content generation.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or "default"
        
        # Initialize prompt composer
        self.prompt_composer = PromptComposer()
        
        # Initialize similarity detection
        self.similarity_corpus = SimilarityCorpus(self.session_id)
        self.repetition_detector = RepetitionDetector(self.session_id, self.similarity_corpus)
        self.repetition_log = RepetitionLog(self.session_id)
        
        # Initialize LLM client
        self.llm_client = OllamaClient()
        
        # Get model from environment or use default
        self.model = os.getenv('WRITER_MODEL', 'qwen3:8b')
        
        # Max retries for content generation
        self.max_retries = int(os.getenv('MAX_SIMILARITY_ATTEMPTS', '3'))
        
        logger.info(f"ProfileAwareGenerator initialized with session {session_id}")
    
    async def generate_depth_section(
        self,
        topic: str,
        depth: int,
        profile: str,
        previous_depths: Optional[Dict[int, str]] = None,
        max_retries: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate content for a specific depth with profile awareness.
        
        Args:
            topic: The topic to generate content about
            depth: The depth level (1, 2, or 3)
            profile: The profile to use (scientific, popular_science, educational)
            previous_depths: Dictionary of previous depth content
            max_retries: Maximum retry attempts for similarity checking
            
        Returns:
            Tuple of (generated content, metadata including decision info)
        """
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries):
            logger.info(f"Generating depth {depth} for profile {profile}, attempt {attempt + 1}/{max_retries}")
            
            # Compose prompt
            context = GenerationContext(
                topic=topic,
                depth=depth,
                profile=profile,
                previous_depths=previous_depths
            )
            
            prompt = self.prompt_composer.compose_prompt(context)
            
            # Generate content
            try:
                content = await self.llm_client.generate(
                    prompt=prompt,
                    model=self.model,
                    temperature=0.7,
                    stream=False
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                content = f"Error generating content for depth {depth}: {str(e)}"
            
            # Check similarity with existing content
            metadata = {
                'depth': depth,
                'profile': profile,
                'topic': topic,
                'attempt': attempt + 1
            }
            
            decision = await self.repetition_detector.analyze_content(
                text=content,
                metadata=metadata,
                profile=profile
            )
            
            # Log the decision  
            self.repetition_log.log_decision(
                decision=asdict(decision) if hasattr(decision, '__dict__') else decision,
                original_text=content,
                metadata=metadata
            )
            
            # Process based on decision
            if decision.action == DecisionAction.ACCEPT:
                logger.info(f"Content accepted for depth {depth}")
                # Store in corpus for future comparisons
                self.similarity_corpus.store_content(content, metadata)
                return content, {
                    'decision': decision.action.value,
                    'similarity_score': decision.similarity_score,
                    'attempts': attempt + 1
                }
            
            elif decision.action == DecisionAction.FLAG:
                logger.warning(f"Content flagged for depth {depth}: {decision.reason}")
                # Store but mark as flagged
                self.similarity_corpus.store_content(content, {**metadata, 'flagged': True})
                return content, {
                    'decision': decision.action.value,
                    'similarity_score': decision.similarity_score,
                    'attempts': attempt + 1,
                    'flagged': True,
                    'reason': decision.reason
                }
            
            elif decision.action == DecisionAction.SKIP:
                logger.warning(f"Content skipped for depth {depth}: {decision.reason}")
                # On retry, add anti-repetition instruction to prompt
                if attempt < max_retries - 1:
                    # Enhance prompt with specific feedback about what was too similar
                    anti_repetition_note = f"""

IMPORTANT: The previous attempt was rejected for being too similar to existing content.
Reason: {decision.reason}
Similarity score: {decision.similarity_score:.2%}

Please generate DIFFERENT content that:
- Uses different examples and perspectives
- Avoids repeating the same concepts or explanations
- Provides new insights not covered before
"""
                    prompt = prompt + anti_repetition_note
        
        # Failed all retries
        error_msg = f"Could not generate acceptable content for depth {depth} with profile {profile} after {max_retries} attempts"
        logger.error(error_msg)
        return error_msg, {
            'decision': 'FAILED',
            'attempts': max_retries,
            'error': 'Max retries exceeded'
        }
    
    async def generate_document(
        self,
        topic: str,
        profile: str,
        max_depth: int = 3
    ) -> Document:
        """
        Generate complete document with all requested depths.
        
        Args:
            topic: The topic to generate content about
            profile: The profile to use
            max_depth: Maximum depth level to generate (default 3)
            
        Returns:
            Complete Document object with all depths
        """
        document = Document(
            topic=topic,
            profile=profile,
            sections={},
            metadata={}
        )
        
        for depth in range(1, max_depth + 1):
            logger.info(f"Generating depth {depth} for profile {profile}")
            
            # Get previous depths for context
            previous_depths = document.get_previous_depths() if depth > 1 else None
            
            # Generate this depth
            content, metadata = await self.generate_depth_section(
                topic=topic,
                depth=depth,
                profile=profile,
                previous_depths=previous_depths
            )
            
            # Add to document
            document.add_section(
                depth=depth,
                content=content,
                metadata=metadata
            )
            
            logger.info(f"Depth {depth} completed with decision: {metadata.get('decision', 'UNKNOWN')}")
        
        # Generate final statistics
        stats = self.repetition_log.get_session_stats()
        logger.info(f"Document generation complete. Stats: {stats}")
        
        return document
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available profiles"""
        return self.prompt_composer.get_profile_names()
    
    def get_max_depth_for_profile(self, profile: str) -> int:
        """Get maximum depth available for a profile"""
        return self.prompt_composer.get_max_depth(profile)