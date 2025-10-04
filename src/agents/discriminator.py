import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from src.agents.base import BaseAgent
from src.utils.parsing import strip_think
from src.prompts.discriminator_prompts import FOCUSED_SYSTEM_PROMPT
from src.states.topic_focused import TopicFocusedState
from src.storage.similarity_corpus import SimilarityCorpus
from src.agents.similarity_checker import SimilarityChecker
from src.agents.similarity_feedback import augment_prompt_with_feedback

logger = logging.getLogger(__name__)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "25"))

class DiscriminatorAgent(BaseAgent):
    
    def __init__(self, agent_id: str = "discriminator", model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, web_search=True, model=model, session_id=session_id, llm_params=llm_params)
                
        self.agent = self._create_agent()
        
        # Initialize similarity checking components
        if session_id:
            self.similarity_corpus = SimilarityCorpus(session_id)
            self.similarity_checker = SimilarityChecker(session_id, self.similarity_corpus)
        else:
            self.similarity_corpus = None
            self.similarity_checker = None
        
        # Load similarity configuration
        self.max_similarity_attempts = int(os.getenv("MAX_SIMILARITY_ATTEMPTS", "5"))
        self.similarity_relaxed_threshold = float(os.getenv("SIMILARITY_RELAXED_THRESHOLD", "0.90"))
        
        logger.info(f"DiscriminatorAgent initialized with model: {self.model} and {len(self.tools)} tools")
        
   
    def _create_agent(self):
        
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=FOCUSED_SYSTEM_PROMPT
        )
    
    async def process(self, generator_response: str, state: TopicFocusedState = None) -> Tuple[str, Dict[str, Any]]:
        """Three-phase processing with similarity checking"""
        
        if state is None:
            state = {}
        
        # Store generator response in state for similarity checking
        state['generator_response'] = generator_response
        
        # Generate unique content with similarity checking
        if self.similarity_checker:
            content, attempts = await self._generate_unique_content(generator_response, state)
            logger.info(f"Generated unique discriminator response after {attempts} attempts")
        else:
            # Fallback to original logic if no similarity checking
            content = await self._process_without_similarity(generator_response, state)
            attempts = 1
        
        # Extract and store new follow-up question
        new_followup = self._extract_followup_question(content)
        
        await self.add_to_history("discriminator", content)
        
        # Return response and state updates
        iteration = state.get('iterations', 0)
        state_updates = {
            'last_followup_question': new_followup,
            'iterations': iteration + 1,
            'similarity_attempts': attempts
        }
        
        if iteration + 1 >= MAX_ITERATIONS:
            logger.warning(f"⚠️ Maximum iterations {MAX_ITERATIONS} reached in DiscriminatorAgent")
            state_updates['STOP'] = True

        return content, state_updates
    
    async def _process_without_similarity(self, generator_response: str, state: Dict[str, Any]) -> str:
        """Original process logic without similarity checking"""
        original_topic = state.get('original_topic', 'unknown topic')
        depth_level = state.get('current_depth_level', 1)
        aspects_explored = state.get('aspects_explored', [])
        topic_summary = state.get('topic_summary', '')
        last_question = state.get('last_followup_question', '')
        recent_context = await self.get_recent_context(n=6)
        
        # Phase 1: Answer last question if present
        immediate_response = ""
        if last_question and last_question.strip():
            answer_prompt = f"""
            ORIGINAL_TOPIC: {original_topic}
            
            RECENT_CONVERSATION_CONTEXT:
            {self._format_context(recent_context)}
            
            GENERATOR'S LATEST RESPONSE:
            {generator_response}
            
            Briefly answer this specific question: {last_question}
            
            Keep your answer to 1-2 sentences. We need to return focus to the main topic: {original_topic}
            """
            immediate_response = await self.generate_with_llm(answer_prompt)
        
        # Phase 2: Continue deepening conversation on original topic
        analysis_prompt = f"""
        ORIGINAL QUESTION: 
        {original_topic}
        
        ---
        
        CURRENT DEPTH LEVEL: {depth_level}/5
        
        ---
        
        TOPIC_SUMMARY: 
        {topic_summary}
        
        ---

        OPPONENT'S RESPONSE:
        {generator_response}

        ---
        
        RECENT_CONVERSATION:
        {self._format_context(recent_context)}
        
        ---

        Your tasks:
        1. Stay focused on the conversation topics within the boundaries of the ORIGINAL QUESTION
        2. Provide deeper reasoning and analysis of ORIGINAL QUESTION based on the conversation so far
        3. Ask ONE follow-up question that pushes understanding of the topic in the context of the ORIGINAL QUESTION further
        4. If ORIGINAL TOPIC thoroughly explored, respond with the detailed summary and "STOP" instead of a follow-up question.
        
        Remember: Your analysis and question must advance the discusson on ORIGINAL QUESTION, not drift into tangents.
        """
        
        topic_advancement = await self.generate_with_llm(analysis_prompt)
        
        # Phase 3: Combine responses
        if immediate_response:
            combined_response = f"{immediate_response}\n{topic_advancement}"
        else:
            combined_response = topic_advancement
        
        return combined_response
    
    def _extract_followup_question(self, response: str) -> str:
        """Extract the follow-up question from the response"""
        import re
        patterns = [
            # Handle numbered format with potential newlines/whitespace after header
            r'\*\*\d+\.\s*Follow[\u2011-]?up\s+questions?\*\*\s*[\n:]*\s*(.+?)(?=\n\n|\n\*\*|\Z)',
            # Handle bold follow-up question with colon and/or whitespace
            r'\*\*Follow[\u2011-]?up\s+questions?:\*\*\s*[\n]*\s*(.+?)(?=\n\n|\n\*\*|\Z)',
            # Handle bold follow-up question without colon
            r'\*\*Follow[\u2011-]?up\s+questions?\*\*\s*[\n:]*\s*(.+?)(?=\n\n|\n\*\*|\Z)',
            # Handle plain text follow-up with potential whitespace
            r'Follow[\u2011-]?up\s+questions?[\s:]*[\n]*\s*(.+?)(?=\n\n|\n\*\*|\Z)',
            r'Follow[\u2011-]?up[\s:]+[\n]*\s*(.+?)(?=\n\n|\n\*\*|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                # Clean up the extracted question
                question = match.group(1).strip()
                # Remove any leading/trailing whitespace and newlines
                question = question.strip('\n\r\t ')
                # If the question spans multiple lines, join them
                question = ' '.join(question.split())
                return question
        
        return ""

    def _format_context(self, context: List[Dict]) -> str:
        """Format recent context while preserving original topic focus"""
        if not context:
            return "No recent context"
        
        formatted = []
        for turn in context[-3:]:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')[:150] + "..."
            formatted.append(f"{role.upper()}: {content}")
        
        return "\n".join(formatted)
    
    async def _generate_unique_content(self, generator_response: str, state: Dict[str, Any]) -> Tuple[str, int]:
        """Generate content with similarity checking to ensure uniqueness.
        
        Args:
            generator_response: The generator's response to discriminate against
            state: Current workflow state
            
        Returns:
            Tuple of (unique content, attempts used)
        """
        attempts = 0
        best_content = None
        best_similarity = 1.0
        current_state = state.copy()
        
        while attempts < self.max_similarity_attempts:
            attempts += 1
            
            # Generate content using existing logic
            content = await self._process_without_similarity(generator_response, current_state)
            
            # Strip any think tags
            content = strip_think(content)
            
            # Check similarity
            result = await self.similarity_checker.check_similarity(content)
            
            # Track best attempt
            if result.overall_similarity < best_similarity:
                best_content = content
                best_similarity = result.overall_similarity
            
            # If unique, we're done
            if result.is_unique:
                logger.info(f"Discriminator content is unique with similarity {result.overall_similarity:.2%}")
                return content, attempts
            
            # Not unique, log and augment state for next attempt
            logger.warning(f"Discriminator content similarity {result.overall_similarity:.2%} exceeds threshold "
                         f"(attempt {attempts}/{self.max_similarity_attempts})")
            
            # Augment state with feedback for next attempt
            current_state = self._augment_state_with_similarity_feedback(state, result)
            
            # Increment similarity checker attempts
            self.similarity_checker.increment_attempts()
        
        # Max attempts reached
        logger.warning(f"Discriminator max similarity attempts ({self.max_similarity_attempts}) reached")
        
        # Check if best content meets relaxed threshold
        if best_similarity < self.similarity_relaxed_threshold:
            logger.info(f"Accepting discriminator content with relaxed threshold "
                       f"(similarity: {best_similarity:.2%} < {self.similarity_relaxed_threshold:.2%})")
        else:
            logger.error(f"Discriminator topic may be exhausted - best similarity {best_similarity:.2%} "
                        f"exceeds relaxed threshold {self.similarity_relaxed_threshold:.2%}")
        
        return best_content, attempts
    
    def _augment_state_with_similarity_feedback(self, original_state: Dict[str, Any], result) -> Dict[str, Any]:
        """Augment state with similarity feedback to encourage unique content.
        
        Args:
            original_state: The original state
            result: SimilarityResult with feedback
            
        Returns:
            Augmented state with similarity feedback embedded
        """
        augmented_state = original_state.copy()
        
        # Add similarity feedback to topic summary to influence generation
        if result.feedback:
            feedback_note = f"\n\nIMPORTANT: Avoid repeating these similar points:\n{result.feedback[:500]}"
            augmented_state['topic_summary'] = augmented_state.get('topic_summary', '') + feedback_note
        
        return augmented_state
