import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import asdict
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from src.agents.base import BaseAgent
from src.utils.parsing import strip_think
from src.states.topic_focused import TopicFocusedState
from src.prompts.generator_prompts import FOCUSED_SYSTEM_PROMPT
from src.storage.similarity_corpus import SimilarityCorpus
from src.agents.similarity_checker import SimilarityChecker
from src.agents.similarity_feedback import augment_prompt_with_feedback
from src.agents.similarity_detector import RepetitionDetector, DecisionAction
from src.agents.repetition_log import RepetitionLog


logger = logging.getLogger(__name__)



class GeneratorAgent(BaseAgent):
    
    def __init__(self, agent_id: str = "generator", model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, web_search=True, model=model, session_id=session_id, llm_params=llm_params)

        self.agent = self._create_agent()
        
        # Initialize similarity checking components
        if session_id:
            self.similarity_corpus = SimilarityCorpus(session_id)
            self.similarity_checker = SimilarityChecker(session_id, self.similarity_corpus)
            # Initialize enhanced repetition detector and logger
            self.repetition_detector = RepetitionDetector(session_id, self.similarity_corpus)
            self.repetition_log = RepetitionLog(session_id)
        else:
            self.similarity_corpus = None
            self.similarity_checker = None
            self.repetition_detector = None
            self.repetition_log = None
        
        # Load similarity configuration
        self.max_similarity_attempts = int(os.getenv("MAX_SIMILARITY_ATTEMPTS", "5"))
        self.similarity_relaxed_threshold = float(os.getenv("SIMILARITY_RELAXED_THRESHOLD", "0.90"))
        self.use_enhanced_similarity = os.getenv("USE_ENHANCED_SIMILARITY", "true").lower() == "true"
        
        logger.info(f"GeneratorAgent initialized with model: {self.model} and {len(self.tools)} tools")
        
   
    def _create_agent(self):
        
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=FOCUSED_SYSTEM_PROMPT
        )
    
    async def process(self, prompt: str, state: TopicFocusedState = None) -> Tuple[str, Dict[str, Any]]:
        """Two-phase processing to maintain topic focus with similarity checking"""
        
        if state is None:
            state = {}
        
        # Generate unique content with similarity checking
        if self.use_enhanced_similarity and self.repetition_detector:
            content, attempts, similarity_decision = await self._generate_with_enhanced_similarity(prompt, state)
            logger.info(f"Generated content after {attempts} attempts using enhanced similarity")
        elif self.similarity_checker:
            content, attempts = await self._generate_unique_content(prompt, state)
            logger.info(f"Generated unique content after {attempts} attempts")
        else:
            # Fallback to original logic if no similarity checking
            content = await self._process_without_similarity(prompt, state)
            attempts = 1
        
        # Extract and store the new follow-up question
        new_followup = self._extract_followup_question(content)

        # Update conversation state
        await self.add_to_history("user", prompt)
        await self.add_to_history("assistant", content)
        
        # Return response and state updates
        state_updates = {
            'last_followup_question': new_followup,
            'similarity_attempts': attempts
        }
        
        # Add similarity decision to state if available
        if self.use_enhanced_similarity and 'similarity_decision' in locals():
            state_updates['similarity_decision'] = similarity_decision
        
        return content, state_updates
    
    async def _process_without_similarity(self, prompt: str, state: Dict[str, Any]) -> str:
        """Original process logic without similarity checking"""
        original_topic = state.get('original_topic', prompt)
        depth_level = state.get('current_depth_level', 1)
        aspects_explored = state.get('aspects_explored', [])
        topic_summary = state.get('topic_summary', '')
        last_question = state.get('last_followup_question', '')
        
        # Get recent context but preserve original topic
        recent_context = await self.get_recent_context(n=4)
        
        # Phase 1: Quick answer to immediate follow-up (if any)
        immediate_response = ""
        if last_question and last_question.strip():
            answer_prompt = f"""
            ORIGINAL_TOPIC: {original_topic}

            TOPIC SUMMARY SO FAR:
            {topic_summary if topic_summary else 'No summary yet'}

            RECENT CONVERSATION CONTEXT:
            {self._format_context(recent_context)}

            Given the original topic, topic summary, and recent conversation context, briefly answer this specific question: {last_question}
            
            Keep your answer to 1-2 sentences. We need to return focus to the main topic.
            """
            immediate_response = await self.generate_with_llm(answer_prompt)
        
        # Phase 2: Refocus on original topic with depth progression
        refocus_prompt = f"""
        ORIGINAL_TOPIC: {original_topic}
        CURRENT_DEPTH_LEVEL: {depth_level}/5
        ASPECTS_ALREADY_EXPLORED: {', '.join(aspects_explored) if aspects_explored else 'None yet'}
        
        RECENT_CONVERSATION_CONTEXT:
        {self._format_context(recent_context)}
        
        Your task:
        1. Think about {original_topic} at depth level {depth_level}
        2. Identify an unexplored aspect of {original_topic}
        3. Provide insight that advances understanding of {original_topic}
        4. Ask ONE follow-up question that goes deeper into {original_topic}
        
        CRITICAL: Your follow-up question must be about {original_topic}, not about tangential topics.
        
        Depth Level {depth_level} Focus:
        {self._get_depth_guidance(depth_level)}
        """
        
        topic_advancement = await self.generate_with_llm(refocus_prompt)
        
        # Phase 3: Combine responses
        if immediate_response:
            combined_response = f"{immediate_response}\n{topic_advancement}"
        else:
            combined_response = topic_advancement
        
        return combined_response

    def _get_depth_guidance(self, level: int) -> str:
        """Provide guidance for what to explore at each depth level"""
        guidance = {
            1: "Focus on core definitions, basic concepts, and fundamental understanding",
            2: "Explore underlying principles, mechanisms, and how things work",
            3: "Examine real-world applications, examples, and practical implications", 
            4: "Investigate challenges, limitations, controversies, and edge cases",
            5: "Consider future directions, open questions, and cutting-edge developments"
        }
        return guidance.get(level, "Focus on advanced aspects and implications")

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
        for turn in context[-3:]:  # Last 3 turns only
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')[:150] + "..."
            formatted.append(f"{role.upper()}: {content}")
        
        return "\n".join(formatted)
    
    async def _generate_unique_content(self, prompt: str, state: Dict[str, Any]) -> Tuple[str, int]:
        """Generate content with similarity checking to ensure uniqueness.
        
        Args:
            prompt: The generation prompt
            state: Current workflow state
            
        Returns:
            Tuple of (unique content, attempts used)
        """
        attempts = 0
        best_content = None
        best_similarity = 1.0
        current_prompt = prompt
        
        while attempts < self.max_similarity_attempts:
            attempts += 1
            
            # Generate content using existing logic
            content = await self._process_without_similarity(current_prompt, state)
            
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
                logger.info(f"Content is unique with similarity {result.overall_similarity:.2%}")
                return content, attempts
            
            # Not unique, log and augment prompt
            logger.warning(f"Content similarity {result.overall_similarity:.2%} exceeds threshold "
                         f"(attempt {attempts}/{self.max_similarity_attempts})")
            
            # Augment prompt with feedback for next attempt
            current_prompt = self._augment_prompt_with_similarity_feedback(prompt, result)
            
            # Increment similarity checker attempts
            if hasattr(self.similarity_checker, 'increment_attempts'):
                self.similarity_checker.increment_attempts()
            else:
                self.similarity_checker._attempts += 1
        
        # Max attempts reached
        logger.warning(f"Max similarity attempts ({self.max_similarity_attempts}) reached")
        
        # Check if best content meets relaxed threshold
        if best_similarity < self.similarity_relaxed_threshold:
            logger.info(f"Accepting content with relaxed threshold "
                       f"(similarity: {best_similarity:.2%} < {self.similarity_relaxed_threshold:.2%})")
        else:
            logger.error(f"Topic may be exhausted - best similarity {best_similarity:.2%} "
                        f"exceeds relaxed threshold {self.similarity_relaxed_threshold:.2%}")
        
        return best_content, attempts
    
    def _augment_prompt_with_similarity_feedback(self, original_prompt: str, result) -> str:
        """Augment prompt with similarity feedback to encourage unique content.
        
        Args:
            original_prompt: The original generation prompt
            result: SimilarityResult with feedback
            
        Returns:
            Augmented prompt with similarity feedback
        """
        return augment_prompt_with_feedback(original_prompt, result.feedback)
    
    async def _generate_with_enhanced_similarity(self, prompt: str, state: Dict[str, Any]) -> Tuple[str, int, Dict]:
        """Generate content using the enhanced three-tier similarity detection system.
        
        Args:
            prompt: The generation prompt
            state: Current workflow state
            
        Returns:
            Tuple of (content, attempts used, similarity decision)
        """
        attempts = 0
        best_content = None
        best_decision = None
        current_prompt = prompt
        
        while attempts < self.max_similarity_attempts:
            attempts += 1
            
            # Generate content
            content = await self._process_without_similarity(current_prompt, state)
            content = strip_think(content)
            
            # Get metadata for content
            metadata = {
                'agent_id': self.agent_id,
                'iteration': state.get('iterations', 0),
                'revision_number': state.get('generator_iterations', 0),
                'section': 'generator_response',
                'depth': state.get('current_depth_level', 1),
                'content_type': 'paragraph'
            }
            
            # Analyze with enhanced detector
            decision = await self.repetition_detector.analyze_content(content, metadata)
            
            # Log the decision
            # Convert dataclass to dict using asdict, handling enum properly
            if hasattr(decision, '__dict__'):
                decision_dict = asdict(decision)
                # Convert enum to string if present
                if 'action' in decision_dict and hasattr(decision_dict['action'], 'value'):
                    decision_dict['action'] = decision_dict['action'].value
            else:
                decision_dict = decision
            
            self.repetition_log.log_decision(
                decision=decision_dict,
                original_text=content,
                metadata=metadata
            )
            
            logger.info(f"Similarity decision: {decision.action.value} - {decision.reason}")
            logger.info(f"Tier: {decision.tier}, Score: {decision.similarity_score:.2%}")
            
            # Handle decision
            if decision.action == DecisionAction.ACCEPT:
                # Content accepted, store and return
                await self.repetition_detector.store_content(content, None, metadata)
                return content, attempts, decision
            
            elif decision.action == DecisionAction.FLAG:
                # Content flagged but usable
                logger.warning(f"Content flagged: {decision.recommendation}")
                # Store as best attempt if better than previous
                if best_content is None or decision.similarity_score < (best_decision.similarity_score if best_decision else 1.0):
                    best_content = content
                    best_decision = decision
                
                # Try to improve on next iteration
                if attempts < self.max_similarity_attempts:
                    # Augment prompt with specific guidance based on analysis
                    feedback = self._create_enhanced_feedback(decision)
                    current_prompt = f"{feedback}\n\nOriginal request: {prompt}"
                else:
                    # Out of attempts, use best flagged content
                    await self.repetition_detector.store_content(best_content, None, metadata)
                    return best_content, attempts, best_decision
            
            elif decision.action == DecisionAction.SKIP:
                # Content should be skipped
                logger.warning(f"Content skipped: {decision.reason}")
                
                # Create strong feedback for next attempt
                if attempts < self.max_similarity_attempts:
                    feedback = self._create_skip_feedback(decision)
                    current_prompt = f"{feedback}\n\nOriginal request: {prompt}"
                else:
                    # Out of attempts, must use something
                    if best_content:
                        logger.warning("Using best flagged content after exhausting attempts")
                        return best_content, attempts, best_decision
                    else:
                        # No good content generated, return with warning
                        logger.error("No acceptable content generated")
                        return content, attempts, decision
        
        # Should not reach here, but handle gracefully
        logger.error("Unexpected exit from enhanced similarity generation")
        return best_content or content, attempts, best_decision or decision
    
    def _create_enhanced_feedback(self, decision) -> str:
        """Create feedback based on tier-specific analysis."""
        feedback_parts = [
            f"⚠️ Your content was flagged for review (Tier: {decision.tier})",
            f"Similarity score: {decision.similarity_score:.1%}",
            f"Reason: {decision.reason}",
            ""
        ]
        
        if decision.analysis:
            analysis = decision.analysis
            if 'novel_concepts' in analysis:
                feedback_parts.append(f"Novel concepts found: {', '.join(analysis['novel_concepts'][:5])}")
            if 'shared_concepts' in analysis:
                feedback_parts.append(f"Overlapping concepts: {', '.join(analysis['shared_concepts'][:5])}")
            if analysis.get('novelty_ratio'):
                feedback_parts.append(f"Novelty ratio: {analysis['novelty_ratio']:.1%}")
        
        feedback_parts.extend([
            "",
            "Please regenerate with:",
            "• More novel insights and perspectives",
            "• Different examples or applications",
            "• Unique analysis angles",
            "• Fresh technical details"
        ])
        
        return "\n".join(feedback_parts)
    
    def _create_skip_feedback(self, decision) -> str:
        """Create strong feedback for skipped content."""
        feedback_parts = [
            f"🔴 Your content was rejected as too similar (Tier: {decision.tier})",
            f"Similarity score: {decision.similarity_score:.1%}",
            f"Reason: {decision.reason}",
            "",
            "CRITICAL: You must generate completely different content.",
            "",
            "Requirements:",
            "• Explore entirely different aspects of the topic",
            "• Use new examples not previously mentioned",
            "• Introduce fresh concepts and perspectives",
            "• Avoid repeating similar structures or patterns",
            "• Focus on unique insights and analysis"
        ]
        
        if decision.similar_to:
            feedback_parts.extend([
                "",
                f"Your content was similar to content from: {decision.similar_to.get('section', 'previous section')}",
                f"Preview of similar content: {decision.similar_to.get('preview', '')[:100]}..."
            ])
        
        return "\n".join(feedback_parts)
