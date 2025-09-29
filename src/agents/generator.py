import logging
import os
import json
from typing import Optional, List, Dict, Any, Tuple
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from src.agents.base import BaseAgent
from src.utils.parsing import strip_think
from src.states.topic_focused import TopicFocusedState
from src.prompts.generator_prompts import FOCUSED_SYSTEM_PROMPT


logger = logging.getLogger(__name__)



class GeneratorAgent(BaseAgent):
    
    def __init__(self, agent_id: str = "generator", model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, web_search=True, model=model, session_id=session_id, llm_params=llm_params)

        self.agent = self._create_agent()
        
        logger.info(f"GeneratorAgent initialized with model: {self.model} and {len(self.tools)} tools")
        
   
    def _create_agent(self):
        
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=FOCUSED_SYSTEM_PROMPT
        )
    
    async def process(self, prompt: str, state: TopicFocusedState = None) -> Tuple[str, Dict[str, Any]]:
        """Two-phase processing to maintain topic focus"""
        
        if state is None:
            state = {}
        
        original_topic = state.get('original_topic', prompt)
        depth_level = state.get('current_depth_level', 1)
        aspects_explored = state.get('aspects_explored', [])
        topic_summary = state.get('topic_summary', '')
        last_question = state.get('last_followup_question', '')
        iteration = state.get('iterations', 0)
        
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
        
        # Extract and store the new follow-up question
        new_followup = self._extract_followup_question(combined_response)

        # Update conversation state
        await self.add_to_history("user", prompt)
        await self.add_to_history("assistant", combined_response)
        
        # Return response and state updates
        state_updates = {
            'last_followup_question': new_followup
        }
        
        return combined_response, state_updates

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
