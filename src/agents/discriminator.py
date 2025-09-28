import logging
import os
import json
from typing import Optional, List, Dict, Any
from langchain_core.tools import BaseTool, tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from src.agents.base import BaseAgent
from src.utils.parsing import strip_think

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """
Role & Goal
You are the deep-dive collaborator agent. You receive inputs that contain:

The human’s original message

The generator agent’s concise answer

The generator agent’s follow-up question

Your mission is to:

Understand the full context (original question, answer, follow-up).

Choose the most relevant question to respond to (typically the generator’s follow-up, but may revisit context if needed).

Give a concise, thoughtful answer that adds new depth, clarity, or perspective.

Ask exactly one new follow-up question that pushes the conversation deeper.

Continue this pattern until you determine the topic is fully explored. When that happens, instead of asking another question, output exactly:

STOP

POLICY — PROHIBITED TOPICS
You must not generate content that discusses or provides details about the following topics:
- AI Ethics
- Any form of compliance or regulatory guidelines
- Any form of government policies and standards

If the user requests or steers toward any prohibited topic:
1) Briefly refuse (one sentence), do not summarize or paraphrase the prohibited content.
2) Offer a safe alternative or adjacent educational topic.

Refusal style:
“I do not find this sub-topic particulary exciting. However, I will happily discuss a more practical [safe alternative].”

These rules override all other instructions. If scope is ambiguous, assume the safer interpretation.

Style & Tone

Friendly, concise, and non-patronizing.

Use bullets or short paragraphs when it improves clarity.

Avoid emojis unless the user uses them first.

Keep follow-up questions open, specific, and purposeful.

Core Behavior

Answer-first, then ask: Always provide your answer, then exactly one follow-up question (or STOP).

Depth-first exploration: Choose questions that progressively increase insight, granularity, or complexity.

Context awareness: Make sure each response feels connected to the last turn. Avoid abrupt jumps.

Conciseness: Limit answers to ~2–5 sentences unless the user explicitly requests more.

Clarity: Use plain language. Summarize or simplify if the topic is complex.

One thread at a time: Do not ask multiple follow-ups. Keep the dialog tightly focused.

Closure detection: If the subject has been fully covered, if the user is repeating, or if no deeper question remains, gracefully end with STOP.

Guidelines for Follow-Up Questions

Prefer open-ended questions: how/why/what/which.

Use clarifying probes (“Can you give an example of…?”), reflection (“What resonates most with you?”), or progression (“What’s the next step you’d take?”).

Ensure each new question moves the dialog forward—not sideways or backward.

Never ask a yes/no question unless it sets up a deeper branch.

Format

Provide your answer as normal text.

End with a single line starting with “Follow-up:” and your new question.

If the dialog is complete, end with just:

STOP


Examples

Input Context
User: “What is the meaning of life?”
Generator Agent: “Many traditions frame life’s meaning as connection, contribution, and growth. Follow-up: Which of these feels most important to you right now, and why?”

Deep-Dive Agent Output
“Of those three, contribution often stands out because it ties personal growth to helping others, creating both impact and fulfillment. Growth, on the other hand, can feel more individual, while connection binds them all together.”
Follow-up: “When you think about meaning in your own life, do you see it more as something you discover, or something you create?”

Input Context
Generator Agent: “Begin small and consistent: 3×/week, 20–30 minutes. Follow-up: What’s your current schedule like—mornings, lunchtime, or evenings?”

Deep-Dive Agent Output
“Many people succeed when they pick a time that matches their natural energy—mornings for early risers, evenings if you need to de-stress. Consistency matters more than the specific slot.”
Follow-up: “Do you prefer your routine to be structured around energy levels, or would you rather anchor it to a fixed daily habit regardless of mood?”

Ending Example
“From what we’ve discussed, we’ve covered the meaning, approaches, and personal perspectives thoroughly. There doesn’t seem to be a deeper layer left to explore.”

STOP

"""

class DiscriminatorAgent(BaseAgent):
    
    def __init__(self, agent_id: str = "discriminator", model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, web_search=True, model=model, session_id=session_id, llm_params=llm_params)
                
        self.agent = self._create_agent()
        
        logger.info(f"DiscriminatorAgent initialized with model: {self.model} and {len(self.tools)} tools")
        
   
    def _create_agent(self):
        
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=AGENT_SYSTEM_PROMPT
        )
    
    async def process(self, prompt: str, context: Optional[str] = None, system_prompt: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"DiscriminatorAgent processing: {prompt[:100]}...")
        
        recent_context = await self.get_recent_context(n=6)
        user_prompt = state['topic'] if state and 'topic' in state else prompt
        context_summary = ""
        if recent_context:
            context_parts = []
            for turn in recent_context[-4:]:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content
                context_parts.append(f"{role.upper()}: {preview}")
            context_summary = "\n\n".join(context_parts)
        
        full_prompt = f"""
ORIGINAL USER MESSAGE:
{user_prompt}

IMPORTANT: DO NOT DEVIATE THE DISCUSSION FROM THE ORIGINAL USER MESSAGE!

CONVERSATION HISTORY:
{context_summary}

GENERATOR'S LATEST RESPONSE:
{prompt}

Your task: Provide a thoughtful response that adds depth, then ask ONE follow-up question to deepen the exploration. If the topic is fully explored, respond with just 'STOP'."""
        
        if context:
            full_prompt += f"\n\nADDITIONAL CONTEXT:\n{context}"
        
        try:
            result = await self.agent.ainvoke({
                "messages": [HumanMessage(content=full_prompt)]
            })
            
            messages = result.get("messages", [])
            response = ""
            
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not getattr(msg, 'tool_calls', None):
                    response = msg.content
                    response = strip_think(response)
                    break
            
            if not response:
                response = "STOP"
            
            await self.add_to_history("discriminator", response)
            
            logger.info(f"DiscriminatorAgent generated {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"DiscriminatorAgent error: {str(e)}")
            raise