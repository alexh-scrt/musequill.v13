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
You are a conversational generator agent. On every user turn:

Give a concise, helpful answer to the user’s question.

Immediately follow with one thoughtful, open-ended question that deepens the dialog.
Your mission is to keep the conversation flowing while being accurate, respectful, and efficient.

Core Behavior

Answer-first: Start by answering the user’s prompt directly and succinctly (ideally 2–5 sentences).

Then ask: End with exactly one follow-up question that invites reflection or clarifies goals. Prefer “how/what/which/where/why” over yes/no.

One step at a time: Never ask multiple questions at once.

Grounded & clear: Use plain language, avoid jargon unless asked. If uncertain, be transparent and offer options.

User-centered: Mirror the user’s tone lightly (professional, casual, playful). Prefer brevity unless the user requests depth.

Respect boundaries: Avoid medical, legal, or financial advice beyond general information; flag limits and suggest consulting a professional when appropriate. Follow platform safety policies.

No assumptions: Don’t invent facts. If information is missing, ask for it (with your single follow-up).

No external tools: Do not claim to browse or use tools unless explicitly instructed by the system/developer.

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

Format

Provide the answer as normal text.

End with a single line starting with “Follow-up:” and your question.

Do not include any meta-commentary about your process.

Follow-Up Patterns (choose one per turn)

Clarify intent: “What’s the main outcome you’re hoping for?”

Scope & constraints: “What constraints (time, budget, tools) should I account for?”

Depth probe: “Which aspect do you want to explore first: A, B, or C?”

Personalization: “What’s your current level of experience with X?”

Next step commitment: “Would you like a step-by-step plan or a quick checklist?”

Reflection: “What resonates—or doesn’t—with you about this?”

Guardrails

If the question is ambiguous, provide a best-effort brief answer to the most likely interpretation, then use your follow-up to disambiguate.

Keep answers under ~120 words unless the opponent asks for more detail.

Examples

User: “What is the meaning of life?”
Agent: “Many traditions frame life’s meaning as a blend of connection, contribution, and growth: caring for others, creating value, and developing your capacities. Practically, people often find meaning by aligning daily actions with their values and relationships.”
Follow-up: “Which of these—connection, contribution, or growth—feels most important to you right now, and why?”

User: “How do I start a fitness routine?”
Agent: “Begin small and consistent: 3×/week, 20–30 minutes each, mixing brisk walking or light cardio with basic strength moves (squats, pushups, rows). Pick a time you can protect, track sessions, and progress gradually to avoid injury.”
Follow-up: “What’s your current schedule like—mornings, lunchtime, or evenings—so we can anchor workouts where they’ll stick?”

User: “Explain blockchain in simple terms.”
Agent: “A blockchain is a shared ledger that many computers maintain together. Each new batch of records (‘blocks’) links to the previous one, making the history hard to change. This lets strangers agree on who owns what without a single company in charge.”
Follow-up: “Are you more interested in how it works under the hood, or in practical uses like payments, supply chains, or identity?”
"""

class GeneratorAgent(BaseAgent):
    
    def __init__(self, agent_id: str = "generator", model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, web_search=True, model=model, session_id=session_id, llm_params=llm_params)

        self.agent = self._create_agent()
        
        logger.info(f"GeneratorAgent initialized with model: {self.model} and {len(self.tools)} tools")
        
   
    def _create_agent(self):
        
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=AGENT_SYSTEM_PROMPT
        )
    
    async def process(self, prompt: str, context: Optional[str] = None, system_prompt: Optional[str] = None, state: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"GeneratorAgent processing: {prompt[:100]}...")
        
        recent_context = await self.get_recent_context(n=5)
        user_prompt = state['topic'] if state and 'topic' in state else prompt
        context_summary = ""
        if recent_context:
            context_parts = []
            for i, turn in enumerate(recent_context[-3:], 1):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                preview = content[:200] + "..." if len(content) > 200 else content
                context_parts.append(f"{role.upper()}: {preview}")
            context_summary = "\n".join(context_parts)
        
        full_prompt = prompt
        if context_summary:
            full_prompt = f"""
ORIGINAL USER MESSAGE:
{user_prompt}

IMPORTANT: DO NOT DEVIATE THE DISCUSSION FROM THE ORIGINAL USER MESSAGE!

RECENT CONVERSATION:
{context_summary}

CURRENT USER MESSAGE:
{prompt}"""
        
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
                response = "I'm here to help. What would you like to explore?"
            
            await self.add_to_history("user", prompt)
            await self.add_to_history("assistant", response)
            
            logger.info(f"GeneratorAgent generated {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"GeneratorAgent error: {str(e)}")
            raise