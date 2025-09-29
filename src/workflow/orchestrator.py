import asyncio
import logging
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, TypedDict, Annotated, Literal, Dict, List, Any, Tuple, Optional
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agents.generator import GeneratorAgent
from src.agents.discriminator import DiscriminatorAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.profiles import GeneratorProfileFactory, DiscriminatorProfileFactory

from src.server.models import AgentResponse
from src.storage.chroma_manager import ChromaManager
from src.states.topic_focused import TopicFocusedState
logger = logging.getLogger(__name__)


MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "25"))

class WorkflowOrchestrator:
    """Orchestrate collaboration between agents using clean LangGraph architecture"""
    
    def __init__(self, session_id: Optional[str] = None, generator_profile: str = "balanced", discriminator_profile: str = "balanced"):
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Get LLM parameters from profile factories
        gen_params = GeneratorProfileFactory.get(generator_profile)
        disc_params = DiscriminatorProfileFactory.get(discriminator_profile)
        
        logger.info(f"Generator profile: {generator_profile}")
        logger.info(f"Discriminator profile: {discriminator_profile}")
        
        self.generator = GeneratorAgent(
            session_id=self.session_id, 
            llm_params=gen_params)
        
        self.discriminator = DiscriminatorAgent(
            session_id=self.session_id, 
            llm_params=disc_params)        

        self.generator_evaluator = EvaluatorAgent(
            session_id=self.session_id
        )

        self.discriminator_evaluator = EvaluatorAgent(
            session_id=self.session_id
        )

        self.summarizer = SummarizerAgent(
            session_id=self.session_id,
            llm_params={'temperature': 0.3}  # Lower temp for consistency
        )

        self.graph = None
        self.compiled_graph = None
        
        self._log_queue = asyncio.Queue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        self._log_filepath = outputs_dir / f"conversation_log_{timestamp}.md"
        self._log_task = None
        
        self._build_graph()
        logger.info(f"WorkflowOrchestrator initialized with session_id: {self.session_id}")
    
    async def _log_worker(self):
        while True:
            try:
                log_entry = await self._log_queue.get()
                
                if log_entry is None:
                    break
                
                agent_id = log_entry.get("agent_id", "user")
                content = log_entry.get("content", "")

                content = content.replace("Follow-up:",'')
                content = re.sub(r'^(\*\*)?(?:Answer|Response):(\*\*)?\s*', '', content, flags=re.MULTILINE)
                
                async with asyncio.Lock():
                    with open(self._log_filepath, "a", encoding="utf-8") as f:
                        f.write(f"<{agent_id}>\n")
                        f.write(f"{content}\n")
                        f.write(f"</{agent_id}>\n\n")
                
                self._log_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in log worker: {e}", exc_info=True)
    
    async def shutdown(self):
        if self._log_task:
            self._log_queue.put_nowait(None)
            await self._log_task
            logger.info(f"✅ Conversation log written to: {self._log_filepath}")
    
    def _build_graph(self):
        """Build the clean workflow graph with explicit phase transitions"""
        workflow = StateGraph(TopicFocusedState)
        
        # Phase 1: Planning nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("discriminator", self._discriminator_node)
        workflow.add_node("summarizer", self._summarizer_node)  # NEW NODE

        # Planning phase edges
        workflow.add_edge(START, "generator")
        workflow.add_edge("generator", "generator_evaluator")
        workflow.add_conditional_edges(
            "generator_evaluator",
            self._should_revise_generator
        )
        workflow.add_edge("discriminator", "discriminator_evaluator")
        workflow.add_conditional_edges(
            "discriminator_evaluator",
            self._should_revise_discriminator
        )
        workflow.add_edge("discriminator_evaluator", "summarizer")
        workflow.add_edge("summarizer", END)

        # Compile with memory
        memory = MemorySaver()
        self.compiled_graph = workflow.compile(checkpointer=memory)
        logger.info("✅ Workflow graph compiled successfully")
    

    def _should_stop(self, state: TopicFocusedState) -> str:
        if state.get("END") or state.get("STOP"):
            return END
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            logger.warning(f"⚠️ Maximum iterations {MAX_ITERATIONS} reached in WorkflowOrchestrator")
            return END
        return "generator"

    async def _summarizer_node(self, state: TopicFocusedState) -> dict:
        """Generate comprehensive summary and quality assessment"""
        logger.info("📊 SUMMARIZER")
        logger.info(f"📊 Generating final summary for session {self.session_id}")
        
        try:
            summary, state_updates = await self.summarizer.process(state=state)
            
            # Log the summary
            self._log_queue.put_nowait({
                "agent_id": "summarizer",
                "content": summary
            })
            
            return {
                **state_updates,
                "current_response": summary,
                "END": True
            }
        except Exception as e:
            logger.error(f"Error in summarizer node: {e}", exc_info=True)
            return {
                "current_response": f"Error generating summary: {str(e)}",
                "END": True
            }



    # ====================
    # Planning Phase Nodes
    # ====================
    
    async def _generator_node(self, state: TopicFocusedState) -> dict:
        logger.info("📋  GENERATOR")
        iteration = state.get("iterations", 0)
        logger.info(f"📋  GENERATOR iteration {iteration + 1}")
        
        original_topic = state.get("original_topic", "")
        current_depth_level = state.get("current_depth_level", 1)
        aspects_explored = state.get("aspects_explored", [])
        topic_summary = state.get("topic_summary", "")
        last_followup_question = state.get("last_followup_question", "")
        
        user_prompt = last_followup_question if last_followup_question else original_topic
        
        response, state_updates = await self.generator.process(user_prompt, state=state)
        
        return {
            **state_updates,
            "current_response": response,
            "iterations": iteration
        }


    async def _discriminator_node(self, state: TopicFocusedState) -> dict:
        logger.info("📋  DISCRIMINATOR")
        iteration = state.get("iterations", 0)
        logger.info(f"📋  DISCRIMINATOR iteration {iteration + 1}")
        
        generator_response = state.get("current_response", "")
        response, state_updates = await self.discriminator.process(generator_response, state)
        
        should_end = "STOP" in response.upper()
        
        return {
            **state_updates,
            "current_response": response,
            "iterations": iteration + 1,
            "END": should_end
        }
    
    async def run_async(
        self,
        topic: str,
        max_iterations: int = 3,
        quality_threshold: float = 75.0
    ) -> AsyncGenerator[AgentResponse, None]:
        """Run the orchestrated workflow with streaming responses"""
        
        logger.info(f"🚀 Starting orchestrated workflow for: {topic}")
        logger.info(f"Config: max_iterations={max_iterations}, threshold={quality_threshold}")
        
        state = TopicFocusedState(
            original_topic=topic,
            current_depth_level=1,
            aspects_explored=[],
            topic_summary="",
            last_followup_question="",
            iterations=0,
            current_response=""
        )
 
        
        self._log_task = asyncio.create_task(self._log_worker())
        self._log_queue.put_nowait({
            "agent_id": "user",
            "content": topic
        })
        
        try:
            # Create config with thread_id and increased recursion limit
            config = {
                "configurable": {"thread_id": f"orch_{topic[:20]}"},
                "recursion_limit": 125
            }
            
            # Stream events from the graph and capture final state
            final_state = None
            async for event in self.compiled_graph.astream_events(
                state,
                config=config,
                version="v2"
            ):
                event_kind = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {})
                
                # Capture final state from finalize node
                if event_kind == "on_chain_end" and event_name == "finalize":
                    final_state = event_data.get("output", {})
                
                # Handle node events
                if event_kind == "on_chain_start" and event_name in [
                    "generator"
                ]:
                    pass
                    # # Generate appropriate message
                    # if "generator" in event_name:
                    #     message = f"⚙️ Processing {event_name}..."
                    # elif "discriminator" in event_name:
                    #     message = f"⚙️ Processing {event_name}..."
                    # else:
                    #     message = f"⚙️ Processing {event_name}..."
                    
                    # yield AgentResponse(
                    #     agent_id=event_name,
                    #     content=message,
                    #     iteration=iteration,
                    #     is_final=False
                    # )
                
                # Handle completion
                if event_kind == "on_chain_end" and event_name in ["generator", "discriminator"]:
                    output = event_data.get("output", {})
                    content = output.get("current_response", "")
                    
                    if content:
                        self._log_queue.put_nowait({
                            "agent_id": event_name,
                            "content": content
                        })
                        
                        preview = content[:1000] + "..." if len(content) > 1000 else content
                        yield AgentResponse(
                            agent_id=event_name,
                            content=preview,
                            iteration=output.get("iteration", 0),
                            is_final=False
                        )
                    
                    await asyncio.sleep(0.5)
            
            # Use captured final state or fallback to initial state
            if not final_state:
                final_state = state
            
            # Send final summary
            final_summary = final_state.get("messages", [])[-1].content if final_state.get("messages") else "Workflow complete"
            
            yield AgentResponse(
                agent_id="system",
                content=final_summary,
                iteration=final_state.get("iteration", 0),
                is_final=True
            )
            
            self._final_content = final_state.get("final_book", "")
            self._final_state = final_state
            
        except Exception as e:
            logger.error(f"Workflow error: {str(e)}", exc_info=True)
            yield AgentResponse(
                agent_id="system",
                content=f"Error: {str(e)}",
                iteration=0,
                is_final=True
            )
        finally:
            await self.shutdown()
    