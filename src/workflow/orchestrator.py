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
QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "60.0"))
MAX_REFINEMENT_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "3"))

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
                
                # Strip leading/trailing whitespace to ensure consistent formatting
                content = content.strip()
                
                # Wrap user content in ** markers
                if agent_id == "user":
                    content = f"**{content}**"
                
                async with asyncio.Lock():
                    with open(self._log_filepath, "a", encoding="utf-8") as f:
                        f.write(f"<{agent_id}>\n")
                        f.write(f"\n{content}\n")
                        f.write(f"\n</{agent_id}>\n\n")
                
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
        
        # Phase 1: Content generation nodes
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("discriminator", self._discriminator_node)

        # Phase 2: Evaluation nodes
        workflow.add_node("generator_evaluator", self._generator_evaluator)
        workflow.add_node("discriminator_evaluator", self._discriminator_evaluator)

        # Phase 3: Summarization node
        workflow.add_node("summarizer", self._summarizer_node)

        # Starting point
        workflow.add_edge(START, "generator")

        # Generator evaluation loop:
        # Generator -> Generator_Evaluator -> (back to Generator OR forward to Discriminator)
        workflow.add_edge("generator", "generator_evaluator")
        workflow.add_conditional_edges(
            "generator_evaluator",
            self._should_revise_generator,
            {
                "generator": "generator",  # Needs revision
                "discriminator": "discriminator"  # Quality met, proceed
            }
        )

        # Discriminator evaluation loop:
        # Discriminator -> Discriminator_Evaluator -> (back to Discriminator OR Generator OR Summarizer)
        workflow.add_edge("discriminator", "discriminator_evaluator")
        workflow.add_conditional_edges(
            "discriminator_evaluator",
            self._should_revise_discriminator,
            {
                "discriminator": "discriminator",  # Needs revision
                "generator": "generator",  # Continue conversation
                "summarizer": "summarizer"  # Max iterations reached
            }
        )

        # Final summary to end
        workflow.add_edge("summarizer", END)

        # Compile with memory
        memory = MemorySaver()
        self.compiled_graph = workflow.compile(checkpointer=memory)
        logger.info("✅ Workflow graph compiled successfully")
    

    async def _generator_evaluator(self, state: TopicFocusedState) -> dict:
        """Evaluate generator content and decide whether to revise or proceed"""
        logger.info("📊 GENERATOR EVALUATOR")
        
        current_response = state.get("current_response", "")
        generator_iterations = state.get("generator_iterations", 0)
        generator_revisions = state.get("generator_revisions", [])
        
        # Prepare previous content for novelty comparison if we have revisions
        previous_content = None
        if generator_revisions:
            previous_content = generator_revisions[-1][0]  # Last revision content
        
        # Use the EvaluatorAgent's sophisticated evaluation
        eval_state = {
            "previous_content": previous_content,
            "context": state.get("context", {})
        }
        
        # Call evaluator and get result
        evaluation_result = await self.generator_evaluator.evaluate(
            content=current_response,
            previous_content=previous_content,
            context=eval_state.get("context")
        )
        
        quality_score = evaluation_result.total_score
        
        # Store this revision with its score
        generator_revisions.append((current_response, quality_score))
        
        logger.info(f"Generator content quality: {quality_score:.1f}/100 ({evaluation_result.tier})")
        if evaluation_result.critical_failures:
            logger.warning(f"Critical failures: {evaluation_result.critical_failures}")
        
        # Use environment variables for thresholds
        meets_quality = quality_score >= QUALITY_THRESHOLD
        max_iterations_reached = generator_iterations >= MAX_REFINEMENT_ITERATIONS - 1
        
        if not meets_quality and not max_iterations_reached:
            # Need revision and haven't hit max iterations
            logger.info(f"Quality {quality_score:.1f} < {QUALITY_THRESHOLD}, requesting revision #{generator_iterations + 1}/{MAX_REFINEMENT_ITERATIONS}")
            
            return {
                "generator_revisions": generator_revisions,
                "generator_iterations": generator_iterations + 1,
                "evaluator_feedback": evaluation_result.detailed_feedback,
                "quality_scores": {**state.get("quality_scores", {}), f"generator_{generator_iterations}": quality_score},
                "evaluation": evaluation_result
            }
        else:
            # Either quality met or max iterations reached - select best content
            best_content, best_score = max(generator_revisions, key=lambda x: x[1])
            
            if meets_quality:
                logger.info(f"✅ Quality threshold met ({quality_score:.1f} >= {QUALITY_THRESHOLD})")
            else:
                logger.info(f"⚠️ Max iterations reached ({MAX_REFINEMENT_ITERATIONS}), using best score: {best_score:.1f}")
            
            logger.info(f"Selecting best generator content with score {best_score:.1f}")
            
            return {
                "generator_revisions": generator_revisions,
                "generator_iterations": generator_iterations,
                "best_generator_content": best_content,
                "current_response": best_content,
                "evaluator_feedback": None,
                "quality_scores": {**state.get("quality_scores", {}), "generator_final": best_score},
                "evaluation": evaluation_result
            }
    
    async def _discriminator_evaluator(self, state: TopicFocusedState) -> dict:
        """Evaluate discriminator response and decide routing"""
        logger.info("📊 DISCRIMINATOR EVALUATOR")
        
        current_response = state.get("current_response", "")
        discriminator_iterations = state.get("discriminator_iterations", 0)
        discriminator_revisions = state.get("discriminator_revisions", [])
        iterations = state.get("iterations", 0)
        
        # Prepare previous content for comparison if we have revisions
        previous_content = None
        if discriminator_revisions:
            previous_content = discriminator_revisions[-1][0]
        
        # Use the EvaluatorAgent's sophisticated evaluation
        eval_state = {
            "previous_content": previous_content,
            "context": state.get("context", {})
        }
        
        # Call evaluator and get result
        evaluation_result = await self.discriminator_evaluator.evaluate(
            content=current_response,
            previous_content=previous_content,
            context=eval_state.get("context")
        )
        
        quality_score = evaluation_result.total_score
        
        # Store this revision
        discriminator_revisions.append((current_response, quality_score))
        
        logger.info(f"Discriminator response quality: {quality_score:.1f}/100 ({evaluation_result.tier})")
        if evaluation_result.critical_failures:
            logger.warning(f"Critical failures: {evaluation_result.critical_failures}")
        
        # Use environment variables for thresholds
        max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))
        meets_quality = quality_score >= QUALITY_THRESHOLD
        max_refinements_reached = discriminator_iterations >= MAX_REFINEMENT_ITERATIONS - 1
        
        # First check if we need revision (quality not met and can still refine)
        if not meets_quality and not max_refinements_reached:
            logger.info(f"Quality {quality_score:.1f} < {QUALITY_THRESHOLD}, requesting revision #{discriminator_iterations + 1}/{MAX_REFINEMENT_ITERATIONS}")
            
            return {
                "discriminator_revisions": discriminator_revisions,
                "discriminator_iterations": discriminator_iterations + 1,
                "evaluator_feedback": evaluation_result.detailed_feedback,
                "quality_scores": {**state.get("quality_scores", {}), f"discriminator_rev_{discriminator_iterations}": quality_score},
                "evaluation": evaluation_result
            }
        
        # Quality met or max refinements reached - select best discriminator response
        best_response, best_score = max(discriminator_revisions, key=lambda x: x[1])
        
        if meets_quality:
            logger.info(f"✅ Quality threshold met ({quality_score:.1f} >= {QUALITY_THRESHOLD})")
        else:
            logger.info(f"⚠️ Max refinements reached ({MAX_REFINEMENT_ITERATIONS}), using best score: {best_score:.1f}")
        
        # Now decide routing based on conversation iterations
        if iterations < max_iterations:
            logger.info(f"Iteration {iterations + 1}/{max_iterations}: Continuing conversation")
            # Route back to generator for next iteration
            return {
                "discriminator_revisions": discriminator_revisions,
                "discriminator_iterations": 0,  # Reset for next round
                "generator_iterations": 0,  # Reset generator iterations
                "iterations": iterations + 1,
                "last_followup_question": best_response,  # Use best response
                "evaluator_feedback": None,
                "generator_revisions": [],  # Clear for next round
                "quality_scores": {**state.get("quality_scores", {}), f"discriminator_{iterations}": best_score},
                "current_response": best_response,  # Ensure best response is used
                "evaluation": evaluation_result
            }
        else:
            logger.info(f"Reached max iterations ({max_iterations}), routing to summarizer")
            # Route to summarizer with best response
            return {
                "discriminator_revisions": discriminator_revisions,
                "discriminator_iterations": discriminator_iterations,
                "iterations": iterations,
                "quality_scores": {**state.get("quality_scores", {}), "discriminator_final": best_score},
                "current_response": best_response,  # Ensure best response is used
                "evaluation": evaluation_result,
                "evaluator_feedback": None  # Clear feedback to ensure proper routing
            }
    
    def _should_revise_generator(self, state: TopicFocusedState) -> str:
        """Determine if generator needs revision or proceed to discriminator"""
        feedback = state.get("evaluator_feedback")
        if feedback:
            # Has feedback, needs revision
            return "generator"
        else:
            # No feedback, proceed to discriminator
            return "discriminator"
    
    def _should_revise_discriminator(self, state: TopicFocusedState) -> str:
        """Determine routing after discriminator evaluation"""
        feedback = state.get("evaluator_feedback")
        iterations = state.get("iterations", 0)
        max_iterations = int(os.getenv("MAX_ITERATIONS", "3"))
        
        if feedback:
            # Has feedback, needs revision
            return "discriminator"
        elif iterations >= max_iterations:
            # Reached iteration threshold, go to summarizer
            return "summarizer"
        else:
            # Continue conversation, back to generator
            return "generator"
    
    
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
        generator_iteration = state.get("generator_iterations", 0)
        logger.info(f"📋  GENERATOR iteration {iteration + 1}, revision {generator_iteration}")
        
        original_topic = state.get("original_topic", "")
        current_depth_level = state.get("current_depth_level", 1)
        aspects_explored = state.get("aspects_explored", [])
        topic_summary = state.get("topic_summary", "")
        last_followup_question = state.get("last_followup_question", "")
        evaluator_feedback = state.get("evaluator_feedback", "")
        
        # Determine the prompt
        if evaluator_feedback and generator_iteration > 0:
            # Include feedback for revision
            user_prompt = f"Previous response received feedback: {evaluator_feedback}\n\nPlease revise your response for: {last_followup_question if last_followup_question else original_topic}"
        else:
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
        discriminator_iteration = state.get("discriminator_iterations", 0)
        logger.info(f"📋  DISCRIMINATOR iteration {iteration + 1}, revision {discriminator_iteration}")
        
        # Use best generator content if available, otherwise current response
        best_generator_content = state.get("best_generator_content")
        generator_response = best_generator_content if best_generator_content else state.get("current_response", "")
        evaluator_feedback = state.get("evaluator_feedback", "")
        
        # Include feedback if this is a revision
        if evaluator_feedback and discriminator_iteration > 0:
            logger.info(f"Including evaluator feedback in discriminator prompt")
            # Modify the response based on feedback
            response, state_updates = await self.discriminator.process(
                f"Feedback on previous response: {evaluator_feedback}\n\nPlease revise your response to: {generator_response}", 
                state
            )
        else:
            response, state_updates = await self.discriminator.process(generator_response, state)
        
        should_end = "STOP" in response.upper()
        
        return {
            **state_updates,
            "current_response": response,
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
            current_response="",
            # Initialize quality tracking fields
            generator_revisions=[],
            discriminator_revisions=[],
            evaluator_feedback=None,
            quality_scores={},
            generator_iterations=0,
            discriminator_iterations=0,
            best_generator_content=None,
            evaluation=None,
            context={}
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
                
                # Handle evaluator completion - log only best revisions
                if event_kind == "on_chain_end" and event_name in ["generator_evaluator", "discriminator_evaluator"]:
                    output = event_data.get("output", {})
                    
                    # Check if this evaluator selected the best content (not requesting revision)
                    if not output.get("evaluator_feedback"):
                        # No feedback means best content was selected
                        best_content = None
                        agent_type = None
                        
                        if event_name == "generator_evaluator":
                            best_content = output.get("best_generator_content") or output.get("current_response")
                            agent_type = "generator"
                        elif event_name == "discriminator_evaluator":
                            best_content = output.get("current_response")
                            agent_type = "discriminator"
                        
                        if best_content and agent_type:
                            # Log the best revision
                            self._log_queue.put_nowait({
                                "agent_id": agent_type,
                                "content": best_content
                            })
                            
                            # Send preview to client
                            preview = best_content[:1000] + "..." if len(best_content) > 1000 else best_content
                            yield AgentResponse(
                                agent_id=agent_type,
                                content=preview,
                                iteration=output.get("iterations", 0),
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
    