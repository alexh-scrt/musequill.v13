from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import os
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.tools import BaseTool, tool
from src.storage.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, web_search: bool = True, model: Optional[str] = None, session_id: Optional[str] = None, llm_params: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.model = model or os.getenv(f'{self.agent_id.upper()}_MODEL') or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        if not self.model:
            raise ValueError(f"Model not specified for agent {self.agent_id}")
        self.conversation_history = []
        self.session_id = session_id
        self.memory = MemoryManager(session_id, agent_id) if session_id else None
        
        default_params = {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": self.model,
            "temperature": 0.7
        }
        if llm_params:
            default_params.update(llm_params)
        
        self.llm = ChatOllama(**default_params)
        self.tools = [] if not web_search else self._setup_tools()

    def _setup_tools(self) -> List[BaseTool]:
        """Setup tools for the planner agent"""
        tools = []
        
        # Add web search tool for research
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                tavily_tool = TavilySearch(
                    api_key=tavily_api_key,
                    max_results=5,
                    include_answer=True,
                    include_raw_content=False,
                    include_images=False,
                    include_image_descriptions=False,
                    search_depth="basic",
                    include_domains=None,
                    exclude_domains=None,
                    include_favicon=False
                )
                tools.append(tavily_tool)
                logger.info("✅ Tavily search tool added for GeneratorAgent")
            except Exception as e:
                logger.error(f"❌ Failed to setup Tavily for GeneratorAgent: {e}")
        else:
            logger.warning("⚠️ TAVILY_API_KEY not set - research capabilities disabled for GeneratorAgent")
                
        return tools
        
    @abstractmethod
    async def process(self, prompt: str, context: Optional[str] = None) -> str:
        """Process input and generate response"""
        pass

    async def add_to_history(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        if self.memory:
            await self.memory.add_turn(role, content, metadata)
    
    def get_history(self) -> list:
        """Get conversation history"""
        return self.conversation_history
    
    async def get_recent_context(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation turns from memory"""
        if self.memory:
            return await self.memory.get_recent_context(n)
        return self.conversation_history[-n:]
    
    def get_relevant_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get semantically relevant conversation turns"""
        if self.memory:
            return self.memory.get_relevant_context(query, k)
        return []
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def __str__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, model={self.model})"