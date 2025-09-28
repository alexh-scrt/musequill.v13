import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from ollama import AsyncClient, Client, ListResponse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "localhost")
        self.port = os.getenv("OLLAMA_PORT", "11434")
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Create both sync and async clients
        self.client = Client(host=self.base_url)
        self.async_client = AsyncClient(host=self.base_url)
        
        # Default models
        self.writer_model = os.getenv("WRITER_MODEL", "llama3.3:70b")
        self.editor_model = os.getenv("EDITOR_MODEL", "qwen3:32b")
        
        logger.info(f"Initialized Ollama client with base URL: {self.base_url}")
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = await self.async_client.list()
            models = [model.model for model in response.models]
            # models = [model['name'] for model in response.get('models', [])]
            logger.info(f"Available models: {models}")
            return models
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> AsyncGenerator[str, None] | str:
        """Generate text using Ollama"""
        
        if model is None:
            model = self.writer_model
        
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            if stream:
                # Return async generator for streaming
                async def stream_generator():
                    stream = await self.async_client.chat(
                        model=model,
                        messages=messages,
                        options=options,
                        stream=True
                    )
                    async for chunk in stream:
                        if chunk.get('message', {}).get('content'):
                            yield chunk['message']['content']
                
                return stream_generator()
            else:
                # Return complete response
                response = await self.async_client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=False
                )
                return response['message']['content']
                
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    async def generate_with_context(
        self,
        prompt: str,
        context: str,
        model: Optional[str] = None,
        role: str = "writer",
        temperature: float = 0.7
    ) -> str:
        """Generate text with context"""
        
        if role == "writer":
            system_prompt = (
                "You are a skilled content writer. Your task is to create engaging, "
                "well-structured content based on the given topic and any provided context. "
                "Focus on clarity, coherence, and creativity."
            )
            model = model or self.writer_model
        elif role == "editor":
            system_prompt = (
                "You are an expert editor. Your task is to review and improve the provided content. "
                "Focus on enhancing clarity, fixing errors, improving flow, and ensuring the content "
                "is engaging and well-structured. Provide constructive feedback and refined content."
            )
            model = model or self.editor_model
        else:
            system_prompt = "You are a helpful AI assistant."
            model = model or self.writer_model
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"
        
        return await self.generate(
            prompt=full_prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            stream=False
        )
    
    def check_connection(self) -> bool:
        """Check if Ollama is accessible"""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection check failed: {str(e)}")
            return False