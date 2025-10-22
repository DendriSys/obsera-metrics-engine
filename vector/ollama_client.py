"""
Ollama Client for Embeddings and AI Insights
Integrates with Ollama for vector embeddings and text generation
"""

import httpx
from typing import List, Optional, Dict, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection"""
    url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    generation_model: str = "codellama:13b"
    timeout: int = 120


class OllamaClient:
    """
    Client for Ollama LLM service
    
    Features:
    - Generate embeddings for text (768-dimensional vectors)
    - Generate AI insights using LLM
    - Async HTTP client for performance
    """
    
    def __init__(self, config: OllamaConfig):
        """
        Initialize Ollama client
        
        Args:
            config: OllamaConfig with connection details
        """
        self.config = config
        self.base_url = config.url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=config.timeout)
        logger.info(f"Ollama client initialized: {self.base_url}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text
        
        Args:
            text: Text to embed
        
        Returns:
            768-dimensional embedding vector
        
        Example:
            >>> embedding = await client.generate_embedding(
            ...     "CPU usage metric showing high values"
            ... )
            >>> len(embedding)
            768
        """
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.config.embedding_model,
                "prompt": text
            }
            
            logger.debug(f"Generating embedding for text ({len(text)} chars)")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding", [])
            
            if not embedding:
                raise ValueError("No embedding returned from Ollama")
            
            logger.info(f"Generated {len(embedding)}-dimensional embedding")
            return embedding
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error generating embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_insights(
        self,
        context: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate AI insights using LLM
        
        Args:
            context: Context information (metric features, anomalies, etc.)
            prompt_template: Optional custom prompt template
        
        Returns:
            AI-generated insights text
        
        Example:
            >>> context = "CPU usage: mean=85%, trend=increasing +15%"
            >>> insights = await client.generate_insights(context)
        """
        try:
            # Default prompt template
            if not prompt_template:
                prompt_template = """Analyze this metric data and provide insights:

{context}

Provide:
1. Key observations
2. Potential issues
3. Recommended actions

Be concise and actionable (max 5 sentences)."""
            
            prompt = prompt_template.format(context=context)
            
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.config.generation_model,
                "prompt": prompt,
                "stream": False
            }
            
            logger.debug("Generating AI insights...")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            insights = data.get("response", "")
            
            logger.info(f"Generated insights ({len(insights)} chars)")
            return insights.strip()
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            # Return empty string on error instead of failing
            return ""
    
    async def health_check(self) -> bool:
        """
        Check if Ollama is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url)
            response.raise_for_status()
            logger.info("Ollama health check passed")
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
        logger.info("Ollama client closed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
