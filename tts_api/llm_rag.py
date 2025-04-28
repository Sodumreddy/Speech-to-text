import os
import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
from langchain.schema import Document
from vector_store import get_vector_store

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LM Studio configuration
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "llama3")

class LLMRagEngine:
    """
    LLM RAG Engine using LM Studio with Llama 3.2
    
    This class integrates the vector store with LM Studio to provide
    contextual responses based on transcriptions.
    """
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize the LLM RAG Engine."""
        self.base_url = base_url or LM_STUDIO_BASE_URL
        self.model = model or LM_STUDIO_MODEL
        self.vector_store = get_vector_store()
        logger.info(f"Initialized LLM RAG Engine with model: {self.model}")
        
    def query(self, 
             query: str, 
             context: Optional[str] = None,
             n_results: int = 3,
             temperature: float = 0.7,
             max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Query the LLM with RAG context.
        
        Args:
            query: The user's query
            context: Optional additional context
            n_results: Number of results to fetch from vector store
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with LLM response and metadata
        """
        try:
            # Step 1: Get relevant documents from vector store
            relevant_docs = []
            if query:
                search_results = self.vector_store.search(query=query, n_results=n_results)
                for result in search_results:
                    relevant_docs.append(result["text"])
            
            # Step 2: Construct prompt with retrieved context
            rag_context = "\n\n".join(relevant_docs)
            
            # Include additional context if provided
            if context:
                rag_context = f"{context}\n\n{rag_context}"
            
            # Create system prompt that instructs the model on how to use the context
            system_prompt = (
                "You are an AI assistant that helps with analyzing call transcriptions and conversations. "
                "Use the following pieces of information from previous conversations to inform your response. "
                "If the information doesn't contain the answer, just say you don't know. "
                "Don't make up an answer. If you need more information, ask clarifying questions."
            )
            
            # Step 3: Call LM Studio API
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context information:\n{rag_context}\n\nQuestion: {query}"}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.info(f"Sending request to LM Studio at {endpoint}")
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Step 4: Extract the text response
            assistant_message = result["choices"][0]["message"]["content"]
            
            return {
                "query": query,
                "response": assistant_message,
                "context_used": rag_context,
                "model": self.model,
                "raw_response": result
            }
            
        except Exception as e:
            logger.error(f"Error querying LLM with RAG: {str(e)}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "error": True
            }
    
    def extract_insights(self, transcription_text: str, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Extract insights from transcription text.
        
        Args:
            transcription_text: The transcription text
            keywords: Optional list of keywords to focus on
            
        Returns:
            Dictionary with insights
        """
        try:
            # Create a prompt that asks for insights based on the transcription
            system_prompt = (
                "You are an AI assistant that analyzes conversation transcripts. "
                "Extract key insights, action items, and important points from the transcript provided."
            )
            
            user_prompt = f"Please analyze this transcript and provide key insights:\n\n{transcription_text}"
            
            if keywords and len(keywords) > 0:
                keyword_str = ", ".join(keywords)
                user_prompt += f"\n\nPay special attention to these keywords: {keyword_str}"
            
            # Call LM Studio API
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,  # Lower temperature for more factual responses
                "max_tokens": 1024
            }
            
            logger.info("Extracting insights from transcription")
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the text response
            insights = result["choices"][0]["message"]["content"]
            
            return {
                "insights": insights,
                "raw_response": result
            }
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return {
                "insights": f"Error extracting insights: {str(e)}",
                "error": True
            }
    
    def answer_from_live_transcription(self, 
                                      transcription_text: str, 
                                      query: Optional[str] = None,
                                      keywords: List[str] = None) -> Dict[str, Any]:
        """
        Generate contextual answers from live transcription.
        
        Args:
            transcription_text: The current live transcription text
            query: Optional specific query to answer
            keywords: Keywords extracted from the transcription
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # If no query provided, generate a generic analysis
            if not query:
                query = "What are the key points and insights from this conversation?"
            
            # Get similar transcriptions for additional context
            search_results = self.vector_store.search(
                query=transcription_text[:200],  # Use beginning of transcription for context
                n_results=2
            )
            
            additional_context = ""
            if search_results:
                additional_context = "Similar conversations from the past:\n"
                for i, result in enumerate(search_results):
                    additional_context += f"Conversation {i+1}: {result['text'][:300]}...\n\n"
            
            # Create a system prompt that instructs how to respond
            system_prompt = (
                "You are an AI assistant participating in a live conversation. "
                "Use the current conversation and relevant similar conversations from the past to provide helpful information. "
                "Keep your answers concise and relevant to the current conversation. "
                "Don't refer to yourself as an AI or assistant in your response."
            )
            
            # Prepare user prompt with transcription and query
            user_prompt = f"Current conversation transcript:\n{transcription_text}\n\n"
            
            if additional_context:
                user_prompt += f"{additional_context}\n\n"
                
            if keywords and len(keywords) > 0:
                keyword_str = ", ".join([k for k in keywords if isinstance(k, str)] + 
                                       [k[0] for k in keywords if isinstance(k, (list, tuple))])
                user_prompt += f"Keywords identified: {keyword_str}\n\n"
                
            user_prompt += f"Question: {query}"
            
            # Call LM Studio API
            endpoint = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 512  # Shorter response for live context
            }
            
            logger.info("Generating response for live transcription")
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the text response
            answer = result["choices"][0]["message"]["content"]
            
            return {
                "query": query,
                "response": answer,
                "keywords_used": keywords,
                "raw_response": result
            }
            
        except Exception as e:
            logger.error(f"Error answering from live transcription: {str(e)}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "error": True
            }

# Create a singleton instance
llm_rag_engine = LLMRagEngine()

def get_llm_rag_engine() -> LLMRagEngine:
    """Get the singleton LLM RAG engine instance."""
    return llm_rag_engine 