import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, collection_name: str = "transcriptions"):
        """Initialize RAG engine with ChromaDB."""
        self.client = chromadb.Client()
        
        # Use all-MiniLM-L6-v2 for embeddings (same as your existing KeyBERT model)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_transcription(self, transcription: Dict[str, Any]) -> str:
        """Add a transcription to the vector database."""
        try:
            # Extract text from utterances
            texts = [u["text"] for u in transcription["utterances"]]
            full_text = " ".join(texts)
            
            # Create metadata
            metadata = {
                "source_type": transcription.get("source_type", "unknown"),
                "timestamp": transcription["timestamp"].isoformat() if isinstance(transcription["timestamp"], datetime) else transcription["timestamp"],
                "speaker_count": len(set(u["speaker"] for u in transcription["utterances"])),
                "keywords": ",".join([kw[0] for kw in transcription.get("keywords", [])]),
            }
            
            # Add to ChromaDB
            doc_id = str(transcription.get("_id", transcription.get("id")))
            self.collection.add(
                documents=[full_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding transcription to vector DB: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              n_results: int = 5, 
              source_type: str = None,
              min_date: str = None,
              max_date: str = None) -> List[Dict[str, Any]]:
        """Search for relevant transcriptions."""
        try:
            # Build where clause for filtering
            where = {}
            if source_type:
                where["source_type"] = source_type
            if min_date:
                where["timestamp"] = {"$gte": min_date}
            if max_date:
                where["timestamp"] = {"$lte": max_date}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": float(results["distances"][0][i]) if "distances" in results else None
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            raise
    
    def get_similar_transcriptions(self, transcription_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find transcriptions similar to a given one."""
        try:
            # Get the original transcription
            result = self.collection.get(
                ids=[transcription_id],
                include=["documents", "embeddings"]
            )
            
            if not result["documents"]:
                raise ValueError(f"Transcription {transcription_id} not found")
            
            # Use the embedding to find similar documents
            results = self.collection.query(
                query_embeddings=[result["embeddings"][0]],
                n_results=n_results + 1  # Add 1 because the original document will be included
            )
            
            # Format and filter out the original document
            formatted_results = []
            for i in range(len(results["ids"][0])):
                if results["ids"][0][i] != transcription_id:
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": float(results["distances"][0][i]) if "distances" in results else None
                    })
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar transcriptions: {str(e)}")
            raise
    
    def extract_topics(self, transcription_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Extract common topics from transcriptions."""
        try:
            # Get all documents or specific ones
            if transcription_ids:
                results = self.collection.get(
                    ids=transcription_ids,
                    include=["documents", "metadatas"]
                )
            else:
                results = self.collection.get()
            
            # Combine all keywords
            all_keywords = []
            for metadata in results["metadatas"]:
                keywords = metadata.get("keywords", "").split(",")
                all_keywords.extend([k for k in keywords if k])
            
            # Count keyword frequencies
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            
            # Return top topics
            return [
                {"topic": topic, "count": count}
                for topic, count in keyword_counts.most_common(10)
            ]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            raise 