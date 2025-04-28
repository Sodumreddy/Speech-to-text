import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector database for storing and retrieving transcription embeddings."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """Initialize the vector store with ChromaDB."""
        # Get configuration from environment variables or use defaults
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "transcriptions")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        
        # Create persistence directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with persistence - updated to new method
        logger.info(f"Initializing ChromaDB with persistence directory: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize embedding function using sentence-transformers
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        # Create or get collection
        logger.info(f"Creating/getting collection: {self.collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Transcription embeddings for semantic search"}
        )
        
        logger.info("Vector store initialized successfully")
    
    def add_transcription(self, transcription: Dict[str, Any]) -> str:
        """Add a transcription to the vector database."""
        try:
            # Extract text from utterances
            texts = [u["text"] for u in transcription.get("utterances", [])]
            full_text = " ".join(texts)
            
            if not full_text.strip():
                logger.warning("Empty transcription text, skipping")
                return None
            
            # Create metadata
            metadata = {
                "source_type": transcription.get("source_type", "unknown"),
                "timestamp": transcription["timestamp"].isoformat() if isinstance(transcription["timestamp"], datetime) else str(transcription["timestamp"]),
                "speaker_count": len(set(u["speaker"] for u in transcription.get("utterances", []))),
                "keywords": ",".join([kw[0] for kw in transcription.get("keywords", [])]),
                "mongodb_id": str(transcription.get("_id", transcription.get("id", ""))),
            }
            
            # Add to ChromaDB
            doc_id = str(transcription.get("_id", transcription.get("id", "")))
            
            # Check if document already exists
            existing_docs = self.collection.get(ids=[doc_id])
            if existing_docs and existing_docs["ids"]:
                logger.info(f"Document {doc_id} already exists, updating")
                self.collection.update(
                    ids=[doc_id],
                    documents=[full_text],
                    metadatas=[metadata]
                )
            else:
                logger.info(f"Adding new document with ID: {doc_id}")
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
              max_date: str = None,
              min_speaker_count: int = None,
              keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant transcriptions with enhanced filtering."""
        try:
            # Build where clause for filtering
            where = {}
            if source_type:
                where["source_type"] = {"$eq": source_type}
            
            # Date filtering
            if min_date or max_date:
                date_filter = {}
                if min_date:
                    date_filter["$gte"] = min_date
                if max_date:
                    date_filter["$lte"] = max_date
                if date_filter:
                    where["timestamp"] = date_filter
            
            # Speaker count filtering
            if min_speaker_count is not None:
                where["speaker_count"] = {"$gte": min_speaker_count}
            
            # Keyword filtering
            if keywords and len(keywords) > 0:
                # Using contains operator to find documents with any of the specified keywords
                keyword_filters = []
                for keyword in keywords:
                    keyword_filters.append({"$contains": keyword})
                
                where["keywords"] = {"$and": keyword_filters}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where if where else None
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": float(results["distances"][0][i]) if "distances" in results and results["distances"] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector DB: {str(e)}")
            raise
    
    def get_similar_transcriptions(self, transcription_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find transcriptions similar to a given one."""
        try:
            # Get the original transcription
            result = self.collection.get(ids=[transcription_id])
            
            if not result["documents"]:
                raise ValueError(f"Transcription {transcription_id} not found")
            
            # Use the text to find similar documents
            results = self.collection.query(
                query_texts=[result["documents"][0]],
                n_results=n_results + 1  # Add 1 because the original document will be included
            )
            
            # Format and filter out the original document
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    if results["ids"][0][i] != transcription_id:
                        formatted_results.append({
                            "id": results["ids"][0][i],
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": float(results["distances"][0][i]) if "distances" in results and results["distances"] else None
                        })
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar transcriptions: {str(e)}")
            raise
    
    def batch_add_transcriptions(self, transcriptions: List[Dict[str, Any]]) -> List[str]:
        """Add multiple transcriptions to the vector database in a single batch."""
        try:
            documents = []
            metadatas = []
            ids = []
            
            for transcription in transcriptions:
                # Extract text from utterances
                texts = [u["text"] for u in transcription.get("utterances", [])]
                full_text = " ".join(texts)
                
                if not full_text.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source_type": transcription.get("source_type", "unknown"),
                    "timestamp": transcription["timestamp"].isoformat() if isinstance(transcription["timestamp"], datetime) else str(transcription["timestamp"]),
                    "speaker_count": len(set(u["speaker"] for u in transcription.get("utterances", []))),
                    "keywords": ",".join([kw[0] for kw in transcription.get("keywords", [])]),
                    "mongodb_id": str(transcription.get("_id", transcription.get("id", ""))),
                }
                
                # Add to batch
                doc_id = str(transcription.get("_id", transcription.get("id", "")))
                documents.append(full_text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if not ids:
                logger.warning("No valid transcriptions to add")
                return []
            
            # Add batch to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(ids)} transcriptions to vector DB")
            return ids
            
        except Exception as e:
            logger.error(f"Error batch adding transcriptions: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def extract_common_topics(self, max_topics: int = 20) -> List[Dict[str, Any]]:
        """
        Extract common topics from all transcriptions.
        
        Args:
            max_topics: Maximum number of topics to return
            
        Returns:
            List of dictionaries with topic and count
        """
        try:
            # Get all transcriptions
            results = self.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                logger.warning("No transcriptions found")
                return []
            
            # Extract all keywords
            all_keywords = []
            for metadata in results["metadatas"]:
                if "keywords" in metadata:
                    keywords = metadata["keywords"].split(",")
                    all_keywords.extend([kw.strip() for kw in keywords if kw.strip()])
            
            # Count keyword frequencies
            keyword_counts = Counter(all_keywords)
            
            # Return top topics
            top_topics = []
            for topic, count in keyword_counts.most_common(max_topics):
                top_topics.append({
                    "topic": topic,
                    "count": count
                })
            
            return top_topics
            
        except Exception as e:
            logger.error(f"Error extracting common topics: {str(e)}")
            raise
            
    def get_transcriptions_by_date_range(self, 
                                         start_date: str, 
                                         end_date: str, 
                                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get transcriptions within a date range.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            limit: Maximum number of transcriptions to return
            
        Returns:
            List of transcriptions
        """
        try:
            # Build where clause for date filtering
            where = {
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # Get transcriptions
            results = self.collection.get(
                where=where,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["ids"])):
                formatted_results.append({
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting transcriptions by date range: {str(e)}")
            raise
            
    def get_source_distribution(self) -> Dict[str, int]:
        """
        Get distribution of transcriptions by source type.
        
        Returns:
            Dictionary with source types as keys and counts as values
        """
        try:
            # Get all transcriptions
            results = self.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                logger.warning("No transcriptions found")
                return {}
            
            # Count source types
            source_counts = Counter()
            for metadata in results["metadatas"]:
                source_type = metadata.get("source_type", "unknown")
                source_counts[source_type] += 1
            
            return dict(source_counts)
            
        except Exception as e:
            logger.error(f"Error getting source distribution: {str(e)}")
            raise
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about speakers across all transcriptions.
        
        Returns:
            Dictionary with statistics about speakers
        """
        try:
            # Get all transcriptions
            results = self.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                logger.warning("No transcriptions found")
                return {
                    "avg_speakers_per_transcription": 0,
                    "max_speakers": 0,
                    "speaker_count_distribution": {}
                }
            
            # Analyze speaker counts
            speaker_counts = []
            for metadata in results["metadatas"]:
                speaker_count = metadata.get("speaker_count", 0)
                if isinstance(speaker_count, str):
                    try:
                        speaker_count = int(speaker_count)
                    except ValueError:
                        speaker_count = 0
                speaker_counts.append(speaker_count)
            
            # Calculate statistics
            avg_speakers = sum(speaker_counts) / len(speaker_counts) if speaker_counts else 0
            max_speakers = max(speaker_counts) if speaker_counts else 0
            
            # Get distribution
            speaker_count_distribution = Counter(speaker_counts)
            
            return {
                "avg_speakers_per_transcription": round(avg_speakers, 2),
                "max_speakers": max_speakers,
                "speaker_count_distribution": dict(speaker_count_distribution)
            }
            
        except Exception as e:
            logger.error(f"Error getting speaker statistics: {str(e)}")
            raise

# Create a singleton instance
vector_store = VectorStore()

def get_vector_store() -> VectorStore:
    """Get the singleton vector store instance."""
    return vector_store 