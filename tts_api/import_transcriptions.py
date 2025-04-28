#!/usr/bin/env python3
"""
Script to import existing transcriptions from MongoDB into ChromaDB.
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
from bson import ObjectId
from vector_store import get_vector_store, VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "Speech-to-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "voice")

def connect_to_mongodb():
    """Connect to MongoDB and return the collection."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

def import_transcriptions(batch_size=50, limit=None):
    """Import transcriptions from MongoDB to ChromaDB."""
    # Connect to MongoDB
    collection = connect_to_mongodb()
    
    # Get vector store
    vector_store = get_vector_store()
    
    # Get collection stats before import
    stats_before = vector_store.get_collection_stats()
    logger.info(f"Vector store before import: {stats_before}")
    
    # Get total count
    total_count = collection.count_documents({})
    if limit:
        total_count = min(total_count, limit)
    logger.info(f"Found {total_count} transcriptions in MongoDB")
    
    # Process in batches
    processed = 0
    start_time = time.time()
    
    cursor = collection.find({}).limit(limit) if limit else collection.find({})
    
    batch = []
    for doc in cursor:
        batch.append(doc)
        
        if len(batch) >= batch_size:
            try:
                # Process batch
                vector_store.batch_add_transcriptions(batch)
                processed += len(batch)
                logger.info(f"Processed {processed}/{total_count} transcriptions ({processed/total_count*100:.2f}%)")
                batch = []
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
    
    # Process remaining items
    if batch:
        try:
            vector_store.batch_add_transcriptions(batch)
            processed += len(batch)
            logger.info(f"Processed {processed}/{total_count} transcriptions ({processed/total_count*100:.2f}%)")
        except Exception as e:
            logger.error(f"Error processing final batch: {e}")
    
    # Get collection stats after import
    stats_after = vector_store.get_collection_stats()
    logger.info(f"Vector store after import: {stats_after}")
    
    # Log summary
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Import completed in {duration:.2f} seconds")
    logger.info(f"Total documents processed: {processed}")
    logger.info(f"Documents per second: {processed/duration:.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Import transcriptions from MongoDB to ChromaDB")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of documents to process")
    
    args = parser.parse_args()
    
    logger.info(f"Starting import with batch size {args.batch_size}, limit {args.limit or 'none'}")
    import_transcriptions(batch_size=args.batch_size, limit=args.limit) 