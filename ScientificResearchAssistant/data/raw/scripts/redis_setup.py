"""
Redis Vector Store Setup for ArXiv Semantic Search
Migrates embeddings and metadata to Redis for fast vector similarity search
"""

import json
import os
import numpy as np
import redis
from redis.commands.search.field import TextField, NumericField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tqdm import tqdm

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

EMB_PATH = "data/processed/embeddings.npy"
CHUNK_CANDIDATES = [
    "data/processed/chunks_full.jsonl",
    "data/processed/chunks.jsonl",
]

INDEX_NAME = "arxiv_chunks_idx"
VECTOR_DIM = 384
DISTANCE_METRIC = "COSINE"  

def load_data():
    """Load embeddings and chunk data (which includes text and categories)"""
    print("Loading embeddings and chunk data...")
    embeddings = np.load(EMB_PATH)
    chunks_path = next((p for p in CHUNK_CANDIDATES if os.path.exists(p)), None)
    if chunks_path is None:
        raise FileNotFoundError(
            "No chunk file found. Expected one of: "
            + ", ".join(CHUNK_CANDIDATES)
        )
    
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    assert len(embeddings) == len(chunks), "Mismatch between embeddings and chunks"
    print(f"Loaded {len(embeddings)} chunks from {chunks_path}")
    return embeddings, chunks

def setup_redis_index(client):
    """Create Redis search index with vector field"""
    try:
        # Drop existing index if it exists
        client.ft(INDEX_NAME).dropindex(delete_documents=True)
        print(f"Dropped existing index: {INDEX_NAME}")
    except:
        print(f"No existing index to drop")
    
    # Define schema
    schema = (
        TextField("chunk_id"),
        TextField("paper_id"),
        TextField("title"),
        TextField("text"),
        NumericField("year"),
        TextField("section"),
        NumericField("chunk_index"),
        TextField("url"),
        TagField("categories"),
        VectorField(
            "embedding",
            "FLAT",  # Use FLAT for <100k vectors, HNSW for larger datasets
            {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIM,
                "DISTANCE_METRIC": DISTANCE_METRIC,
            }
        ),
    )
    
    # Create index
    client.ft(INDEX_NAME).create_index(
        fields=schema,
        definition=IndexDefinition(prefix=["chunk:"], index_type=IndexType.HASH)
    )
    print(f"✅ Created index: {INDEX_NAME}")

def populate_redis(client, embeddings, chunks):
    """Insert all chunks into Redis"""
    print("Populating Redis...")
    pipeline = client.pipeline(transaction=False)
    
    for i, (emb, chunk) in enumerate(tqdm(zip(embeddings, chunks), total=len(embeddings))):
        key = f"chunk:{chunk['chunk_id']}"
        
        # Extract categories from nested meta field
        categories = chunk.get("meta", {}).get("categories", [])
        categories_str = ",".join(categories) if categories else ""
        
        # Prepare document
        doc = {
            "chunk_id": chunk["chunk_id"],
            "paper_id": chunk.get("paper_id", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),  # NOW we have the actual text!
            "year": chunk.get("year", 0) or 0,
            "section": chunk.get("section", ""),
            "chunk_index": chunk.get("chunk_index", 0),
            "url": chunk.get("meta", {}).get("url", ""),  # URL is in nested meta
            "categories": categories_str,
            "embedding": emb.astype(np.float32).tobytes(),  # Store as binary
        }
        
        pipeline.hset(key, mapping=doc)
        
        # Execute in batches
        if (i + 1) % 1000 == 0:
            pipeline.execute()
            pipeline = client.pipeline(transaction=False)
    
    # Execute remaining
    pipeline.execute()
    print(f"✅ Inserted {len(embeddings)} chunks into Redis")

def test_search(client, query_text="knowledge graph rag in manufacturing"):
    """Test vector similarity search"""
    from sentence_transformers import SentenceTransformer
    
    print(f"\n--- Testing search: '{query_text}' ---")
    
    # Encode query
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]
    query_bytes = query_vec.astype(np.float32).tobytes()
    
    # Create KNN query
    q = (
        Query(f"*=>[KNN 5 @embedding $vec AS score]")
        .return_fields("chunk_id", "title", "year", "section", "score")
        .sort_by("score")
        .dialect(2)
    )
    
    # Execute search
    results = client.ft(INDEX_NAME).search(q, query_params={"vec": query_bytes})
    
    print(f"Found {results.total} results in {results.duration}ms:\n")
    for i, doc in enumerate(results.docs, 1):
        print(f"{i}. [{doc.score}] {doc.title}")
        print(f"   {doc.year} • {doc.section} • {doc.chunk_id}\n")

def main():
    # Connect to Redis
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    
    try:
        client.ping()
        print("✅ Connected to Redis")
    except redis.ConnectionError:
        print("❌ Could not connect to Redis. Make sure Redis is running:")
        print("   docker run -d -p 6379:6379 redis/redis-stack:latest")
        return
    
    # Load data
    embeddings, chunks = load_data()
    
    # Setup index
    setup_redis_index(client)
    
    # Populate data
    populate_redis(client, embeddings, chunks)
    
    # Test search
    test_search(client, "knowledge graph rag in manufacturing")
    
    print("\n✅ Redis setup complete!")
    print(f"   Index: {INDEX_NAME}")
    print(f"   Documents: {len(embeddings)}")
    print(f"   Vector dimension: {VECTOR_DIM}")

if __name__ == "__main__":
    main()
