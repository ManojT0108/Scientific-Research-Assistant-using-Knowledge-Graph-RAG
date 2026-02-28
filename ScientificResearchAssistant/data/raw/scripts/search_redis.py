"""
Fast Redis-based semantic search for ArXiv papers
Uses Redis vector similarity search for sub-second query times
"""

import argparse
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
INDEX_NAME = "arxiv_chunks_idx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def search_redis(client, model, query_text, k=5, filters=None):
    """
    Perform vector similarity search in Redis
    
    Args:
        client: Redis client
        model: SentenceTransformer model
        query_text: Search query
        k: Number of results to return
        filters: Optional dict with filters like {'year': 2024, 'section': 'abstract'}
    """
    # Encode query
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]
    query_bytes = query_vec.astype(np.float32).tobytes()
    
    # Build query with optional filters
    filter_clause = ""
    if filters:
        conditions = []
        if 'year' in filters:
            conditions.append(f"@year:[{filters['year']} {filters['year']}]")
        if 'section' in filters:
            conditions.append(f"@section:{{{filters['section']}}}")
        if 'categories' in filters:
            # Tag field search
            conditions.append(f"@categories:{{{filters['categories']}}}")
        
        if conditions:
            filter_clause = " ".join(conditions) + " "
    
    # Create KNN query with filters
    query_str = f"{filter_clause}*=>[KNN {k} @embedding $vec AS score]"
    q = (
        Query(query_str)
        .return_fields("chunk_id", "paper_id", "title", "year", "section", "chunk_index", "url", "categories", "text", "score")
        .sort_by("score")
        .dialect(2)
    )
    
    # Execute search
    results = client.ft(INDEX_NAME).search(q, query_params={"vec": query_bytes})
    
    return results

def format_results(results, query_text):
    """Format and display search results"""
    print(f"\n{'='*100}")
    print(f"Query: {query_text}")
    print(f"Found {results.total} results in {results.duration}ms")
    print(f"{'='*100}\n")
    
    for i, doc in enumerate(results.docs, 1):
        # Redis returns bytes, need to decode
        title = doc.title.decode('utf-8') if isinstance(doc.title, bytes) else doc.title
        section = doc.section.decode('utf-8') if isinstance(doc.section, bytes) else doc.section
        url = doc.url.decode('utf-8') if isinstance(doc.url, bytes) else doc.url
        text = doc.text.decode('utf-8') if isinstance(doc.text, bytes) else doc.text
        categories = doc.categories.decode('utf-8') if isinstance(doc.categories, bytes) else doc.categories
        
        print(f"{i}. [Score: {float(doc.score):.4f}] {title}")
        print(f"   Year: {doc.year} | Section: {section} | Chunk: {doc.chunk_index}")
        print(f"   Categories: {categories}")
        print(f"   URL: {url}")
        
        # Display text snippet (first 200 chars)
        if text:
            snippet = text[:200] + "..." if len(text) > 200 else text
            print(f"   Text: {snippet}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Fast semantic search using Redis")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--year", type=int, help="Filter by year")
    parser.add_argument("--section", type=str, help="Filter by section")
    parser.add_argument("--category", type=str, help="Filter by category (e.g., cs.LG)")
    
    args = parser.parse_args()
    query_text = " ".join(args.query)
    
    # Build filters
    filters = {}
    if args.year:
        filters['year'] = args.year
    if args.section:
        filters['section'] = args.section
    if args.category:
        filters['categories'] = args.category
    
    # Connect to Redis
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        client.ping()
    except redis.ConnectionError:
        print("❌ Could not connect to Redis. Make sure Redis is running:")
        print("   docker run -d -p 6379:6379 redis/redis-stack:latest")
        return
    
    # Load model (cached after first run)
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    
    # Search
    results = search_redis(client, model, query_text, k=args.k, filters=filters)
    
    # Display results
    format_results(results, query_text)
    
    # Show filter info if used
    if filters:
        print(f"Applied filters: {filters}")

if __name__ == "__main__":
    main()