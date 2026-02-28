"""
Script 7: Hybrid Search - Combining Vector Search + Knowledge Graph

Implements:
- Vector search (Redis semantic similarity)
- Graph expansion (related papers through shared entities)
- Score fusion (combine vector + graph scores)
- Re-ranking strategies

Runtime: < 1 second per query
"""

import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import os
import argparse

INDEX_NAME = "arxiv_chunks_idx"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

class HybridSearchEngine:
    """
    Hybrid search combining semantic vectors and knowledge graph
    
    Three-stage pipeline:
    1. Vector search: Get top-k semantically similar chunks
    2. Graph expansion: Find related papers through shared entities
    3. Score fusion: Combine and re-rank results
    """
    
    def __init__(self, redis_host=None, redis_port=None):
        """Initialize search engine"""
        redis_host = redis_host or REDIS_HOST
        redis_port = redis_port or REDIS_PORT
        
        print("Initializing Hybrid Search Engine...")
        
        # Connect to Redis
        print("  Connecting to Redis...")
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False
        )
        self.redis_client.ping()
        print("  ✅ Redis connected")
        
        # Load embedding model
        print("  Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✅ Model loaded")
        
        # Load knowledge graph
        print("  Loading knowledge graph...")
        with open('data/processed/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        print(f"  ✅ Graph loaded ({self.graph.number_of_nodes():,} nodes)")
        
        # Load paper connections
        print("  Loading paper connections...")
        with open('data/processed/paper_connections.json') as f:
            self.paper_connections = json.load(f)
        print(f"  ✅ Connections loaded")
        
        # Load entities for filtering
        print("  Loading entities...")
        with open('data/processed/entities.json') as f:
            self.entities = json.load(f)
        print(f"  ✅ Entities loaded")
        
        print("\n✅ Hybrid Search Engine ready!\n")
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Stage 1: Vector search using Redis
        
        Returns: List of {paper_id, chunk_id, title, text, score}
        """
        
        # Encode query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search Redis
        from redis.commands.search.query import Query
        
        try:
            results = self.redis_client.ft(INDEX_NAME).search(
                Query(f'*=>[KNN {top_k} @embedding $vec AS score]')
                    .sort_by('score')
                    .return_fields('paper_id', 'chunk_id', 'title', 'section', 'text', 'score')
                    .dialect(2),
                query_params={'vec': query_embedding.astype(np.float32).tobytes()}
            )
        except Exception as e:
            print(f"⚠️  Vector search failed: {e}")
            return []
        
        # Parse results
        vector_results = []
        for doc in results.docs:
            vector_results.append({
                'paper_id': doc.paper_id.decode() if isinstance(doc.paper_id, bytes) else doc.paper_id,
                'chunk_id': doc.chunk_id.decode() if isinstance(doc.chunk_id, bytes) else doc.chunk_id,
                'title': doc.title.decode() if isinstance(doc.title, bytes) else doc.title,
                'section': doc.section.decode() if isinstance(doc.section, bytes) else doc.section,
                'text': doc.text.decode() if isinstance(doc.text, bytes) else doc.text,
                'vector_score': float(doc.score),
                'source': 'vector'
            })
        
        return vector_results
    
    def graph_expansion(self, seed_papers: List[str], max_expand: int = 10) -> List[Tuple[str, float]]:
        """
        Stage 2: IMPROVED Query-aware graph expansion
        
        Find related papers through shared entities with:
        1. Entity type weighting (tasks=3x, datasets=2.5x, methods=1x)
        2. Generic entity penalization (arxiv, github, attention → 0.3x)
        3. Task overlap requirement (requires shared tasks OR high score)
        
        Returns: List of (paper_id, graph_score)
        """
        
        # Entity weights by type specificity
        ENTITY_WEIGHTS = {
            'task': 3.0,      # Highly specific (recommendation, retrieval, classification)
            'dataset': 2.5,   # Specific (movielens, mimic-iii, imagenet)
            'metric': 2.0,    # Somewhat specific (ndcg, mrr, map)
            'method': 1.0,    # Common (attention, transformer, bert)
        }
        
        # Generic entities that appear in too many papers - penalize these
        GENERIC_ENTITIES = {
            'arxiv', 'github', 'adam', 'accuracy', 'loss',
            'attention', 'transformer', 'bert', 'gpt', 'llama'
        }
        
        # Count weighted shared entities
        related_scores = Counter()
        processed_pairs = set()
        
        for seed_paper in seed_papers:
            if seed_paper not in self.paper_connections:
                continue
            
            related_papers = self.paper_connections[seed_paper]
            
            for related_paper in related_papers:
                # Skip if already processed or in seed papers
                pair_key = tuple(sorted([seed_paper, related_paper]))
                if pair_key in processed_pairs or related_paper in seed_papers:
                    continue
                processed_pairs.add(pair_key)
                
                if seed_paper not in self.graph or related_paper not in self.graph:
                    continue
                
                # Get entities with their types
                seed_entities = {
                    (self.graph.nodes[e].get('node_type'), self.graph.nodes[e].get('name', e))
                    for e in self.graph.neighbors(seed_paper)
                }
                
                related_entities = {
                    (self.graph.nodes[e].get('node_type'), self.graph.nodes[e].get('name', e))
                    for e in self.graph.neighbors(related_paper)
                }
                
                # Find shared entities
                shared = seed_entities & related_entities
                if not shared:
                    continue
                
                # Calculate weighted score
                weighted_score = 0.0
                has_task_overlap = False
                
                for entity_type, entity_name in shared:
                    # Check if it's a generic entity
                    is_generic = entity_name.lower() in GENERIC_ENTITIES
                    
                    # Get weight for this entity type
                    weight = ENTITY_WEIGHTS.get(entity_type, 0.5)
                    
                    # Penalize generic entities heavily
                    if is_generic:
                        weight *= 0.3
                    
                    weighted_score += weight
                    
                    # Track if we have task overlap (important!)
                    if entity_type == 'task':
                        has_task_overlap = True
                
                # CRITICAL FILTER: Require task overlap OR high weighted score
                # This prevents false positives from papers connected only through generic entities
                if has_task_overlap or weighted_score >= 5.0:
                    related_scores[related_paper] += weighted_score
        
        # Normalize scores
        if related_scores:
            max_score = max(related_scores.values())
            if max_score > 0:
                for paper in related_scores:
                    related_scores[paper] = related_scores[paper] / max_score
        
        # Return top related papers
        return related_scores.most_common(max_expand)
    
    def fuse_scores(
        self, 
        vector_results: List[Dict], 
        graph_results: List[Tuple[str, float]],
        vector_weight: float = 0.7,
        graph_weight: float = 0.3
    ) -> List[Dict]:
        """
        Stage 3: Fuse vector and graph scores
        
        Combines results using weighted scoring
        """
        
        # Aggregate by paper_id
        paper_scores = defaultdict(lambda: {
            'vector_score': 0.0,
            'graph_score': 0.0,
            'chunks': [],
            'title': '',
            'paper_id': ''
        })
        
        # Add vector results
        for result in vector_results:
            paper_id = result['paper_id']
            paper_scores[paper_id]['paper_id'] = paper_id
            paper_scores[paper_id]['title'] = result['title']
            paper_scores[paper_id]['vector_score'] = max(
                paper_scores[paper_id]['vector_score'],
                result['vector_score']
            )
            paper_scores[paper_id]['chunks'].append({
                'text': result['text'],
                'section': result['section'],
                'score': result['vector_score']
            })
        
        # Create a set of vector paper IDs for quick lookup
        vector_paper_ids = set(paper_scores.keys())
        
        # Add graph results
        for paper_id, graph_score in graph_results:
            paper_scores[paper_id]['paper_id'] = paper_id
            paper_scores[paper_id]['graph_score'] = graph_score
            
            # If new paper from graph (not in vector results), get title
            if paper_id not in vector_paper_ids and paper_id in self.entities:
                paper_scores[paper_id]['title'] = self.entities[paper_id]['title']
        
        # Compute final scores with BOOSTING for papers that appear in both
        final_results = []
        for paper_id, data in paper_scores.items():
            # Check if paper appears in both vector and graph
            in_both = data['vector_score'] > 0 and data['graph_score'] > 0
            
            if in_both:
                # BOOST: Papers in both vector and graph get higher scores
                final_score = (
                    vector_weight * data['vector_score'] + 
                    graph_weight * data['graph_score'] +
                    0.1  # Bonus for appearing in both!
                )
            else:
                # Regular weighted score
                final_score = (
                    vector_weight * data['vector_score'] + 
                    graph_weight * data['graph_score']
                )
            
            final_results.append({
                'paper_id': paper_id,
                'title': data['title'],
                'final_score': final_score,
                'vector_score': data['vector_score'],
                'graph_score': data['graph_score'],
                'in_both': in_both,
                'chunks': sorted(data['chunks'], key=lambda x: x['score'], reverse=True)[:3],
                'num_chunks': len(data['chunks'])
            })
        
        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        vector_k: int = 20,
        graph_expand: int = 10,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        use_graph: bool = True
    ) -> List[Dict]:
        """Hybrid search pipeline"""
        
        # Stage 1: Vector search
        vector_results = self.vector_search(query, top_k=vector_k)
        print(f"\nDEBUG: Vector search returned {len(vector_results)} results")
        
        if not use_graph:
            return vector_results[:top_k]
        
        # Stage 2: Graph expansion
        seed_papers = list(set([r['paper_id'] for r in vector_results]))
        print(f"DEBUG: Seed papers: {seed_papers[:3]}")
        
        graph_results = self.graph_expansion(seed_papers, max_expand=graph_expand)
        print(f"DEBUG: Graph expansion returned {len(graph_results)} results")
        if graph_results:
            print(f"DEBUG: Top graph result: {graph_results[0]}")
        
        # Stage 3: Score fusion
        final_results = self.fuse_scores(
            vector_results, 
            graph_results,
            vector_weight=vector_weight,
            graph_weight=graph_weight
        )
        print(f"DEBUG: Final results: {len(final_results)}")
        if final_results:
            print(f"DEBUG: Top result - vector: {final_results[0]['vector_score']:.3f}, graph: {final_results[0]['graph_score']:.3f}")
        
        return final_results[:top_k]
        
    def compare_search_methods(self, query: str, top_k: int = 5):
        """
        Compare vector-only vs hybrid search
        
        Useful for evaluation and demonstration
        """
        
        print(f"Query: {query}\n")
        print("="*80)
        
        # Vector-only search
        print("VECTOR-ONLY SEARCH:")
        print("-"*80)
        vector_only = self.search(query, top_k=top_k, use_graph=False)
        for i, result in enumerate(vector_only, 1):
            print(f"{i}. [Score: {result['vector_score']:.3f}] {result['title'][:70]}...")
        
        print("\n" + "="*80)
        
        # Hybrid search
        print("HYBRID SEARCH (Vector + Graph):")
        print("-"*80)
        hybrid = self.search(query, top_k=top_k, use_graph=True)
        for i, result in enumerate(hybrid, 1):
            print(f"{i}. [Final: {result['final_score']:.3f}, Vector: {result['vector_score']:.3f}, Graph: {result['graph_score']:.3f}]")
            print(f"   {result['title'][:70]}...")
        
        print("\n" + "="*80)

def main():
    """Run hybrid/vector search for a user-provided query."""
    parser = argparse.ArgumentParser(description="Hybrid semantic search")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show side-by-side vector-only vs hybrid comparison"
    )
    parser.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    args = parser.parse_args()

    engine = HybridSearchEngine()
    query_text = " ".join(args.query)

    if args.compare:
        engine.compare_search_methods(query_text, top_k=args.k)
        return

    results = engine.search(query_text, top_k=args.k, use_graph=not args.no_graph)
    for i, result in enumerate(results, 1):
        if "final_score" in result:
            print(
                f"{i}. [Final: {result['final_score']:.3f}, "
                f"Vector: {result['vector_score']:.3f}, Graph: {result['graph_score']:.3f}] "
                f"{result['title']}"
            )
        else:
            print(f"{i}. [Vector: {result['vector_score']:.3f}] {result['title']}")

if __name__ == "__main__":
    main()
