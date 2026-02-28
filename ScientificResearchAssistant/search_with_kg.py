"""
IMPROVED Search - Filters Generic Entities
===========================================

NEW FIXES:
1. ✅ Only count related papers through SPECIFIC entities (datasets, methods)
2. ✅ Ignore generic metrics (accuracy, F1) for relatedness
3. ✅ Boost papers sharing important entities (ImageNet, CIFAR-10, BERT)
4. ✅ Better filtering of irrelevant results

This version prioritizes:
- Dataset entities (ImageNet, CIFAR-10, MNIST)
- Method entities (BERT, Transformer, ResNet)

Over generic ones:
- Metric entities (accuracy, F1, AUC) - TOO GENERIC
- Task entities (classification, detection) - TOO GENERIC
"""

import json
import numpy as np
import redis
import networkx as nx
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from collections import defaultdict
import os
from pathlib import Path

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
INDEX_NAME = "arxiv_chunks_idx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Find the graph file
def find_graph_file():
    """Find knowledge graph file in common locations"""
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent if current_dir.name == 'scripts' else current_dir
    
    possible_paths = [
        project_root / "data" / "knowledge_graph.graphml",
        project_root / "data" / "processed" / "knowledge_graph.graphml",
        Path("data/knowledge_graph.graphml"),
        Path("knowledge_graph.graphml"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

GRAPH_PATH = find_graph_file()


class ImprovedSearch:
    def __init__(self):
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.model = SentenceTransformer(MODEL_NAME)
        self.graph = None
        self.paper_metadata = {}
        
        # Define which entity types are "specific" vs "generic"
        self.specific_entity_types = {'datasets', 'methods'}  # Important!
        self.generic_entity_types = {'metrics', 'tasks'}      # Too generic!
        
    def load_graph(self):
        """Load knowledge graph"""
        if GRAPH_PATH is None:
            print("⚠️  No knowledge graph found")
            self.graph = None
            return
        
        try:
            print(f"Loading graph from: {GRAPH_PATH}")
            self.graph = nx.read_graphml(GRAPH_PATH)
            
            node_types = defaultdict(int)
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get('type', 'unknown')
                node_types[node_type] += 1
            
            print(f"✅ Loaded graph: {self.graph.number_of_nodes():,} nodes, {self.graph.number_of_edges():,} edges")
            print(f"   Papers: {node_types['paper']:,}, Entities: {node_types['entity']:,}, Topics: {node_types['topic']:,}")
            
        except Exception as e:
            print(f"⚠️  Error loading graph: {e}")
            self.graph = None
    
    def get_paper_from_redis(self, paper_id):
        """Get paper metadata from Redis"""
        if paper_id in self.paper_metadata:
            return self.paper_metadata[paper_id]
        
        try:
            q = Query(f"@paper_id:{{{paper_id}}}").return_fields("paper_id", "title", "url", "year")
            results = self.redis_client.ft(INDEX_NAME).search(q)
            
            if results.docs:
                doc = results.docs[0]
                metadata = {
                    'title': doc.title.decode('utf-8') if isinstance(doc.title, bytes) else doc.title,
                    'url': doc.url.decode('utf-8') if isinstance(doc.url, bytes) else doc.url,
                    'year': doc.year,
                }
                self.paper_metadata[paper_id] = metadata
                return metadata
        except:
            pass
        
        return None
    
    def vector_search(self, query_text, k=10):
        """Initial vector similarity search via Redis"""
        query_vec = self.model.encode([query_text], normalize_embeddings=True)[0]
        query_bytes = query_vec.astype(np.float32).tobytes()
        
        q = (
            Query(f"*=>[KNN {k} @embedding $vec AS score]")
            .return_fields("chunk_id", "paper_id", "title", "year", "section", "url", "score")
            .sort_by("score")
            .dialect(2)
        )
        
        results = self.redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_bytes})
        
        papers = []
        for doc in results.docs:
            papers.append({
                'paper_id': doc.paper_id.decode('utf-8') if isinstance(doc.paper_id, bytes) else doc.paper_id,
                'title': doc.title.decode('utf-8') if isinstance(doc.title, bytes) else doc.title,
                'year': doc.year,
                'url': doc.url.decode('utf-8') if isinstance(doc.url, bytes) else doc.url,
                'score': float(doc.score),
                'chunk_id': doc.chunk_id.decode('utf-8') if isinstance(doc.chunk_id, bytes) else doc.chunk_id,
            })
        
        return papers
    
    def get_paper_entities(self, paper_id, specific_only=False):
        """
        Extract entities for a paper from the graph
        
        Args:
            paper_id: Paper to get entities for
            specific_only: If True, only return datasets/methods (not metrics/tasks)
        """
        if not self.graph or paper_id not in self.graph:
            return []
        
        entities = []
        neighbors = list(self.graph.neighbors(paper_id))
        
        for neighbor in neighbors:
            node_data = self.graph.nodes.get(neighbor, {})
            node_type = node_data.get('type', 'unknown')
            
            if node_type == 'entity':
                entity_type = node_data.get('entity_type', '')
                entity_name = node_data.get('name', '')
                
                # Filter based on specific_only flag
                if specific_only:
                    if entity_type in self.specific_entity_types and entity_name:
                        entities.append(entity_name)
                else:
                    if entity_name:
                        entities.append(entity_name)
        
        return entities
    
    def expand_via_graph(self, initial_papers, max_expand=5):
        """
        IMPROVED: Expand through SPECIFIC entities only (datasets, methods)
        Ignore generic metrics/tasks that connect too many papers
        """
        if not self.graph:
            return []
        
        expanded = []
        paper_scores = defaultdict(float)
        
        for paper in initial_papers[:max_expand]:
            paper_id = paper['paper_id']
            
            if paper_id not in self.graph:
                continue
            
            neighbors = list(self.graph.neighbors(paper_id))
            
            for neighbor in neighbors:
                node_data = self.graph.nodes.get(neighbor, {})
                node_type = node_data.get('type', 'unknown')
                
                if node_type == 'paper':
                    # Direct paper similarity
                    edge_data = self.graph.get_edge_data(paper_id, neighbor)
                    weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                    paper_scores[neighbor] += weight * paper['score']
                    
                elif node_type == 'entity':
                    entity_type = node_data.get('entity_type', '')
                    entity_name = node_data.get('name', '')
                    
                    # FILTER: Only expand through SPECIFIC entities!
                    if entity_type in self.specific_entity_types:
                        # HIGH WEIGHT for dataset/method connections
                        entity_neighbors = list(self.graph.neighbors(neighbor))
                        for entity_paper in entity_neighbors:
                            if entity_paper != paper_id:
                                ep_data = self.graph.nodes.get(entity_paper, {})
                                if ep_data.get('type') == 'paper':
                                    paper_scores[entity_paper] += 0.6 * paper['score']
                    
                    # IGNORE generic entities (metrics, tasks)
                    # They connect too many unrelated papers!
                
                elif node_type == 'topic':
                    # LOW WEIGHT for topic clusters
                    topic_neighbors = list(self.graph.neighbors(neighbor))
                    for topic_paper in topic_neighbors:
                        if topic_paper != paper_id:
                            tp_data = self.graph.nodes.get(topic_paper, {})
                            if tp_data.get('type') == 'paper':
                                paper_scores[topic_paper] += 0.15 * paper['score']
        
        # Convert to result format
        for paper_id, score in sorted(paper_scores.items(), key=lambda x: -x[1])[:10]:
            if paper_id in self.graph:
                node_data = self.graph.nodes[paper_id]
                
                # Get URL
                url = node_data.get('url', '')
                if not url:
                    redis_data = self.get_paper_from_redis(paper_id)
                    if redis_data:
                        url = redis_data.get('url', '')
                
                expanded.append({
                    'paper_id': paper_id,
                    'title': node_data.get('title', ''),
                    'year': int(node_data.get('year', 0)) if node_data.get('year') else 0,
                    'score': score,
                    'source': 'graph_expansion',
                    'url': url,
                })
        
        return expanded
    
    def count_related_papers(self, paper_id):
        """
        Count related papers through SPECIFIC entities only
        FIXED: Ignore generic metrics/tasks that connect everything
        """
        if not self.graph or paper_id not in self.graph:
            return 0
        
        related = set()
        neighbors = list(self.graph.neighbors(paper_id))
        
        for neighbor in neighbors:
            node_data = self.graph.nodes.get(neighbor, {})
            node_type = node_data.get('type', 'unknown')
            
            if node_type == 'paper':
                # Direct paper connection
                related.add(neighbor)
                
            elif node_type == 'entity':
                entity_type = node_data.get('entity_type', '')
                
                # ONLY count connections through datasets/methods!
                if entity_type in self.specific_entity_types:
                    entity_neighbors = list(self.graph.neighbors(neighbor))
                    for ep in entity_neighbors:
                        ep_data = self.graph.nodes.get(ep, {})
                        if ep_data.get('type') == 'paper' and ep != paper_id:
                            related.add(ep)
        
        return len(related)
    
    def search(self, query_text, k=5, use_graph=True):
        """Combined search with improved filtering"""
        print(f"\n{'='*100}")
        print(f"Query: {query_text}")
        print(f"{'='*100}")
        
        # 1. Vector search
        print("🔍 Stage 1: Vector similarity search...")
        initial_results = self.vector_search(query_text, k=k*2)
        print(f"   Found {len(initial_results)} initial results")
        
        # 2. Graph expansion
        expanded_results = []
        if use_graph and self.graph:
            print("🕸️  Stage 2: Knowledge graph expansion...")
            expanded_results = self.expand_via_graph(initial_results, max_expand=5)
            print(f"   Found {len(expanded_results)} related papers")
        
        # 3. Display results
        print("📊 Top Results:")
        print("-" * 100)
        
        all_results = initial_results[:3] + expanded_results[:7]
        seen = set()
        display_count = 0
        
        for result in all_results:
            paper_id = result['paper_id']
            if paper_id in seen:
                continue
            seen.add(paper_id)
            display_count += 1
            
            source = result.get('source', 'vector_search')
            
            print(f"\n{display_count}. [Score: {result['score']:.4f}] {result['title']}")
            print(f"   Paper ID: {paper_id} | Year: {result['year']}")
            print(f"   Source: {source}")
            
            # URL
            url = result.get('url', '')
            if not url and self.graph and paper_id in self.graph:
                node_data = self.graph.nodes[paper_id]
                url = node_data.get('url', '')
            if not url:
                redis_data = self.get_paper_from_redis(paper_id)
                if redis_data:
                    url = redis_data.get('url', '')
            
            if url:
                print(f"   URL: {url}")
            
            # Show ALL entities
            if self.graph:
                all_entities = self.get_paper_entities(paper_id, specific_only=False)
                if all_entities:
                    entities_str = ", ".join(all_entities[:8])
                    print(f"   Entities: {entities_str}")
                
                # Show related count (through specific entities only!)
                related_count = self.count_related_papers(paper_id)
                if related_count > 0:
                    print(f"   Related papers: {related_count} (via datasets/methods)")
        
        print("\n" + "="*100)
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved semantic search with filtered entities")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results to show")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph expansion")
    
    args = parser.parse_args()
    
    # Initialize search
    searcher = ImprovedSearch()
    
    # Load knowledge graph
    if not args.no_graph:
        searcher.load_graph()
    
    # Perform search
    query_text = " ".join(args.query)
    results = searcher.search(query_text, k=args.k, use_graph=not args.no_graph)


if __name__ == "__main__":
    main()
