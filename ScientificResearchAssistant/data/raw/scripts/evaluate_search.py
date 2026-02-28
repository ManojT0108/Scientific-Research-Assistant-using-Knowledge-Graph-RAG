"""
Script 8: Evaluate and Benchmark Search Quality

Measures:
- Precision@K
- Recall@K  
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Latency

Compares:
- Vector-only search
- Hybrid (vector + graph) search

Runtime: ~5 minutes for comprehensive evaluation
"""

import time
import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import sys

# Import hybrid search engine
sys.path.append('.')
from hybrid_search import HybridSearchEngine

class SearchEvaluator:
    """
    Evaluate search quality with various metrics
    """
    
    def __init__(self, engine: HybridSearchEngine):
        self.engine = engine
        
        # Load entities for creating test queries
        with open('data/processed/entities.json') as f:
            self.entities = json.load(f)
    
    def create_test_queries(self, num_queries: int = 50) -> List[Dict]:
        """
        Create test queries from paper entities
        
        Strategy: Use entity combinations as queries
        Papers that share these entities are considered relevant
        """
        
        print("Creating test queries...")
        
        test_queries = []
        
        # Get papers with good entity coverage
        papers_with_entities = [
            (pid, data) for pid, data in self.entities.items()
            if len(data['entities']['methods']) >= 2 and 
               len(data['entities']['tasks']) >= 1
        ]
        
        # Sample papers
        import random
        random.seed(42)
        sampled_papers = random.sample(
            papers_with_entities, 
            min(num_queries, len(papers_with_entities))
        )
        
        for paper_id, data in sampled_papers:
            # Create query from entities
            methods = data['entities']['methods'][:2]
            tasks = data['entities']['tasks'][:1]
            
            query_parts = methods + tasks
            query = ' '.join(query_parts)
            
            # Relevant papers: those sharing these entities
            relevant_papers = set([paper_id])  # The source paper is definitely relevant
            
            # Find other papers with same entities
            for other_id, other_data in self.entities.items():
                if other_id == paper_id:
                    continue
                
                # Check overlap
                other_methods = set(other_data['entities']['methods'])
                other_tasks = set(other_data['entities']['tasks'])
                
                query_methods = set(methods)
                query_tasks = set(tasks)
                
                # If significant overlap, consider relevant
                if (len(query_methods & other_methods) >= 1 and 
                    len(query_tasks & other_tasks) >= 1):
                    relevant_papers.add(other_id)
            
            # Only keep queries with multiple relevant papers
            if len(relevant_papers) >= 3:
                test_queries.append({
                    'query': query,
                    'source_paper': paper_id,
                    'relevant_papers': list(relevant_papers)
                })
        
        print(f"Created {len(test_queries)} test queries")
        return test_queries
    
    def precision_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Precision@K: fraction of retrieved that are relevant"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Recall@K: fraction of relevant that are retrieved"""
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & relevant)
        return relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
    
    def dcg_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Discounted Cumulative Gain@K"""
        dcg = 0.0
        for i, paper_id in enumerate(retrieved[:k], 1):
            if paper_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevant: set, k: int) -> float:
        """Normalized DCG@K"""
        dcg = self.dcg_at_k(retrieved, relevant, k)
        
        # Ideal DCG: all relevant papers ranked first
        ideal_retrieved = list(relevant) + [p for p in retrieved if p not in relevant]
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mrr(self, retrieved: List[str], relevant: set) -> float:
        """Mean Reciprocal Rank: rank of first relevant result"""
        for i, paper_id in enumerate(retrieved, 1):
            if paper_id in relevant:
                return 1.0 / i
        return 0.0
    
    def evaluate_query(
        self, 
        query: str, 
        relevant_papers: List[str],
        use_graph: bool = True,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict:
        """
        Evaluate a single query
        
        Returns metrics for this query
        """
        
        relevant_set = set(relevant_papers)
        
        # Perform search
        start_time = time.time()
        results = self.engine.search(
            query, 
            top_k=max(k_values), 
            use_graph=use_graph
        )
        latency = time.time() - start_time
        
        # Extract paper IDs
        retrieved = [r['paper_id'] for r in results]
        
        # Compute metrics for each k
        metrics = {'latency': latency}
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(retrieved, relevant_set, k)
            metrics[f'recall@{k}'] = self.recall_at_k(retrieved, relevant_set, k)
            metrics[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevant_set, k)
        
        metrics['mrr'] = self.mrr(retrieved, relevant_set)
        
        return metrics
    
    def evaluate_all(
        self, 
        test_queries: List[Dict],
        k_values: List[int] = [5, 10, 20]
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate all queries for both vector-only and hybrid
        
        Returns: (vector_metrics, hybrid_metrics)
        """
        
        print(f"\nEvaluating {len(test_queries)} queries...")
        print("This will take a few minutes...\n")
        
        vector_metrics = defaultdict(list)
        hybrid_metrics = defaultdict(list)
        
        for i, test_query in enumerate(test_queries, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_queries)}")
            
            query = test_query['query']
            relevant = test_query['relevant_papers']
            
            # Evaluate vector-only
            vector_result = self.evaluate_query(
                query, relevant, use_graph=False, k_values=k_values
            )
            for metric, value in vector_result.items():
                vector_metrics[metric].append(value)
            
            # Evaluate hybrid
            hybrid_result = self.evaluate_query(
                query, relevant, use_graph=True, k_values=k_values
            )
            for metric, value in hybrid_result.items():
                hybrid_metrics[metric].append(value)
        
        # Average metrics
        vector_avg = {k: np.mean(v) for k, v in vector_metrics.items()}
        hybrid_avg = {k: np.mean(v) for k, v in hybrid_metrics.items()}
        
        return vector_avg, hybrid_avg
    
    def print_comparison(self, vector_metrics: Dict, hybrid_metrics: Dict):
        """Print comparison table"""
        
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n{'Metric':<20} {'Vector-Only':<15} {'Hybrid':<15} {'Improvement':<15}")
        print("-"*80)
        
        for metric in sorted(vector_metrics.keys()):
            if metric == 'latency':
                continue
            
            vector_val = vector_metrics[metric]
            hybrid_val = hybrid_metrics[metric]
            improvement = ((hybrid_val - vector_val) / vector_val * 100) if vector_val > 0 else 0
            
            print(f"{metric:<20} {vector_val:<15.4f} {hybrid_val:<15.4f} {improvement:>+14.1f}%")
        
        print("-"*80)
        print(f"{'Latency (ms)':<20} {vector_metrics['latency']*1000:<15.1f} {hybrid_metrics['latency']*1000:<15.1f}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Calculate average improvement
        improvements = []
        for metric in vector_metrics.keys():
            if metric != 'latency':
                vector_val = vector_metrics[metric]
                hybrid_val = hybrid_metrics[metric]
                if vector_val > 0:
                    improvement = ((hybrid_val - vector_val) / vector_val * 100)
                    improvements.append(improvement)
        
        avg_improvement = np.mean(improvements) if improvements else 0
        
        print(f"\nAverage improvement: {avg_improvement:+.1f}%")
        
        if avg_improvement > 5:
            print("✅ Hybrid search significantly outperforms vector-only search!")
        elif avg_improvement > 0:
            print("✅ Hybrid search shows modest improvements")
        else:
            print("⚠️  Hybrid search does not improve over vector-only")
        
        print(f"\nLatency increase: {(hybrid_metrics['latency'] - vector_metrics['latency'])*1000:.1f}ms")
        if hybrid_metrics['latency'] < 1.0:
            print("✅ Hybrid search maintains sub-second latency")
    
    def save_results(
        self, 
        vector_metrics: Dict, 
        hybrid_metrics: Dict,
        output_file: str = 'data/processed/evaluation_results.json'
    ):
        """Save evaluation results"""
        
        results = {
            'vector_only': vector_metrics,
            'hybrid': hybrid_metrics,
            'improvement': {
                metric: ((hybrid_metrics[metric] - vector_metrics[metric]) / vector_metrics[metric] * 100)
                if vector_metrics[metric] > 0 else 0
                for metric in vector_metrics.keys()
                if metric != 'latency'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")

def main():
    """Run comprehensive evaluation"""
    
    print("="*80)
    print("PHASE 4: SEARCH EVALUATION & BENCHMARKING")
    print("="*80)
    print()
    
    # Initialize engine
    print("Initializing search engine...")
    engine = HybridSearchEngine(redis_host='localhost', redis_port=6379)
    
    # Create evaluator
    evaluator = SearchEvaluator(engine)
    
    # Create test queries
    test_queries = evaluator.create_test_queries(num_queries=50)
    
    # Evaluate
    vector_metrics, hybrid_metrics = evaluator.evaluate_all(
        test_queries, 
        k_values=[5, 10, 20]
    )
    
    # Print results
    evaluator.print_comparison(vector_metrics, hybrid_metrics)
    
    # Save results
    evaluator.save_results(vector_metrics, hybrid_metrics)
    
    print("\n✅ Phase 4 Complete!")
    print("\nNext: Phase 5 - RAG Integration")

if __name__ == "__main__":
    main()