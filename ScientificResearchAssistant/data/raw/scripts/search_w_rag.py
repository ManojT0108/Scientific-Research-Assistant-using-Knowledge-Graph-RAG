"""
Phase 5: Graph-Enriched RAG System

This RAG system demonstrates the value of knowledge graphs by:
1. Using hybrid search (vector + graph) to retrieve papers
2. Extracting entity relationships and connections
3. Providing LLM with BOTH content AND graph context
4. Generating answers that leverage entity relationships

Goal: Show that graph context improves answer quality beyond semantic similarity
"""

import json
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import sys
sys.path.append('.')
from hybrid_search import HybridSearchEngine

class GraphEnrichedRAG:
    """
    RAG system that leverages knowledge graph for richer context
    
    Differentiator: Not just "what papers say" but "how papers connect"
    """
    
    def __init__(self, engine: HybridSearchEngine):
        self.engine = engine
        
        # Load graph for entity analysis
        print("Loading knowledge graph for entity analysis...")
        with open('data/processed/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        
        with open('data/processed/entities.json') as f:
            self.entities = json.load(f)
        
        print("✅ Graph-enriched RAG ready!")
    
    def analyze_entity_context(self, paper_ids: List[str]) -> Dict:
        """
        Extract entity relationships from retrieved papers
        
        This is what makes graph-enriched RAG special!
        Returns: Rich entity context
        """
        
        # Aggregate entities across all papers
        all_datasets = Counter()
        all_methods = Counter()
        all_metrics = Counter()
        all_tasks = Counter()
        
        # Track which papers use which entities
        entity_usage = defaultdict(list)
        
        for paper_id in paper_ids:
            if paper_id not in self.entities:
                continue
            
            paper_entities = self.entities[paper_id]['entities']
            paper_title = self.entities[paper_id]['title']
            
            for dataset in paper_entities['datasets']:
                all_datasets[dataset] += 1
                entity_usage[f"dataset:{dataset}"].append(paper_title[:50])
            
            for method in paper_entities['methods']:
                all_methods[method] += 1
                entity_usage[f"method:{method}"].append(paper_title[:50])
            
            for metric in paper_entities['metrics']:
                all_metrics[metric] += 1
                entity_usage[f"metric:{metric}"].append(paper_title[:50])
            
            for task in paper_entities['tasks']:
                all_tasks[task] += 1
                entity_usage[f"task:{task}"].append(paper_title[:50])
        
        # Find consensus (entities used by multiple papers)
        consensus_datasets = {k: v for k, v in all_datasets.items() if v >= 2}
        consensus_methods = {k: v for k, v in all_methods.items() if v >= 2}
        consensus_metrics = {k: v for k, v in all_metrics.items() if v >= 2}
        consensus_tasks = {k: v for k, v in all_tasks.items() if v >= 2}
        
        return {
            'datasets': dict(all_datasets.most_common(5)),
            'methods': dict(all_methods.most_common(10)),
            'metrics': dict(all_metrics.most_common(5)),
            'tasks': dict(all_tasks.most_common(5)),
            'consensus': {
                'datasets': consensus_datasets,
                'methods': consensus_methods,
                'metrics': consensus_metrics,
                'tasks': consensus_tasks
            },
            'entity_usage': dict(entity_usage)
        }
    
    def find_paper_connections(self, paper_ids: List[str]) -> Dict:
        """
        Analyze how retrieved papers connect through shared entities
        
        This shows the graph's value in finding relationships
        """
        
        connections = defaultdict(int)
        shared_entities = defaultdict(set)
        
        # For each pair of papers, find shared entities
        for i, paper1 in enumerate(paper_ids):
            for paper2 in paper_ids[i+1:]:
                if paper1 in self.graph and paper2 in self.graph:
                    # Get entities for each paper
                    entities1 = set(self.graph.neighbors(paper1))
                    entities2 = set(self.graph.neighbors(paper2))
                    
                    # Find shared entities
                    shared = entities1 & entities2
                    
                    if shared:
                        pair_key = f"{paper1}||{paper2}"
                        connections[pair_key] = len(shared)
                        
                        for entity in shared:
                            entity_name = self.graph.nodes[entity].get('name', entity)
                            shared_entities[pair_key].add(entity_name)
        
        return {
            'num_connections': len(connections),
            'top_connections': sorted(
                connections.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            'shared_entities': {k: list(v) for k, v in shared_entities.items()}
        }
    
    def build_graph_context_prompt(
        self, 
        query: str,
        retrieved_papers: List[Dict],
        entity_analysis: Dict,
        connections: Dict
    ) -> str:
        """
        Build a prompt that includes BOTH content AND graph context
        
        This is the key differentiator!
        """
        
        prompt = f"""You are a research assistant analyzing scientific papers using both semantic content and knowledge graph relationships.

QUERY: {query}

## RETRIEVED PAPERS (Hybrid: Vector + Graph Search)

"""
        
        # Add retrieved papers with their source
        for i, paper in enumerate(retrieved_papers[:5], 1):
            source_type = "Both Vector+Graph" if paper.get('in_both') else (
                "Vector Search" if paper['vector_score'] > 0 else "Graph Expansion"
            )
            
            prompt += f"### Paper {i}: {paper['title']}\n"
            prompt += f"**Source**: {source_type} | "
            prompt += f"Vector: {paper['vector_score']:.3f}, Graph: {paper['graph_score']:.3f}\n\n"
            
            # Add chunks
            if paper.get('chunks'):
                prompt += f"**Content**:\n{paper['chunks'][0]['text'][:300]}...\n\n"
        
        # Add entity analysis - THE GRAPH CONTEXT!
        prompt += "\n## KNOWLEDGE GRAPH ANALYSIS\n\n"
        
        # Common methods
        if entity_analysis['methods']:
            prompt += "**Common Methods** (used across multiple papers):\n"
            for method, count in list(entity_analysis['methods'].items())[:5]:
                prompt += f"- {method} (used in {count} papers)\n"
            prompt += "\n"
        
        # Common datasets
        if entity_analysis['datasets']:
            prompt += "**Common Datasets**:\n"
            for dataset, count in list(entity_analysis['datasets'].items())[:5]:
                prompt += f"- {dataset} (used in {count} papers)\n"
            prompt += "\n"
        
        # Common tasks
        if entity_analysis['tasks']:
            prompt += "**Research Tasks Addressed**:\n"
            for task, count in list(entity_analysis['tasks'].items())[:5]:
                prompt += f"- {task} (addressed by {count} papers)\n"
            prompt += "\n"
        
        # Paper connections
        if connections['num_connections'] > 0:
            prompt += f"**Paper Connections**: {connections['num_connections']} connections found through shared entities\n\n"
        
        # Consensus entities (THE KEY INSIGHT!)
        consensus_methods = entity_analysis['consensus']['methods']
        if consensus_methods:
            prompt += "**🔑 CONSENSUS INSIGHTS** (entities appearing in 2+ papers):\n"
            prompt += f"- Methods with consensus: {', '.join(list(consensus_methods.keys())[:5])}\n"
        
        consensus_datasets = entity_analysis['consensus']['datasets']
        if consensus_datasets:
            prompt += f"- Commonly used datasets: {', '.join(list(consensus_datasets.keys())[:3])}\n"
        
        prompt += "\n"
        
        # Add instructions
        prompt += """## INSTRUCTIONS

Generate a comprehensive answer that:
1. **Directly answers the query** using information from retrieved papers
2. **Highlights consensus findings** - what do multiple papers agree on?
3. **Shows relationships** - how are these papers connected through shared methods/datasets/tasks?
4. **Provides context** - mention which methods, datasets, or techniques are commonly used
5. **Cite specific papers** by title when making claims

**IMPORTANT**: Your answer should demonstrate insights that come from seeing papers as a connected knowledge graph, not just independent documents. Highlight connections, consensus, and relationships.

Generate a well-structured answer (200-300 words):
"""
        
        return prompt
    
    def generate_answer(self, query: str, top_k: int = 10) -> Dict:
        """
        Generate graph-enriched answer for a query
        
        Returns: Answer with graph context
        """
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")
        
        # Stage 1: Hybrid retrieval
        print("🔍 Stage 1: Hybrid Search (Vector + Graph)...")
        results = self.engine.search(query, top_k=top_k, use_graph=True)
        paper_ids = [r['paper_id'] for r in results]
        
        print(f"   Retrieved {len(results)} papers")
        print(f"   - Papers from vector only: {sum(1 for r in results if r['vector_score'] > 0 and r['graph_score'] == 0)}")
        print(f"   - Papers from graph only: {sum(1 for r in results if r['vector_score'] == 0 and r['graph_score'] > 0)}")
        print(f"   - Papers from both: {sum(1 for r in results if r['vector_score'] > 0 and r['graph_score'] > 0)}")
        
        # Stage 2: Entity analysis
        print("\n📊 Stage 2: Analyzing Entity Relationships...")
        entity_analysis = self.analyze_entity_context(paper_ids)
        print(f"   Found {len(entity_analysis['methods'])} unique methods")
        print(f"   Found {len(entity_analysis['datasets'])} unique datasets")
        print(f"   Consensus methods: {len(entity_analysis['consensus']['methods'])}")
        
        # Stage 3: Find connections
        print("\n🔗 Stage 3: Finding Paper Connections...")
        connections = self.find_paper_connections(paper_ids)
        print(f"   Found {connections['num_connections']} paper connections")
        
        # Stage 4: Build context
        print("\n📝 Stage 4: Building Graph-Enriched Context...")
        prompt = self.build_graph_context_prompt(
            query, results, entity_analysis, connections
        )
        
        print(f"   Context size: {len(prompt)} characters")
        
        # Stage 5: Generate (placeholder - we'll add LLM next)
        print("\n🤖 Stage 5: Answer Generation...")
        print("   [LLM generation will be added next]")
        
        return {
            'query': query,
            'retrieved_papers': results,
            'entity_analysis': entity_analysis,
            'connections': connections,
            'prompt': prompt,
            'answer': "[LLM answer will be generated here]"
        }
    
    def demonstrate_graph_value(self, query: str):
        """
        Show side-by-side comparison: Vector-only vs Graph-enriched
        
        This demonstrates the value proposition!
        """
        
        print(f"\n{'='*80}")
        print(f"DEMONSTRATING KNOWLEDGE GRAPH VALUE")
        print(f"{'='*80}")
        print(f"\nQuery: {query}\n")
        
        # Vector-only retrieval
        print("="*80)
        print("APPROACH A: Vector-Only (Baseline)")
        print("="*80)
        vector_only = self.engine.search(query, top_k=10, use_graph=False)
        print(f"\nTop 5 Results (Vector-Only):")
        for i, paper in enumerate(vector_only[:5], 1):
            print(f"{i}. {paper['title'][:60]}...")
            print(f"   Score: {paper['vector_score']:.3f}\n")
        
        # Analyze entities (vector-only)
        vector_paper_ids = [p['paper_id'] for p in vector_only]
        vector_entities = self.analyze_entity_context(vector_paper_ids)
        print(f"Entity Diversity:")
        print(f"  Methods: {len(vector_entities['methods'])}")
        print(f"  Datasets: {len(vector_entities['datasets'])}")
        print(f"  Consensus methods: {len(vector_entities['consensus']['methods'])}")
        
        # Hybrid retrieval
        print("\n" + "="*80)
        print("APPROACH B: Hybrid (Vector + Graph)")
        print("="*80)
        hybrid = self.engine.search(query, top_k=10, use_graph=True)
        print(f"\nTop 5 Results (Hybrid):")
        for i, paper in enumerate(hybrid[:5], 1):
            source = "Vector+Graph" if paper.get('in_both') else (
                "Vector" if paper['vector_score'] > 0 else "Graph"
            )
            print(f"{i}. {paper['title'][:60]}...")
            print(f"   Source: {source} | V:{paper['vector_score']:.3f} G:{paper['graph_score']:.3f}\n")
        
        # Analyze entities (hybrid)
        hybrid_paper_ids = [p['paper_id'] for p in hybrid]
        hybrid_entities = self.analyze_entity_context(hybrid_paper_ids)
        print(f"Entity Diversity:")
        print(f"  Methods: {len(hybrid_entities['methods'])}")
        print(f"  Datasets: {len(hybrid_entities['datasets'])}")
        print(f"  Consensus methods: {len(hybrid_entities['consensus']['methods'])}")
        
        # Compare
        print("\n" + "="*80)
        print("📊 COMPARISON")
        print("="*80)
        
        print(f"\n**Method Diversity**:")
        print(f"  Vector-only: {len(vector_entities['methods'])} unique methods")
        print(f"  Hybrid: {len(hybrid_entities['methods'])} unique methods")
        improvement = ((len(hybrid_entities['methods']) - len(vector_entities['methods'])) 
                      / len(vector_entities['methods']) * 100 if len(vector_entities['methods']) > 0 else 0)
        print(f"  Improvement: {improvement:+.1f}%")
        
        print(f"\n**Consensus Insights**:")
        print(f"  Vector-only: {len(vector_entities['consensus']['methods'])} consensus methods")
        print(f"  Hybrid: {len(hybrid_entities['consensus']['methods'])} consensus methods")
        
        print(f"\n**Key Insight**: Graph expansion finds papers with diverse methods")
        print(f"that vector search might miss, enabling richer contextual answers!")

def main():
    """Demo graph-enriched RAG"""
    
    print("="*80)
    print("PHASE 5: GRAPH-ENRICHED RAG SYSTEM")
    print("="*80)
    print()
    
    # Initialize hybrid search engine
    print("Initializing hybrid search engine...")
    engine = HybridSearchEngine(redis_host='localhost', redis_port=6379)
    
    # Initialize RAG system
    rag = GraphEnrichedRAG(engine)
    
    # Test queries
    test_queries = [
        "What methods are effective for healthcare prediction?",
        "How do graph neural networks improve recommendation systems?",
    ]
    
    for query in test_queries:
        # Show graph value
        rag.demonstrate_graph_value(query)
        
        print("\n")
        
        # Generate answer
        result = rag.generate_answer(query, top_k=10)
        
        # Show the prompt (this is what would go to LLM)
        print("\n" + "="*80)
        print("GRAPH-ENRICHED PROMPT (What LLM sees):")
        print("="*80)
        print(result['prompt'][:2000] + "\n...[truncated]...")
        
        print("\n" + "="*80)
        print()
    
    print("✅ Phase 5 Demo Complete!")
    print("\nNext: Add LLM for actual answer generation")

if __name__ == "__main__":
    main()