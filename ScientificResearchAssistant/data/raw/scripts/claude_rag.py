"""
Quick test of Graph-Enriched RAG with Claude API

Much faster than local LLM!
"""

import sys
import os
sys.path.append('.')
from complete_rag import MultiLLMRAG
from hybrid_search import HybridSearchEngine

def main():
    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running claude_rag.py")
    
    print("="*80)
    print("GRAPH-ENRICHED RAG WITH CLAUDE API")
    print("="*80)
    
    # Initialize search engine
    print("\nInitializing hybrid search engine...")
    engine = HybridSearchEngine(redis_host='localhost', redis_port=6379)
    
    # Initialize RAG with Claude
    print("\nInitializing Claude API...")
    rag = MultiLLMRAG(
        engine, 
        llm_provider="claude",
        api_key=claude_api_key
    )
    
    # REAL researcher questions - specific and actionable
    test_queries = [
        "Which papers compare BERT and GPT-4 for text classification?",
        "What are the main challenges in multi-hop question answering?",
        "Which datasets should I use to evaluate my question answering system?",
        "How do I implement contrastive learning for recommendations?",
        "What's the state of the art accuracy for sentiment analysis?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        # Generate answer
        result = rag.answer_query(query, top_k=10)

        # Display result
        print(f"\n📄 ANSWER:")
        print("="*80)
        print(result['answer'])
        print("="*80)

        print(f"\n⏱️  Generation time: {result['generation_time']:.2f} seconds")
        print(f"📊 Methods found: {len(result['entity_analysis']['methods'])}")
        print(f"📚 Papers retrieved: {len(result['papers'])}")

        # Show entity context
        print(f"\n🔬 Entity Context (What made the answer rich):")
        print(f"   Common methods: {', '.join(list(result['entity_analysis']['methods'].keys())[:5])}")
        print(f"   Common datasets: {', '.join(list(result['entity_analysis']['datasets'].keys())[:3])}")
        print(f"   Consensus methods: {len(result['entity_analysis']['consensus']['methods'])}")
        print()

    print("\n✅ All tests complete!")

if __name__ == "__main__":
    main()
