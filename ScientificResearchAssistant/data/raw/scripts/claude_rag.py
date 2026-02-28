"""
Quick test of Graph-Enriched RAG with Claude API

Much faster than local LLM!
"""

import sys
import os
import argparse
sys.path.append('.')
from complete_rag import MultiLLMRAG
from hybrid_search import HybridSearchEngine

def main():
    parser = argparse.ArgumentParser(
        description="Ask a research question with Claude + graph-enriched RAG"
    )
    parser.add_argument("query", nargs="+", help="Research question")
    parser.add_argument("--top-k", type=int, default=10, help="Number of papers to retrieve")
    args = parser.parse_args()

    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running claude_rag.py")
    
    print("="*80)
    print("GRAPH-ENRICHED RAG WITH CLAUDE API")
    print("="*80)
    
    # Initialize search engine
    print("\nInitializing hybrid search engine...")
    engine = HybridSearchEngine()
    
    # Initialize RAG with Claude
    print("\nInitializing Claude API...")
    rag = MultiLLMRAG(
        engine, 
        llm_provider="claude",
        api_key=claude_api_key
    )
    
    query_text = " ".join(args.query)
    result = rag.answer_query(query_text, top_k=args.top_k)

    print(f"\n📄 ANSWER:")
    print("="*80)
    print(result['answer'])
    print("="*80)
    print(f"\n⏱️  Generation time: {result['generation_time']:.2f} seconds")
    print(f"📊 Methods found: {len(result['entity_analysis']['methods'])}")
    print(f"📚 Papers retrieved: {len(result['papers'])}")

if __name__ == "__main__":
    main()
