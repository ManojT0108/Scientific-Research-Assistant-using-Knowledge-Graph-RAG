"""
Enhanced RAG System for Researchers

Improvements:
1. Shows paper IDs and ArXiv links in answers
2. Adds "Recommended Papers" section with rankings
3. Includes year, relevance scores, and direct links
4. Better structured output for researchers
"""

import json
import pickle
import argparse
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import sys
import os
sys.path.append('.')
from hybrid_search import HybridSearchEngine

class EnhancedRAG:
    """
    Research-friendly RAG with paper IDs, links, and rankings
    """

    def __init__(
        self,
        engine: HybridSearchEngine,
        llm_provider: str = "claude",
        api_key: Optional[str] = None
    ):
        self.engine = engine
        self.llm_provider = llm_provider

        # Load graph for entity analysis
        print("Loading knowledge graph for entity analysis...")
        with open('data/processed/knowledge_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)

        with open('data/processed/entities.json') as f:
            self.entities = json.load(f)

        # Initialize LLM
        print(f"\nInitializing {llm_provider.upper()} LLM...")
        self._initialize_llm(api_key)

        print("✅ Enhanced RAG ready!")

    def _initialize_llm(self, api_key: Optional[str] = None):
        """Initialize Claude API"""
        if self.llm_provider == "claude":
            if not api_key:
                raise ValueError("Claude API requires api_key parameter")

            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            self.claude_model = "claude-sonnet-4-20250514"
            print(f"  ✅ Claude API initialized (model: {self.claude_model})")
        else:
            raise ValueError(f"Only Claude is supported in this version")

    def _generate_answer(self, prompt: str, max_tokens: int = 600) -> str:
        """Generate answer using Claude API"""
        import time

        max_retries = 5
        base_delay = 3

        for attempt in range(max_retries):
            try:
                response = self.claude_client.messages.create(
                    model=self.claude_model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            except Exception as e:
                error_str = str(e).lower()

                if ("overload" in error_str or "529" in error_str) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"   ⚠️  API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"   ❌ Error: {e}")
                    raise

        raise Exception(f"Failed after {max_retries} retries")

    def analyze_entity_context(self, paper_ids: List[str]) -> Dict:
        """Extract entity relationships from retrieved papers"""

        all_datasets = Counter()
        all_methods = Counter()
        all_metrics = Counter()
        all_tasks = Counter()

        for paper_id in paper_ids:
            if paper_id not in self.entities:
                continue

            paper_entities = self.entities[paper_id]['entities']

            for dataset in paper_entities['datasets']:
                all_datasets[dataset] += 1
            for method in paper_entities['methods']:
                all_methods[method] += 1
            for metric in paper_entities['metrics']:
                all_metrics[metric] += 1
            for task in paper_entities['tasks']:
                all_tasks[task] += 1

        # Find consensus
        consensus_datasets = {k: v for k, v in all_datasets.items() if v >= 2}
        consensus_methods = {k: v for k, v in all_methods.items() if v >= 2}

        return {
            'datasets': dict(all_datasets.most_common(5)),
            'methods': dict(all_methods.most_common(10)),
            'metrics': dict(all_metrics.most_common(5)),
            'tasks': dict(all_tasks.most_common(5)),
            'consensus': {
                'datasets': consensus_datasets,
                'methods': consensus_methods
            }
        }

    def _get_arxiv_link(self, paper_id: str) -> str:
        """Convert paper ID to ArXiv link"""
        # Remove version suffix if present (e.g., v1, v2)
        base_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
        return f"https://arxiv.org/abs/{base_id}"

    def _get_paper_metadata(self, paper_id: str) -> Dict:
        """Get full metadata for a paper"""
        if paper_id in self.entities:
            return {
                'paper_id': paper_id,
                'title': self.entities[paper_id].get('title', 'Unknown'),
                'year': self.entities[paper_id].get('year', 2024),
                'arxiv_link': self._get_arxiv_link(paper_id)
            }
        return {
            'paper_id': paper_id,
            'title': 'Unknown',
            'year': 2024,
            'arxiv_link': self._get_arxiv_link(paper_id)
        }

    def build_enhanced_prompt(
        self,
        query: str,
        retrieved_papers: List[Dict],
        entity_analysis: Dict
    ) -> str:
        """Build research-focused prompt with paper IDs and metadata"""

        prompt = f"""You are a research assistant helping a researcher find relevant papers.

QUERY: {query}

## RETRIEVED PAPERS (with IDs and links)

"""

        # Add top 5 papers with full metadata
        for i, paper in enumerate(retrieved_papers[:5], 1):
            metadata = self._get_paper_metadata(paper['paper_id'])

            prompt += f"""### Paper {i}: {metadata['title']}
**Paper ID**: {metadata['paper_id']}
**Year**: {metadata['year']}
**ArXiv**: {metadata['arxiv_link']}
**Relevance Score**: {paper['final_score']:.3f}

"""

            if paper.get('chunks'):
                prompt += f"{paper['chunks'][0]['text'][:400]}...\n\n"

        # Add graph context
        prompt += "\n## KNOWLEDGE GRAPH INSIGHTS\n\n"

        if entity_analysis['methods']:
            prompt += "**Common Methods** (across retrieved papers):\n"
            for method, count in list(entity_analysis['methods'].items())[:5]:
                prompt += f"- {method} ({count} papers)\n"
            prompt += "\n"

        if entity_analysis['datasets']:
            prompt += "**Common Datasets**:\n"
            for dataset, count in list(entity_analysis['datasets'].items())[:3]:
                prompt += f"- {dataset} ({count} papers)\n"
            prompt += "\n"

        if entity_analysis['consensus']['methods']:
            prompt += f"**Consensus Methods** (used by 2+ papers): {', '.join(list(entity_analysis['consensus']['methods'].keys())[:5])}\n\n"

        prompt += """## INSTRUCTIONS

Generate a comprehensive answer (250-350 words) that:
1. Directly answers the query using the retrieved papers
2. **IMPORTANT**: When citing papers, use the format "Paper Title (paper_id)"
   Example: "According to PerLTQA (2406.00032v2), BERT achieved..."
3. Highlights consensus findings from multiple papers
4. Mentions specific methods, datasets, or techniques
5. Includes numbers and metrics when available

Your answer:"""

        return prompt

    def format_paper_recommendations(self, retrieved_papers: List[Dict]) -> str:
        """Create a ranked list of papers to read"""

        output = "\n\n" + "="*80 + "\n"
        output += "📚 RECOMMENDED PAPERS TO READ\n"
        output += "="*80 + "\n\n"

        for i, paper in enumerate(retrieved_papers[:10], 1):
            metadata = self._get_paper_metadata(paper['paper_id'])

            # Determine priority
            if i <= 3:
                priority = "🔥 HIGH PRIORITY"
            elif i <= 6:
                priority = "⭐ RECOMMENDED"
            else:
                priority = "📖 Additional Reading"

            output += f"{i}. {priority}\n"
            output += f"   Title: {metadata['title']}\n"
            output += f"   Paper ID: {metadata['paper_id']}\n"
            output += f"   Year: {metadata['year']}\n"
            output += f"   Link: {metadata['arxiv_link']}\n"
            output += f"   Relevance: {paper['final_score']:.3f}"

            # Show why it's relevant
            if paper.get('vector_score', 0) > 0 and paper.get('graph_score', 0) > 0:
                output += " (semantic + graph match)"
            elif paper.get('vector_score', 0) > 0:
                output += " (semantic match)"
            elif paper.get('graph_score', 0) > 0:
                output += " (related via knowledge graph)"

            output += "\n\n"

        return output

    def answer_query(self, query: str, top_k: int = 10) -> Dict:
        """
        Complete enhanced RAG pipeline
        """

        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        # Stage 1: Hybrid retrieval
        print("🔍 Retrieving papers...")
        results = self.engine.search(query, top_k=top_k, use_graph=True)
        paper_ids = [r['paper_id'] for r in results]
        print(f"   Retrieved {len(results)} papers")

        # Stage 2: Entity analysis
        print("📊 Analyzing entities...")
        entity_analysis = self.analyze_entity_context(paper_ids)
        print(f"   Found {len(entity_analysis['methods'])} unique methods")

        # Stage 3: Build enhanced prompt
        print("📝 Building prompt...")
        prompt = self.build_enhanced_prompt(query, results, entity_analysis)

        # Stage 4: Generate answer
        print(f"🤖 Generating answer with CLAUDE...")
        import time
        start_time = time.time()
        answer = self._generate_answer(prompt, max_tokens=600)
        generation_time = time.time() - start_time
        print(f"   Generated in {generation_time:.1f} seconds")

        # Stage 5: Format paper recommendations
        paper_recommendations = self.format_paper_recommendations(results)

        return {
            'query': query,
            'answer': answer,
            'paper_recommendations': paper_recommendations,
            'papers': results,
            'entity_analysis': entity_analysis,
            'generation_time': generation_time
        }

def main():
    """Run enhanced RAG for a user-provided query."""
    parser = argparse.ArgumentParser(
        description="Ask a research question using graph-enriched RAG (Claude)"
    )
    parser.add_argument("query", nargs="+", help="Research question")
    parser.add_argument("--top-k", type=int, default=10, help="Number of papers to retrieve")
    args = parser.parse_args()

    claude_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not claude_api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running enhanced_rag.py")

    print("="*80)
    print("ENHANCED RAG FOR RESEARCHERS")
    print("="*80)

    # Initialize
    print("\nInitializing hybrid search engine...")
    engine = HybridSearchEngine()

    print("\nInitializing Enhanced RAG...")
    rag = EnhancedRAG(
        engine,
        llm_provider="claude",
        api_key=claude_api_key
    )

    query_text = " ".join(args.query)
    result = rag.answer_query(query_text, top_k=args.top_k)

    print("\n📄 ANSWER:")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
    print(result["paper_recommendations"])
    print(f"⏱️  Generation time: {result['generation_time']:.2f} seconds")
    print(f"📊 Methods found: {len(result['entity_analysis']['methods'])}")
    print(f"📚 Papers retrieved: {len(result['papers'])}")

if __name__ == "__main__":
    main()
