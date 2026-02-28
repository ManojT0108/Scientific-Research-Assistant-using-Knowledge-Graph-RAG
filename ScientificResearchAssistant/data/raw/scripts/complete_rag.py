"""
Phase 5: Complete Graph-Enriched RAG with Multiple LLM Options

Supports:
1. Local LLM (Llama 3.2-3B on GPU)
2. Claude API (Anthropic)
3. Groq API (Fast, free)

Compare quality across all three!
"""

import json
import pickle
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import sys
import os
sys.path.append('.')
from hybrid_search import HybridSearchEngine

class MultiLLMRAG:
    """
    Graph-Enriched RAG with multiple LLM backends
    
    Choose your LLM: local GPU, Claude API, or Groq
    """
    
    def __init__(
        self, 
        engine: HybridSearchEngine,
        llm_provider: str = "local",  # "local", "claude", or "groq"
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
        
        print("✅ Graph-enriched RAG ready!")
    
    def _initialize_llm(self, api_key: Optional[str] = None):
        """Initialize the chosen LLM backend"""
        
        if self.llm_provider == "local":
            self._init_local_llm()
        
        elif self.llm_provider == "claude":
            if not api_key:
                raise ValueError("Claude API requires api_key parameter")
            self._init_claude(api_key)
        
        elif self.llm_provider == "groq":
            if not api_key:
                raise ValueError("Groq API requires api_key parameter")
            self._init_groq(api_key)
        
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def _init_local_llm(self):
        """Initialize local Llama model on GPU"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("  Loading Llama 3.2-3B-Instruct...")
        print("  This will download ~6GB on first run (cached after)")
        
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"  # Automatically use GPU if available
        )
        
        # Check device
        device = next(self.model.parameters()).device
        print(f"  ✅ Model loaded on: {device}")
        
        if torch.cuda.is_available():
            print(f"  ✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠️  Using CPU (will be slow)")
    
    def _init_claude(self, api_key: str):
        """Initialize Claude API"""
        import anthropic
        
        self.claude_client = anthropic.Anthropic(api_key=api_key)
        self.claude_model = "claude-sonnet-4-20250514"
        print(f"  ✅ Claude API initialized (model: {self.claude_model})")
    
    def _init_groq(self, api_key: str):
        """Initialize Groq API"""
        from groq import Groq
        
        self.groq_client = Groq(api_key=api_key)
        self.groq_model = "llama-3.1-70b-versatile"
        print(f"  ✅ Groq API initialized (model: {self.groq_model})")
    
    def generate_answer(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate answer using configured LLM
        
        Routes to appropriate backend
        """
        
        if self.llm_provider == "local":
            return self._generate_local(prompt, max_tokens)
        
        elif self.llm_provider == "claude":
            return self._generate_claude(prompt, max_tokens)
        
        elif self.llm_provider == "groq":
            return self._generate_groq(prompt, max_tokens)
    
    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Generate with local Llama model"""
        import torch
        
        # Format for Llama 3.2 Instruct
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _generate_claude(self, prompt: str, max_tokens: int) -> str:
        """Generate with Claude API - with robust retry for overload errors"""
        
        import time
        
        max_retries = 5  # More retries
        base_delay = 3   # Longer initial delay
        
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
                
                # Check if it's an overload error
                if ("overload" in error_str or "529" in error_str) and attempt < max_retries - 1:
                    # Exponential backoff: 3s, 6s, 12s, 24s, 48s
                    delay = base_delay * (2 ** attempt)
                    print(f"   ⚠️  API overloaded (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    # Not overload error, or out of retries
                    print(f"   ❌ Error: {e}")
                    raise
        
        raise Exception(f"Failed after {max_retries} retries - API still overloaded")
    
    def _generate_groq(self, prompt: str, max_tokens: int) -> str:
        """Generate with Groq API"""
        
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
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
    
    def build_prompt(
        self,
        query: str,
        retrieved_papers: List[Dict],
        entity_analysis: Dict
    ) -> str:
        """Build graph-enriched prompt"""
        
        prompt = f"""You are a research assistant analyzing scientific papers using both semantic content and knowledge graph relationships.

QUERY: {query}

## RETRIEVED PAPERS

"""
        
        # Add top 5 papers
        for i, paper in enumerate(retrieved_papers[:5], 1):
            prompt += f"### Paper {i}: {paper['title']}\n\n"
            
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

Generate a comprehensive answer (200-300 words) that:
1. Directly answers the query using the retrieved papers
2. Highlights consensus findings from multiple papers
3. Mentions specific methods, datasets, or techniques commonly used
4. Cites paper titles when making claims

Your answer:"""
        
        return prompt
    
    def answer_query(self, query: str, top_k: int = 10) -> Dict:
        """
        Complete RAG pipeline with answer generation
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
        
        # Stage 3: Build prompt
        print("📝 Building prompt...")
        prompt = self.build_prompt(query, results, entity_analysis)
        
        # Stage 4: Generate answer
        print(f"🤖 Generating answer with {self.llm_provider.upper()}...")
        import time
        start_time = time.time()
        answer = self.generate_answer(prompt, max_tokens=500)
        generation_time = time.time() - start_time
        print(f"   Generated in {generation_time:.1f} seconds")
        
        return {
            'query': query,
            'answer': answer,
            'papers': results[:5],
            'entity_analysis': entity_analysis,
            'generation_time': generation_time,
            'llm_provider': self.llm_provider
        }

def compare_llms(query: str, engine: HybridSearchEngine):
    """
    Compare all three LLM options side-by-side
    
    Requires API keys for Claude and Groq
    """
    
    print("="*80)
    print("COMPARING LLM OPTIONS")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    # Test local LLM
    print("\n" + "="*80)
    print("OPTION 1: LOCAL LLM (Llama 3.2-3B on GPU)")
    print("="*80)
    try:
        rag_local = MultiLLMRAG(engine, llm_provider="local")
        result_local = rag_local.answer_query(query, top_k=10)
        
        print(f"\n📄 ANSWER (Local LLM):")
        print("-"*80)
        print(result_local['answer'])
        print(f"\n⏱️  Generation time: {result_local['generation_time']:.1f}s")
    except Exception as e:
        print(f"❌ Local LLM failed: {e}")
        result_local = None
    
    # Test Claude API (if key provided)
    print("\n" + "="*80)
    print("OPTION 2: CLAUDE API")
    print("="*80)
    
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        try:
            rag_claude = MultiLLMRAG(engine, llm_provider="claude", api_key=claude_key)
            result_claude = rag_claude.answer_query(query, top_k=10)
            
            print(f"\n📄 ANSWER (Claude):")
            print("-"*80)
            print(result_claude['answer'])
            print(f"\n⏱️  Generation time: {result_claude['generation_time']:.1f}s")
        except Exception as e:
            print(f"❌ Claude API failed: {e}")
            result_claude = None
    else:
        print("⚠️  Set ANTHROPIC_API_KEY environment variable to test Claude")
        result_claude = None
    
    # Test Groq API (if key provided)
    print("\n" + "="*80)
    print("OPTION 3: GROQ API")
    print("="*80)
    
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            rag_groq = MultiLLMRAG(engine, llm_provider="groq", api_key=groq_key)
            result_groq = rag_groq.answer_query(query, top_k=10)
            
            print(f"\n📄 ANSWER (Groq):")
            print("-"*80)
            print(result_groq['answer'])
            print(f"\n⏱️  Generation time: {result_groq['generation_time']:.1f}s")
        except Exception as e:
            print(f"❌ Groq API failed: {e}")
            result_groq = None
    else:
        print("⚠️  Set GROQ_API_KEY environment variable to test Groq")
        result_groq = None
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if result_local:
        print(f"✅ Local LLM: {result_local['generation_time']:.1f}s (Free, runs on GPU)")
    if result_claude:
        print(f"✅ Claude API: {result_claude['generation_time']:.1f}s (Best quality, ~$0.01/query)")
    if result_groq:
        print(f"✅ Groq API: {result_groq['generation_time']:.1f}s (Fast, free)")

def main():
    """Demo with local LLM first"""
    
    print("="*80)
    print("PHASE 5: GRAPH-ENRICHED RAG WITH LLM")
    print("="*80)
    print()
    
    # Initialize search engine
    print("Initializing hybrid search engine...")
    engine = HybridSearchEngine(redis_host='localhost', redis_port=6379)
    
    # Test query
    query = "What methods are effective for healthcare prediction?"
    
    # Option 1: Just test local LLM
    print("\n" + "="*80)
    print("TESTING: LOCAL LLM (Llama 3.2-3B)")
    print("="*80)
    
    rag = MultiLLMRAG(engine, llm_provider="local")
    result = rag.answer_query(query, top_k=10)
    
    print(f"\n📄 ANSWER:")
    print("="*80)
    print(result['answer'])
    print("="*80)
    
    print(f"\n⏱️  Generation time: {result['generation_time']:.1f} seconds")
    print(f"📊 Methods found: {len(result['entity_analysis']['methods'])}")
    print(f"📚 Papers retrieved: {len(result['papers'])}")
    
    print("\n✅ Phase 5 Complete!")
    print("\nTo compare all LLMs, set API keys and run compare_llms()")
    print("  export ANTHROPIC_API_KEY='your-key'")
    print("  export GROQ_API_KEY='your-key'")

if __name__ == "__main__":
    main()