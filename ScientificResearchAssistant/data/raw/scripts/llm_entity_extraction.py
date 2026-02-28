"""
Improved Entity Extraction using LLM (Claude API)

This replaces spaCy-based extraction with Claude for:
1. Context-aware entity extraction
2. Domain-specific terminology recognition
3. More accurate method/dataset/metric identification
4. Reduced false positives

Usage:
    python llm_entity_extraction.py
"""

import json
import anthropic
import os
from tqdm import tqdm
import time

class LLMEntityExtractor:
    """Extract entities from research papers using Claude API"""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def extract_entities(self, paper_id: str, title: str, text: str) -> dict:
        """
        Extract entities from a single paper using Claude
        
        Returns:
            {
                'methods': [...],
                'datasets': [...],
                'metrics': [...],
                'tasks': [...]
            }
        """
        
        # Build prompt
        prompt = f"""You are a research paper entity extractor. Extract technical entities from this paper.

PAPER TITLE: {title}

TEXT: {text}

Extract the following entities. Be specific and only include entities explicitly mentioned:

1. METHODS: Machine learning techniques, algorithms, architectures
   Examples: "transformer", "BERT", "federated learning", "GCN", "contrastive learning"
   
2. DATASETS: Specific named datasets (not generic terms)
   Examples: "ImageNet", "MIMIC-III", "MovieLens", "MS MARCO", "TREC"
   NOT: "dataset", "data", "corpus"
   
3. METRICS: Evaluation metrics
   Examples: "NDCG", "MRR", "precision", "recall", "F1", "accuracy"
   
4. TASKS: Research tasks or problems
   Examples: "recommendation", "retrieval", "classification", "generation", "question answering"

Return ONLY a JSON object with these exact keys: methods, datasets, metrics, tasks
Each value should be a list of lowercase strings.
If no entities found for a category, return empty list.

Example output:
{{
  "methods": ["transformer", "bert", "attention"],
  "datasets": ["imagenet", "coco"],
  "metrics": ["accuracy", "f1"],
  "tasks": ["classification", "detection"]
}}

JSON OUTPUT:"""

        # Call Claude with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse JSON from response
                response_text = response.content[0].text.strip()
                
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()
                
                entities = json.loads(response_text)
                
                # Validate structure
                required_keys = ['methods', 'datasets', 'metrics', 'tasks']
                if all(key in entities for key in required_keys):
                    return entities
                else:
                    raise ValueError(f"Missing required keys in response")
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error for {paper_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return {
                        'methods': [],
                        'datasets': [],
                        'metrics': [],
                        'tasks': []
                    }
            
            except Exception as e:
                if "overload" in str(e).lower() and attempt < max_retries - 1:
                    delay = 3 * (2 ** attempt)
                    print(f"  ⚠️  API overloaded, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  ❌ Error extracting entities for {paper_id}: {e}")
                    return {
                        'methods': [],
                        'datasets': [],
                        'metrics': [],
                        'tasks': []
                    }
    
    def process_papers(self, papers_file: str, output_file: str, limit: int = None):
        """
        Process all papers and extract entities
        
        Args:
            papers_file: Path to parsed_papers_full.jsonl
            output_file: Path to save entities (JSONL format)
            limit: Process only N papers (for testing)
        """
        
        print("="*80)
        print("LLM-BASED ENTITY EXTRACTION")
        print("="*80)
        print()
        
        # Load papers from JSONL
        print(f"Loading papers from {papers_file}...")
        papers = []
        
        with open(papers_file, 'r') as f:
            for line in f:
                paper = json.loads(line.strip())
                papers.append(paper)
                
                if limit and len(papers) >= limit:
                    break
        
        if limit:
            print(f"⚠️  Processing only {limit} papers (testing mode)")
        
        print(f"Loaded {len(papers)} papers")
        print()
        
        # Extract entities
        print("Extracting entities using Claude API...")
        print("This will take a while (~2-3 seconds per paper)")
        print()
        
        results = []
        failed = []
        skipped = []
        
        for paper in tqdm(papers, desc="Processing papers"):
            try:
                paper_id = paper['paper_id']
                title = paper.get('title', '')
                
                # Get text from sections
                sections = paper.get('sections', {})
                text = None
                
                if isinstance(sections, dict):
                    # Priority order: abstract > introduction > first available section
                    if 'abstract' in sections and sections['abstract']:
                        text = sections['abstract']
                    elif 'introduction' in sections and sections['introduction']:
                        text = sections['introduction']
                    else:
                        # Get first non-empty section
                        for key, value in sections.items():
                            if value and isinstance(value, str) and len(value.strip()) > 100:
                                text = value
                                break
                
                # Truncate to first 1500 characters for API efficiency
                if text:
                    text = text[:1500]
                
                # Skip if no text found
                if not text or len(text.strip()) < 100:
                    skipped.append(paper_id)
                    continue
                
                # Extract entities
                entities = self.extract_entities(paper_id, title, text)
                
                # Save result
                result = {
                    'paper_id': paper_id,
                    'title': title,
                    'year': paper.get('year', ''),
                    'categories': paper.get('categories', []),
                    'entities': entities
                }
                results.append(result)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"\n❌ Failed to process {paper.get('paper_id', 'unknown')}: {e}")
                failed.append(paper.get('paper_id', 'unknown'))
        
        # Save results in JSONL format
        print()
        print("="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"✅ Saved {len(results)} papers to {output_file}")
        
        if skipped:
            print(f"⚠️  Skipped {len(skipped)} papers (no text found)")
        
        if failed:
            print(f"❌ Failed to process {len(failed)} papers")
        
        # Statistics
        print()
        print("="*80)
        print("STATISTICS")
        print("="*80)
        
        if len(results) == 0:
            print("⚠️  No papers were processed!")
            print("\nPlease check your data file format.")
            return
        
        total_methods = sum(len(p['entities']['methods']) for p in results)
        total_datasets = sum(len(p['entities']['datasets']) for p in results)
        total_metrics = sum(len(p['entities']['metrics']) for p in results)
        total_tasks = sum(len(p['entities']['tasks']) for p in results)
        
        print(f"Successfully processed: {len(results)} papers")
        print(f"\nTotal entities extracted:")
        print(f"  Methods:  {total_methods} (avg: {total_methods/len(results):.1f} per paper)")
        print(f"  Datasets: {total_datasets} (avg: {total_datasets/len(results):.1f} per paper)")
        print(f"  Metrics:  {total_metrics} (avg: {total_metrics/len(results):.1f} per paper)")
        print(f"  Tasks:    {total_tasks} (avg: {total_tasks/len(results):.1f} per paper)")
        
        # Show sample entities from first paper
        if results:
            print(f"\n📋 Sample from first paper ({results[0]['paper_id']}):")
            sample = results[0]['entities']
            print(f"   Methods:  {sample['methods'][:5]}")
            print(f"   Datasets: {sample['datasets'][:5]}")
            print(f"   Metrics:  {sample['metrics'][:5]}")
            print(f"   Tasks:    {sample['tasks'][:5]}")
        
        print()
        print("✅ Entity extraction complete!")
        print()
        print("Next steps:")
        print(f"  1. Review {output_file}")
        print("  2. Rebuild knowledge graph with new entities")
        print("  3. Re-run evaluation with improved entities")


def main():
    """Run LLM entity extraction"""
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ Error: Set ANTHROPIC_API_KEY before running this script.")
        return
    
    # Initialize extractor
    extractor = LLMEntityExtractor(api_key=api_key)
    
    # Process papers - test with 10 first
    print("Starting with 10 papers as test...")
    print("If successful, you can process all papers.")
    print()
    
    extractor.process_papers(
        papers_file='data/processed/parsed_papers_full.jsonl',
        output_file='data/processed/entities_llm_test.jsonl',
        limit=10
    )
    
    print()
    print("="*80)
    print("TEST COMPLETE!")
    print("="*80)
    print()
    print("To process ALL papers, edit the script and change:")
    print("  limit=10  →  limit=None")


if __name__ == "__main__":
    main()
