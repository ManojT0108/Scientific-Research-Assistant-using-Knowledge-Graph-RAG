"""
Script 5: Extract entities from papers for knowledge graph

Extracts:
- Datasets (MNIST, ImageNet, etc.)
- Methods (BERT, Transformer, etc.)
- Metrics (Accuracy, F1-score, etc.)
- Tasks (Classification, etc.)

Runtime: ~5-10 minutes for 12,187 chunks
"""

import json
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import redis

class EntityExtractor:
    """
    Extract scientific entities from research papers
    
    Uses dictionary-based matching for speed and reliability
    """
    
    def __init__(self):
        # Common datasets in CS/ML/DB
        self.datasets = {
            # Vision
            'mnist', 'cifar-10', 'cifar-100', 'imagenet', 'coco', 'voc',
            'places365', 'kinetics', 'ucf101',
            
            # NLP
            'glue', 'superglue', 'squad', 'ms marco', 'wikitext', 
            'bookcorpus', 'wikipedia', 'common crawl', 'webtext',
            
            # Medical
            'mimic-iii', 'mimic-iv', 'eicu', 'physionet', 'chexpert',
            'isic', 'brats', 'tcga',
            
            # Graph/DB
            'ogbn-arxiv', 'cora', 'citeseer', 'pubmed', 'nell',
            'freebase', 'wikidata', 'yago',
            
            # IR/RecSys
            'movielens', 'netflix', 'amazon reviews', 'yelp',
            'trec', 'ms-coco', 'beir',
            
            # General
            'arxiv', 'github', 'stackexchange'
        }
        
        # Common methods/models
        self.methods = {
            # Transformers
            'bert', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 't5', 'roberta',
            'electra', 'albert', 'xlnet', 'longformer', 'bigbird',
            'llama', 'llama-2', 'mistral', 'claude', 'gemini',
            
            # Vision models
            'resnet', 'vgg', 'alexnet', 'inception', 'efficientnet',
            'vision transformer', 'vit', 'swin transformer', 'deit',
            'clip', 'dall-e', 'stable diffusion',
            
            # Graph models
            'gcn', 'gat', 'graphsage', 'gin', 'gnn', 'graph neural network',
            
            # Traditional ML
            'svm', 'random forest', 'xgboost', 'lightgbm', 'catboost',
            'naive bayes', 'logistic regression', 'decision tree',
            
            # Deep learning
            'cnn', 'rnn', 'lstm', 'gru', 'seq2seq', 'attention',
            'transformer', 'autoencoder', 'vae', 'gan', 'diffusion',
            
            # Optimization
            'adam', 'sgd', 'rmsprop', 'adagrad', 'momentum',
            
            # Database/IR
            'bm25', 'tf-idf', 'pagerank', 'lucene', 'elasticsearch',
            'faiss', 'annoy', 'hnsw', 'vector search',
            
            # Techniques
            'fine-tuning', 'transfer learning', 'few-shot learning',
            'zero-shot learning', 'meta-learning', 'reinforcement learning',
            'supervised learning', 'unsupervised learning', 'self-supervised learning',
            'contrastive learning', 'knowledge distillation', 'pruning',
            'quantization', 'retrieval augmented generation', 'rag'
        }
        
        # Common metrics
        self.metrics = {
            'accuracy', 'precision', 'recall', 'f1', 'f1-score',
            'auc', 'roc', 'map', 'ndcg', 'mrr', 'hit rate',
            'bleu', 'rouge', 'meteor', 'bertscore',
            'perplexity', 'loss', 'mse', 'mae', 'rmse',
            'iou', 'dice', 'jaccard',
            'throughput', 'latency', 'qps'
        }
        
        # Common tasks
        self.tasks = {
            'classification', 'regression', 'detection', 'segmentation',
            'generation', 'translation', 'summarization', 'question answering',
            'named entity recognition', 'ner', 'pos tagging',
            'sentiment analysis', 'text classification',
            'image classification', 'object detection',
            'recommendation', 'ranking', 'retrieval',
            'clustering', 'anomaly detection'
        }
    
    def extract_from_text(self, text):
        """
        Extract entities from text using pattern matching
        
        Returns: dict with entity types and found entities
        """
        text_lower = text.lower()
        
        entities = {
            'datasets': set(),
            'methods': set(),
            'metrics': set(),
            'tasks': set()
        }
        
        # Extract datasets
        for dataset in self.datasets:
            if dataset in text_lower:
                # Verify it's a word boundary match
                pattern = r'\b' + re.escape(dataset) + r'\b'
                if re.search(pattern, text_lower):
                    entities['datasets'].add(dataset)
        
        # Extract methods
        for method in self.methods:
            if method in text_lower:
                pattern = r'\b' + re.escape(method) + r'\b'
                if re.search(pattern, text_lower):
                    entities['methods'].add(method)
        
        # Extract metrics
        for metric in self.metrics:
            if metric in text_lower:
                pattern = r'\b' + re.escape(metric) + r'\b'
                if re.search(pattern, text_lower):
                    entities['metrics'].add(metric)
        
        # Extract tasks
        for task in self.tasks:
            if task in text_lower:
                pattern = r'\b' + re.escape(task) + r'\b'
                if re.search(pattern, text_lower):
                    entities['tasks'].add(task)
        
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in entities.items()}
    
    def extract_from_chunks(self, chunks_file='data/processed/chunks_full.jsonl'):
        """
        Extract entities from all chunks
        
        Returns: paper-level entity aggregation
        """
        print("Extracting entities from chunks...")
        
        # Aggregate entities per paper
        paper_entities = defaultdict(lambda: {
            'datasets': set(),
            'methods': set(),
            'metrics': set(),
            'tasks': set(),
            'title': '',
            'year': 2024,
            'categories': []
        })
        
        # Process each chunk
        with open(chunks_file) as f:
            for line in tqdm(f, desc="Processing chunks"):
                chunk = json.loads(line)
                paper_id = chunk['paper_id']
                text = chunk.get('text', '')
                
                # Extract entities from this chunk
                entities = self.extract_from_text(text)
                
                # Aggregate to paper level
                for entity_type, entity_list in entities.items():
                    paper_entities[paper_id][entity_type].update(entity_list)
                
                # Store metadata
                if not paper_entities[paper_id]['title']:
                    paper_entities[paper_id]['title'] = chunk.get('title', '')
                    paper_entities[paper_id]['year'] = chunk.get('year', 2024)
                    paper_entities[paper_id]['categories'] = chunk.get('categories', [])
        
        # Convert sets to lists and create final output
        result = {}
        for paper_id, data in paper_entities.items():
            result[paper_id] = {
                'title': data['title'],
                'year': data['year'],
                'categories': data['categories'],
                'entities': {
                    'datasets': sorted(list(data['datasets'])),
                    'methods': sorted(list(data['methods'])),
                    'metrics': sorted(list(data['metrics'])),
                    'tasks': sorted(list(data['tasks']))
                }
            }
        
        return result
    
    def save_entities(self, paper_entities, output_file='data/processed/entities.json'):
        """Save extracted entities to file"""
        with open(output_file, 'w') as f:
            json.dump(paper_entities, f, indent=2)
        
        print(f"\nSaved entities to {output_file}")
    
    def print_statistics(self, paper_entities):
        """Print entity extraction statistics"""
        
        # Count entities
        all_datasets = set()
        all_methods = set()
        all_metrics = set()
        all_tasks = set()
        
        papers_with_datasets = 0
        papers_with_methods = 0
        papers_with_metrics = 0
        papers_with_tasks = 0
        
        for paper_id, data in paper_entities.items():
            entities = data['entities']
            
            if entities['datasets']:
                papers_with_datasets += 1
                all_datasets.update(entities['datasets'])
            
            if entities['methods']:
                papers_with_methods += 1
                all_methods.update(entities['methods'])
            
            if entities['metrics']:
                papers_with_metrics += 1
                all_metrics.update(entities['metrics'])
            
            if entities['tasks']:
                papers_with_tasks += 1
                all_tasks.update(entities['tasks'])
        
        print("\n" + "="*60)
        print("ENTITY EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total papers: {len(paper_entities)}")
        print(f"\nPapers with datasets: {papers_with_datasets} ({papers_with_datasets/len(paper_entities)*100:.1f}%)")
        print(f"Unique datasets found: {len(all_datasets)}")
        print(f"\nPapers with methods: {papers_with_methods} ({papers_with_methods/len(paper_entities)*100:.1f}%)")
        print(f"Unique methods found: {len(all_methods)}")
        print(f"\nPapers with metrics: {papers_with_metrics} ({papers_with_metrics/len(paper_entities)*100:.1f}%)")
        print(f"Unique metrics found: {len(all_metrics)}")
        print(f"\nPapers with tasks: {papers_with_tasks} ({papers_with_tasks/len(paper_entities)*100:.1f}%)")
        print(f"Unique tasks found: {len(all_tasks)}")
        
        # Most common entities
        dataset_counter = Counter()
        method_counter = Counter()
        metric_counter = Counter()
        task_counter = Counter()
        
        for paper_id, data in paper_entities.items():
            entities = data['entities']
            dataset_counter.update(entities['datasets'])
            method_counter.update(entities['methods'])
            metric_counter.update(entities['metrics'])
            task_counter.update(entities['tasks'])
        
        print("\n" + "="*60)
        print("TOP ENTITIES")
        print("="*60)
        
        print("\nTop 10 Datasets:")
        for entity, count in dataset_counter.most_common(10):
            print(f"  {entity:25s}: {count:3d} papers")
        
        print("\nTop 10 Methods:")
        for entity, count in method_counter.most_common(10):
            print(f"  {entity:25s}: {count:3d} papers")
        
        print("\nTop 10 Metrics:")
        for entity, count in metric_counter.most_common(10):
            print(f"  {entity:25s}: {count:3d} papers")
        
        print("\nTop 10 Tasks:")
        for entity, count in task_counter.most_common(10):
            print(f"  {entity:25s}: {count:3d} papers")

def main():
    """Main execution"""
    
    print("="*60)
    print("PHASE 3: ENTITY EXTRACTION")
    print("="*60)
    print("\nThis will extract scientific entities from papers:")
    print("  - Datasets (MNIST, ImageNet, etc.)")
    print("  - Methods (BERT, Transformer, etc.)")
    print("  - Metrics (Accuracy, F1, etc.)")
    print("  - Tasks (Classification, etc.)")
    print("\nEstimated time: 5-10 minutes\n")
    
    extractor = EntityExtractor()
    
    # Extract entities
    paper_entities = extractor.extract_from_chunks()
    
    # Save to file
    extractor.save_entities(paper_entities)
    
    # Print statistics
    extractor.print_statistics(paper_entities)
    
    print("\n✅ Phase 3 Complete!")
    print("\nNext: Run 6_build_knowledge_graph.py")

if __name__ == "__main__":
    main()