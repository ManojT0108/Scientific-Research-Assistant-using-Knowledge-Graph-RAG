"""
IMPROVED Knowledge Graph Builder

Key improvements:
1. Filters generic/uninformative entities
2. TF-IDF weighting for entity importance
3. Quality-based paper connections (not just any shared entity)
4. Better statistics and validation

Runtime: ~2-3 minutes
"""

import json
import networkx as nx
from collections import defaultdict, Counter
import pickle
import math

class ImprovedKnowledgeGraphBuilder:
    """
    Build high-quality knowledge graph with entity filtering and weighting
    """

    # Generic entities that add noise, not signal
    GENERIC_ENTITIES = {
        # Publication venues (NOT datasets!)
        'arxiv', 'github', 'pubmed', 'wikipedia', 'huggingface',
        'openai', 'google', 'microsoft', 'meta', 'anthropic',

        # Too common metrics
        'accuracy', 'loss', 'precision', 'recall', 'f1', 'auc',

        # Too common optimizers/methods
        'adam', 'sgd', 'optimizer', 'gradient descent',

        # Too common architectures (appear in 30%+ of papers)
        'attention', 'transformer', 'bert', 'gpt', 'llama',
        'neural network', 'deep learning', 'machine learning',

        # Too generic tasks
        'training', 'inference', 'evaluation', 'testing',
    }

    def __init__(self):
        self.graph = nx.Graph()
        self.paper_connections = defaultdict(set)
        self.entity_idf = {}  # Inverse document frequency scores
        self.filtered_entities = set()  # Track what we filtered

    def load_entities(self, entities_file='data/processed/entities.json'):
        """Load extracted entities"""
        print("Loading entities...")
        with open(entities_file) as f:
            self.entities = json.load(f)
        print(f"Loaded entities for {len(self.entities)} papers")

    def compute_entity_idf(self):
        """
        Compute IDF (inverse document frequency) for all entities

        IDF = log(total_papers / papers_containing_entity)
        Higher IDF = more rare/specific = more informative
        """
        print("\nComputing entity importance (TF-IDF)...")

        entity_doc_count = Counter()
        total_papers = len(self.entities)

        # Count how many papers each entity appears in
        for paper_id, data in self.entities.items():
            seen_entities = set()
            for entity_type in ['datasets', 'methods', 'metrics', 'tasks']:
                for entity in data['entities'][entity_type]:
                    entity_lower = entity.lower()
                    if entity_lower not in self.GENERIC_ENTITIES:
                        seen_entities.add(entity)

            for entity in seen_entities:
                entity_doc_count[entity] += 1

        # Compute IDF scores
        for entity, doc_count in entity_doc_count.items():
            self.entity_idf[entity] = math.log(total_papers / doc_count)

        # Statistics
        idf_values = list(self.entity_idf.values())
        if idf_values:
            print(f"   Computed IDF for {len(self.entity_idf)} entities")
            print(f"   IDF range: {min(idf_values):.2f} to {max(idf_values):.2f}")
            print(f"   Mean IDF: {sum(idf_values)/len(idf_values):.2f}")

    def is_entity_informative(self, entity: str, entity_type: str) -> bool:
        """
        Decide if an entity should be included in the graph

        Filters out:
        - Generic entities (from GENERIC_ENTITIES set)
        - Entities appearing in >50% of papers (too common)
        - Very rare entities (<3 papers) for certain types
        """
        entity_lower = entity.lower()

        # Filter generic entities
        if entity_lower in self.GENERIC_ENTITIES:
            self.filtered_entities.add(entity)
            return False

        # Get IDF if available
        if entity in self.entity_idf:
            idf = self.entity_idf[entity]

            # Filter entities appearing in >50% of papers (IDF < 0.69)
            if idf < 0.69:
                self.filtered_entities.add(entity)
                return False

            # For methods, only keep if rare enough (IDF > 1.0)
            # This filters out semi-common methods
            if entity_type == 'method' and idf < 1.0:
                self.filtered_entities.add(entity)
                return False

        return True

    def build_graph(self):
        """
        Build bipartite graph: Papers <-> High-quality entities
        """
        print("\nBuilding knowledge graph (with filtering)...")

        # Add paper nodes
        for paper_id, data in self.entities.items():
            self.graph.add_node(
                paper_id,
                node_type='paper',
                title=data['title'],
                year=data['year'],
                categories=data['categories']
            )

        print(f"Added {self.graph.number_of_nodes()} paper nodes")

        # Track entity type counts
        entity_counts = defaultdict(int)
        filtered_counts = defaultdict(int)

        # Add entity nodes and edges (with filtering)
        for paper_id, data in self.entities.items():
            entities = data['entities']

            # Connect to datasets
            for dataset in entities['datasets']:
                if self.is_entity_informative(dataset, 'dataset'):
                    entity_id = f"dataset:{dataset}"
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            node_type='dataset',
                            name=dataset,
                            idf=self.entity_idf.get(dataset, 0)
                        )
                    self.graph.add_edge(paper_id, entity_id, relation='uses')
                    entity_counts['dataset'] += 1
                else:
                    filtered_counts['dataset'] += 1

            # Connect to methods
            for method in entities['methods']:
                if self.is_entity_informative(method, 'method'):
                    entity_id = f"method:{method}"
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            node_type='method',
                            name=method,
                            idf=self.entity_idf.get(method, 0)
                        )
                    self.graph.add_edge(paper_id, entity_id, relation='applies')
                    entity_counts['method'] += 1
                else:
                    filtered_counts['method'] += 1

            # Connect to metrics
            for metric in entities['metrics']:
                if self.is_entity_informative(metric, 'metric'):
                    entity_id = f"metric:{metric}"
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            node_type='metric',
                            name=metric,
                            idf=self.entity_idf.get(metric, 0)
                        )
                    self.graph.add_edge(paper_id, entity_id, relation='evaluates')
                    entity_counts['metric'] += 1
                else:
                    filtered_counts['metric'] += 1

            # Connect to tasks
            for task in entities['tasks']:
                if self.is_entity_informative(task, 'task'):
                    entity_id = f"task:{task}"
                    if not self.graph.has_node(entity_id):
                        self.graph.add_node(
                            entity_id,
                            node_type='task',
                            name=task,
                            idf=self.entity_idf.get(task, 0)
                        )
                    self.graph.add_edge(paper_id, entity_id, relation='addresses')
                    entity_counts['task'] += 1
                else:
                    filtered_counts['task'] += 1

        print(f"✅ Graph built:")
        print(f"   Nodes: {self.graph.number_of_nodes():,}")
        print(f"   Edges: {self.graph.number_of_edges():,}")
        print(f"\n   Entities kept: {sum(entity_counts.values()):,}")
        print(f"   Entities filtered: {sum(filtered_counts.values()):,}")

        for entity_type in ['dataset', 'method', 'metric', 'task']:
            kept = entity_counts[entity_type]
            filtered = filtered_counts[entity_type]
            total = kept + filtered
            if total > 0:
                pct = (kept / total) * 100
                print(f"     {entity_type:10s}: kept {kept:4d} / {total:4d} ({pct:.1f}%)")

    def compute_paper_similarities(self):
        """
        Compute WEIGHTED paper-to-paper connections

        Papers are connected if they share:
        - 2+ entities of any type, OR
        - 1 high-IDF entity (IDF > 2.5), OR
        - 1 task entity (tasks are very specific)
        """
        print("\nComputing weighted paper-to-paper similarities...")

        # For each entity, find papers that use it
        entity_to_papers = defaultdict(set)

        for node, data in self.graph.nodes(data=True):
            if data['node_type'] != 'paper':
                # This is an entity - find connected papers
                papers = [n for n in self.graph.neighbors(node)
                         if self.graph.nodes[n]['node_type'] == 'paper']
                entity_to_papers[node].update(papers)

        # Connect papers with quality thresholds
        connection_count = 0
        weighted_scores = []

        for entity, papers in entity_to_papers.items():
            papers_list = list(papers)
            entity_data = self.graph.nodes[entity]
            entity_type = entity_data['node_type']
            entity_idf = entity_data.get('idf', 1.0)

            for i in range(len(papers_list)):
                for j in range(i+1, len(papers_list)):
                    paper1, paper2 = papers_list[i], papers_list[j]

                    # Compute weighted similarity score
                    shared_entities = (
                        set(self.graph.neighbors(paper1)) &
                        set(self.graph.neighbors(paper2))
                    )

                    # Weighted score based on IDF
                    weighted_score = sum(
                        self.graph.nodes[e].get('idf', 1.0)
                        for e in shared_entities
                    )

                    # Quality threshold: require meaningful connection
                    # Accept if:
                    # - Share 2+ entities, OR
                    # - Share 1 high-IDF entity (>2.5), OR
                    # - Share a task (tasks are specific)
                    num_shared = len(shared_entities)
                    has_task = any(
                        self.graph.nodes[e]['node_type'] == 'task'
                        for e in shared_entities
                    )
                    max_idf = max(
                        (self.graph.nodes[e].get('idf', 0) for e in shared_entities),
                        default=0
                    )

                    if num_shared >= 2 or max_idf > 2.5 or has_task:
                        self.paper_connections[paper1].add(paper2)
                        self.paper_connections[paper2].add(paper1)
                        connection_count += 1
                        weighted_scores.append(weighted_score)

        print(f"✅ Found {connection_count:,} high-quality paper connections")

        # Statistics
        connection_counts = [len(conns) for conns in self.paper_connections.values()]
        if connection_counts:
            avg_connections = sum(connection_counts) / len(connection_counts)
            max_connections = max(connection_counts)
            median_connections = sorted(connection_counts)[len(connection_counts)//2]

            print(f"   Average connections per paper: {avg_connections:.1f}")
            print(f"   Median connections per paper: {median_connections}")
            print(f"   Max connections: {max_connections}")

            # Compare to total possible
            num_papers = len([n for n, d in self.graph.nodes(data=True) if d['node_type'] == 'paper'])
            max_possible = num_papers - 1
            density = (avg_connections / max_possible) * 100
            print(f"   Graph density: {density:.1f}% (lower is better!)")

    def get_related_papers(self, paper_id, top_k=10):
        """
        Get papers most related to given paper with IDF-weighted scoring
        """
        if paper_id not in self.paper_connections:
            return []

        # Count weighted shared entities
        shared_scores = {}

        for related_paper in self.paper_connections[paper_id]:
            # Get shared entities
            paper1_neighbors = set(self.graph.neighbors(paper_id))
            paper2_neighbors = set(self.graph.neighbors(related_paper))
            shared_entities = paper1_neighbors & paper2_neighbors

            # Weighted score based on entity IDF
            weighted_score = sum(
                self.graph.nodes[entity].get('idf', 1.0)
                for entity in shared_entities
            )

            shared_scores[related_paper] = weighted_score

        # Return top-k by weighted score
        return sorted(shared_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def save_graph(self, graph_file='data/processed/knowledge_graph.pkl',
                   connections_file='data/processed/paper_connections.json'):
        """Save improved graph to file"""

        # Save full graph
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"\n✅ Saved graph to {graph_file}")

        # Save paper connections
        connections_json = {
            paper: list(related)
            for paper, related in self.paper_connections.items()
        }
        with open(connections_file, 'w') as f:
            json.dump(connections_json, f)
        print(f"✅ Saved connections to {connections_file}")

    def print_statistics(self):
        """Print detailed graph statistics with quality indicators"""

        # Node type counts
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data['node_type']] += 1

        print("\n" + "="*60)
        print("IMPROVED KNOWLEDGE GRAPH STATISTICS")
        print("="*60)

        print("\nNode Types:")
        for node_type, count in sorted(node_types.items()):
            print(f"  {node_type:15s}: {count:5,} nodes")

        # Edge counts by relation
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            relation_counts[data['relation']] += 1

        print("\nEdge Types:")
        for relation, count in sorted(relation_counts.items()):
            print(f"  {relation:15s}: {count:5,} edges")

        # Most connected entities (with IDF scores)
        print("\nMost Connected Entities (Informative Only):")
        entity_degrees = []
        for node, data in self.graph.nodes(data=True):
            if data['node_type'] != 'paper':
                degree = self.graph.degree(node)
                idf = data.get('idf', 0)
                entity_degrees.append((node, data.get('name', node), degree, idf))

        entity_degrees.sort(key=lambda x: x[2], reverse=True)

        for node_id, name, degree, idf in entity_degrees[:20]:
            node_type = node_id.split(':')[0]
            print(f"  {name:35s} ({node_type:8s}): {degree:3d} papers | IDF: {idf:.2f}")

        # Show filtered entities
        print(f"\n🚫 Filtered Entities (Top 20 Generic):")
        filtered_list = sorted(list(self.filtered_entities))[:20]
        print(f"   {', '.join(filtered_list)}")

        # Sample related papers
        print("\n" + "="*60)
        print("SAMPLE RELATED PAPERS (Weighted by IDF)")
        print("="*60)

        if self.paper_connections:
            sample_paper = max(self.paper_connections.items(),
                             key=lambda x: len(x[1]))[0]

            paper_title = self.graph.nodes[sample_paper]['title']
            print(f"\nPaper: {paper_title}")
            print(f"Paper ID: {sample_paper}")

            related = self.get_related_papers(sample_paper, top_k=5)
            print(f"\nTop 5 Related Papers (by weighted similarity):")
            for related_paper, score in related:
                related_title = self.graph.nodes[related_paper]['title']
                print(f"  [score: {score:.2f}] {related_title[:70]}...")

def main():
    """Main execution"""

    print("="*60)
    print("IMPROVED KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*60)
    print("\nImprovements over basic KG:")
    print("  ✓ Filters generic/uninformative entities")
    print("  ✓ TF-IDF weighting for entity importance")
    print("  ✓ Quality-based paper connections")
    print("  ✓ Prevents over-connection through noise")
    print("\nEstimated time: 2-3 minutes\n")

    builder = ImprovedKnowledgeGraphBuilder()

    # Load entities
    builder.load_entities()

    # Compute entity importance
    builder.compute_entity_idf()

    # Build graph with filtering
    builder.build_graph()

    # Compute weighted similarities
    builder.compute_paper_similarities()

    # Save graph (overwrites old one)
    builder.save_graph()

    # Print statistics
    builder.print_statistics()

    print("\n✅ Improved KG Complete!")
    print("\n💡 Key Improvements:")
    print("   - Much lower connection density (less noise)")
    print("   - Entities are now informative (no 'arxiv', 'github')")
    print("   - Connections weighted by entity specificity")
    print("\nNext: Re-run claude_rag.py to see quality improvement!")

if __name__ == "__main__":
    main()
