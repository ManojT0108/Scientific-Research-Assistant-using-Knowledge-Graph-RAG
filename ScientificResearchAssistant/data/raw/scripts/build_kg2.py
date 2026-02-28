"""
Script 6: Build Knowledge Graph from extracted entities

Creates graph with:
- Nodes: Papers, Datasets, Methods, Metrics, Tasks
- Edges: uses, applies, evaluates, addresses

Runtime: ~2-3 minutes
"""

import json
import networkx as nx
from collections import defaultdict, Counter
import pickle

class KnowledgeGraphBuilder:
    """
    Build knowledge graph connecting papers through shared entities
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.paper_connections = defaultdict(set)
    
    def load_entities(self, entities_file='data/processed/entities.json'):
        """Load extracted entities"""
        print("Loading entities...")
        with open(entities_file) as f:
            self.entities = json.load(f)
        print(f"Loaded entities for {len(self.entities)} papers")
    
    def build_graph(self):
        """
        Build bipartite graph: Papers <-> Entities
        
        Graph structure:
        - Paper nodes: {type: 'paper', ...}
        - Entity nodes: {type: 'dataset/method/metric/task', name: '...'}
        - Edges: papers connected to their entities
        """
        print("\nBuilding knowledge graph...")
        
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
        
        # Add entity nodes and edges
        for paper_id, data in self.entities.items():
            entities = data['entities']
            
            # Connect to datasets
            for dataset in entities['datasets']:
                entity_id = f"dataset:{dataset}"
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, node_type='dataset', name=dataset)
                self.graph.add_edge(paper_id, entity_id, relation='uses')
            
            # Connect to methods
            for method in entities['methods']:
                entity_id = f"method:{method}"
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, node_type='method', name=method)
                self.graph.add_edge(paper_id, entity_id, relation='applies')
            
            # Connect to metrics
            for metric in entities['metrics']:
                entity_id = f"metric:{metric}"
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, node_type='metric', name=metric)
                self.graph.add_edge(paper_id, entity_id, relation='evaluates')
            
            # Connect to tasks
            for task in entities['tasks']:
                entity_id = f"task:{task}"
                if not self.graph.has_node(entity_id):
                    self.graph.add_node(entity_id, node_type='task', name=task)
                self.graph.add_edge(paper_id, entity_id, relation='addresses')
        
        print(f"✅ Graph built:")
        print(f"   Nodes: {self.graph.number_of_nodes():,}")
        print(f"   Edges: {self.graph.number_of_edges():,}")
    
    def compute_paper_similarities(self):
        """
        Compute paper-to-paper connections through shared entities
        
        Two papers are connected if they share entities
        """
        print("\nComputing paper-to-paper similarities...")
        
        # For each entity, find papers that use it
        entity_to_papers = defaultdict(set)
        
        for node, data in self.graph.nodes(data=True):
            if data['node_type'] != 'paper':
                # This is an entity - find connected papers
                papers = [n for n in self.graph.neighbors(node) 
                         if self.graph.nodes[n]['node_type'] == 'paper']
                entity_to_papers[node].update(papers)
        
        # Connect papers that share entities
        connection_count = 0
        for entity, papers in entity_to_papers.items():
            papers_list = list(papers)
            for i in range(len(papers_list)):
                for j in range(i+1, len(papers_list)):
                    paper1, paper2 = papers_list[i], papers_list[j]
                    self.paper_connections[paper1].add(paper2)
                    self.paper_connections[paper2].add(paper1)
                    connection_count += 1
        
        print(f"✅ Found {connection_count:,} paper-to-paper connections")
        
        # Statistics
        connection_counts = [len(conns) for conns in self.paper_connections.values()]
        if connection_counts:
            avg_connections = sum(connection_counts) / len(connection_counts)
            max_connections = max(connection_counts)
            print(f"   Average connections per paper: {avg_connections:.1f}")
            print(f"   Max connections: {max_connections}")
    
    def get_related_papers(self, paper_id, top_k=10):
        """
        Get papers most related to given paper
        
        Based on number of shared entities
        """
        if paper_id not in self.paper_connections:
            return []
        
        # Count shared entities with each related paper
        shared_entity_counts = Counter()
        
        for related_paper in self.paper_connections[paper_id]:
            # Count shared entities
            shared = 0
            
            # Get entities for both papers
            paper1_neighbors = set(self.graph.neighbors(paper_id))
            paper2_neighbors = set(self.graph.neighbors(related_paper))
            
            # Count shared entity nodes
            shared = len(paper1_neighbors & paper2_neighbors)
            shared_entity_counts[related_paper] = shared
        
        # Return top-k most related
        return shared_entity_counts.most_common(top_k)
    
    def save_graph(self, graph_file='data/processed/knowledge_graph.pkl',
                   connections_file='data/processed/paper_connections.json'):
        """Save graph to file"""
        
        # Save full graph
        with open(graph_file, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"\n✅ Saved graph to {graph_file}")
        
        # Save paper connections (JSON for easier access)
        connections_json = {
            paper: list(related) 
            for paper, related in self.paper_connections.items()
        }
        with open(connections_file, 'w') as f:
            json.dump(connections_json, f)
        print(f"✅ Saved connections to {connections_file}")
    
    def print_statistics(self):
        """Print detailed graph statistics"""
        
        # Node type counts
        node_types = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            node_types[data['node_type']] += 1
        
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH STATISTICS")
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
        
        # Most connected entities
        print("\nMost Connected Entities:")
        entity_degrees = []
        for node, data in self.graph.nodes(data=True):
            if data['node_type'] != 'paper':
                degree = self.graph.degree(node)
                entity_degrees.append((node, data.get('name', node), degree))
        
        entity_degrees.sort(key=lambda x: x[2], reverse=True)
        
        for node_id, name, degree in entity_degrees[:15]:
            node_type = node_id.split(':')[0]
            print(f"  {name:30s} ({node_type:8s}): {degree:3d} papers")
        
        # Sample related papers
        print("\n" + "="*60)
        print("SAMPLE RELATED PAPERS")
        print("="*60)
        
        # Pick a paper with many connections
        if self.paper_connections:
            sample_paper = max(self.paper_connections.items(), 
                             key=lambda x: len(x[1]))[0]
            
            paper_title = self.graph.nodes[sample_paper]['title']
            print(f"\nPaper: {paper_title}")
            print(f"Paper ID: {sample_paper}")
            
            related = self.get_related_papers(sample_paper, top_k=5)
            print(f"\nTop 5 Related Papers (by shared entities):")
            for related_paper, shared_count in related:
                related_title = self.graph.nodes[related_paper]['title']
                print(f"  [{shared_count} shared] {related_title[:80]}...")

def main():
    """Main execution"""
    
    print("="*60)
    print("PHASE 3: KNOWLEDGE GRAPH CONSTRUCTION")
    print("="*60)
    print("\nThis will build a knowledge graph connecting papers")
    print("through shared datasets, methods, metrics, and tasks.")
    print("\nEstimated time: 2-3 minutes\n")
    
    builder = KnowledgeGraphBuilder()
    
    # Load entities
    builder.load_entities()
    
    # Build graph
    builder.build_graph()
    
    # Compute paper similarities
    builder.compute_paper_similarities()
    
    # Save graph
    builder.save_graph()
    
    # Print statistics
    builder.print_statistics()
    
    print("\n✅ Phase 3 Complete!")
    print("\nNext: Run 7_hybrid_search.py for Phase 4")

if __name__ == "__main__":
    main()