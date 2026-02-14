import networkx as nx
import community as community_louvain # Fixed import for python-louvain
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class KnowledgeNetwork:
    """
    Advanced Scientific Knowledge Graph.
    Implements TDMS Schema (Task, Dataset, Method, Metric) and PageRank scoring.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paper_embeddings = {} 

    def add_paper(self, paper_id: str, embedding: np.array, metadata: dict):
        """Adds a paper node with rich metadata."""
        self.graph.add_node(
            paper_id, 
            type="PAPER", 
            label=metadata.get('title', paper_id),
            year=metadata.get('year', 0),
            venue=metadata.get('venue', 'Unknown'),
            embedding=embedding
        )
        self.paper_embeddings[paper_id] = embedding

    def add_tdms_entity(self, paper_id: str, entity_name: str, entity_type: str, relation: str, evidence: str, cso_topics: list = None):
        """
        Adds a typed entity (METHOD, TASK, DATASET, METRIC) and links it to the paper.
        entity_type must be one of ['METHOD', 'TASK', 'DATASET', 'METRIC', 'RESULT']
        """
        entity_id = entity_name.upper().strip().replace(" ", "_")
        
        # 1. Ensure Entity Node Exists
        if entity_id not in self.graph:
            self.graph.add_node(entity_id, type=entity_type, label=entity_name)
            
        # Add CSO topics if provided
        if cso_topics:
            for topic in cso_topics:
                topic_id = f"CSO_{topic.upper().strip().replace(' ', '_')}"
                if topic_id not in self.graph:
                    self.graph.add_node(topic_id, type="TOPIC", label=topic)
                # Link Entity -> Topic
                self.graph.add_edge(entity_id, topic_id, relation="BELONGS_TO_TOPIC")
        
        # 2. Add Edge with 'Evidence' (Snippet) to ground the claim
        self.graph.add_edge(
            paper_id, 
            entity_id, 
            relation=relation, 
            evidence=evidence  # Store the text snippet that justified this link
        )

    def add_citation_link(self, source_id: str, target_id: str, intent="REFERENCES"):
        """
        Adds a directed citation edge.
        Intent can be: BACKGROUND, COMPARES_WITH, EXTENDS, USES
        """
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, relation=intent)

    def compute_importance(self):
        """
        Runs PageRank to identify 'Foundational Papers' and 'Dominant Methods'.
        """
        # Personalized PageRank prioritizing recent papers could be added here
        pr = nx.pagerank(self.graph, weight='weight')
        nx.set_node_attributes(self.graph, pr, 'importance')
        return pr

    def detect_communities(self):
        """
        Uses Louvain Modularity to find true sub-fields (e.g., 'PEFT', 'RAG').
        """
        # Louvain requires undirected graph
        undirected = self.graph.to_undirected()
        partition = community_louvain.best_partition(undirected)
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities: communities[comm_id] = []
            communities[comm_id].append(node)
            
        return communities

    def get_community_subgraph(self, community_nodes):
        """Returns the subgraph for a specific community to be analyzed."""
        return self.graph.subgraph(community_nodes).copy()

    def find_competitors(self, paper_id: str):
        """
        Identifies competing papers using Graph Reasoning logic (Section 5.1).
        Two papers are competitors if:
        1. They share the same TASK.
        2. They share the same DATASET.
        3. They have different METHOD nodes.
        OR
        4. There is a DIRECT "COMPARISON" or "CONTRADICTION" edge between them.
        """
        if paper_id not in self.graph: return []
        
        competitors = set()
        
        # Strategy A: Structural Overlap (Same Task + Dataset, Diff Method)
        # Get neighbors of type TASK and DATASET
        my_tasks = [n for n in self.graph.neighbors(paper_id) if self.graph.nodes[n].get('type') == 'TASK']
        my_datasets = [n for n in self.graph.neighbors(paper_id) if self.graph.nodes[n].get('type') == 'DATASET']
        my_methods = {n for n in self.graph.neighbors(paper_id) if self.graph.nodes[n].get('type') == 'METHOD'}
        
        for t in my_tasks:
            for d in my_datasets:
                # Find other papers connected to this Task AND Dataset
                task_papers = set(self.graph.predecessors(t))
                dataset_papers = set(self.graph.predecessors(d))
                
                potential_rivals = task_papers.intersection(dataset_papers)
                for rival in potential_rivals:
                    if rival != paper_id and self.graph.nodes[rival].get('type') == 'PAPER':
                        # Check 3: Disjoint Method Nodes
                        rival_methods = {n for n in self.graph.neighbors(rival) if self.graph.nodes[n].get('type') == 'METHOD'}
                        
                        # Competitors must not share the exact same method (that would be 'Same Story')
                        # AND both must actually have methods defined to be considered distinct approaches
                        if my_methods and rival_methods and my_methods.isdisjoint(rival_methods):
                             competitors.add(rival)

        # Strategy B: Explicit Contrastive Edges
        # Check outgoing edges for explicit comparisons
        for neighbor in self.graph.neighbors(paper_id):
            edge_data = self.graph.get_edge_data(paper_id, neighbor)
            if edge_data and edge_data.get('relation') in ['COMPARISON', 'CONTRADICTION', 'COMPARES_WITH']:
                 if self.graph.nodes[neighbor].get('type') == 'PAPER':
                     competitors.add(neighbor)

        return list(competitors)

    def analyze_conflict_path(self, source_pid: str, target_pid: str):
        """
        Reasoning: Graph Traversal (Pathfinding).
        Finds the specific chain of citations/comparisons that connects two competing papers.
        Used to generate 'informative' explanations of rivalry.
        """
        try:
             # Create a view of the graph with only argumentative edges
            def filter_edge(n1, n2):
                edge_data = self.graph.get_edge_data(n1, n2)
                if not edge_data: return False
                rel = edge_data.get('relation', '')
                return rel in ['COMPARISON', 'CONTRADICTION', 'COMPARES_WITH', 'EXTENDS', 'REFERENCES']
            
            # We treat this as undirected for finding the "connection" in the discourse
            view = nx.subgraph_view(self.graph, filter_edge=filter_edge).to_undirected()
            
            path = nx.shortest_path(view, source=source_pid, target=target_pid)
            
            # Re-construct the path with edge labels
            explanation = []
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                # Check original direction
                if self.graph.has_edge(u, v):
                    rel = self.graph[u][v].get('relation')
                    explanation.append(f"{self.graph.nodes[u].get('label')} --[{rel}]--> {self.graph.nodes[v].get('label')}")
                elif self.graph.has_edge(v, u):
                    rel = self.graph[v][u].get('relation')
                    explanation.append(f"{self.graph.nodes[v].get('label')} --[{rel}]--> {self.graph.nodes[u].get('label')}")
            
            return " -> ".join(explanation)

        except nx.NetworkXNoPath:
            return None
        except Exception:
            return None

    def run_ppr_search(self, seed_nodes: list, alpha=0.85, max_iter=100):
        """
        Implements HippoRAG's Associative Memory retrieval.
        Runs Personalized PageRank (PPR) starting from specific seed nodes (e.g., query entities).
        """
        personalization = {node: 0.0 for node in self.graph.nodes()}
        
        # Distribute probability mass among seed nodes
        seed_count = len(seed_nodes)
        if seed_count == 0: return {}
        
        for node in seed_nodes:
            if node in personalization:
                personalization[node] = 1.0 / seed_count
        
        # Run PPR
        try:
            ppr_scores = nx.pagerank(
                self.graph, 
                alpha=alpha, 
                personalization=personalization, 
                max_iter=max_iter,
                weight='weight' # Assuming edge weights exist, or defaults to 1
            )
        except nx.PowerIterationFailedConvergence:
            return {}

        # Sort by score
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results excluding the seeds themselves
        return [n for n, score in sorted_nodes if n not in seed_nodes and score > 0.001]

    def find_similar_papers(self, paper_id: str, threshold=0.3):
        """
        Identifies 'Same Story' / Redundant papers (Section 5.2).
        Uses Jaccard Similarity of structural neighborhoods.
        """
        if paper_id not in self.graph: return []
        
        candidates = []
        my_neighbors = set(self.graph.neighbors(paper_id))
        if not my_neighbors: return []

        # Iterate over all other papers (optimization: restrict to same community)
        for node in self.graph.nodes():
            if node != paper_id and self.graph.nodes[node].get('type') == 'PAPER':
                other_neighbors = set(self.graph.neighbors(node))
                if not other_neighbors: continue
                
                # Jaccard Similarity
                intersection = len(my_neighbors.intersection(other_neighbors))
                union = len(my_neighbors.union(other_neighbors))
                score = intersection / union if union > 0 else 0
                
                if score > threshold:
                    candidates.append((node, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]

    def get_related_by_structure(self, paper_id, top_k=5):
        """
        HippoRAG-style retrieval: Finds papers sharing structural neighbors 
        (Same Task + Same Dataset) even if they don't cite each other.
        """
        if paper_id not in self.graph: return []
        
        # Get neighbors (Methods, Datasets, Tasks)
        neighbors = list(self.graph.neighbors(paper_id))
        
        # Find other papers connecting to these neighbors
        candidates = []
        for n in neighbors:
            for second_hop in self.graph.predecessors(n): # Papers pointing to this Method/Dataset
                if second_hop != paper_id and self.graph.nodes[second_hop].get('type') == 'PAPER':
                    candidates.append(second_hop)
        
        # Return most frequent co-occurrences
        return [c for c, count in Counter(candidates).most_common(top_k)]

    def trace_lineage(self, paper_id: str):
        """
        Traces the intellectual lineage (Section 5.3).
        Returns a list of 'Foundational' papers (ancestors via background/method citations)
        and 'Derivative' papers (descendants via extension citations).
        """
        if paper_id not in self.graph: return {'foundational': [], 'derivative': []}

        # 1. Foundational (Backward traversal on BACKGROUND/METHOD/USES edges)
        foundational = []
        # Local BFS upstream
        q = [paper_id]
        visited = {paper_id}
        depth = 0
        while q and depth < 3:
            current = q.pop(0)
            for cited in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, cited)
                relation = edge_data.get('relation', '')
                if relation in ['BACKGROUND', 'METHOD', 'USES', 'REFERENCES']:
                    if cited not in visited and self.graph.nodes[cited].get('type') == 'PAPER':
                        foundational.append(cited)
                        visited.add(cited)
                        q.append(cited)
            depth += 1

        # 2. Derivative (Forward traversal on EXTENSION/COMPARISON edges)
        derivative = []
        # Local BFS downstream
        q = [paper_id]
        visited_deriv = {paper_id}
        depth = 0
        while q and depth < 3:
            current = q.pop(0)
            for citing in self.graph.predecessors(current):
                edge_data = self.graph.get_edge_data(citing, current)
                relation = edge_data.get('relation', '')
                if relation in ['EXTENSION', 'COMPARES_WITH', 'CONTRADICTION']:
                    if citing not in visited_deriv and self.graph.nodes[citing].get('type') == 'PAPER':
                        derivative.append(citing)
                        visited_deriv.add(citing)
                        q.append(citing)
            depth += 1
            
        return {'foundational': list(set(foundational)), 'derivative': list(set(derivative))}