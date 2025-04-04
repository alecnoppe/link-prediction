import networkx as nx

from src.link_prediction.LinkPredictor import LinkPredictor


class SimilarityPredictor(LinkPredictor):
    """
    SimilarityPredictor class. Can use a flexible selection of similarity based metrics.
    
    NOTE: for metrics with parameter-dependent arguments, provide the similarity metric as a functools.partial() function
    with the desired arguments specified. 
    """
    def __init__(self, threshold:float, similarity:callable):
        """Init class for SimilarityPredictor.

        Args:
            threshold: float; threshold above which an edge is classified as missing and below which an edge is spurious
            similarity: callable; the similarity-based metric
        """
        self.threshold = threshold
        self.similarity = similarity
        
    def predict_link(self, graph: nx.Graph, source: int, target: int) -> bool:
        """Predict whether a link belongs in the observed graph G0

        Args:
            graph: nx.Graph; The observed graph G0
            source: int; The source node
            target: int; The target node

        Returns:
            bool: Whether the edge between (source, target) belongs in the observed graph G0
        """
        similarity_score = self.similarity(graph, source, target)
        return True if similarity_score >= self.threshold else False
    
    def predict_all_links(self, graph: nx.Graph, spurious: bool=False) -> nx.Graph:
        """
        Predict all missing (and spurious) links in an observed graph G0. Returns a new nx.Graph instance with the
        new links added (and spurious links removed if spurious=True).

        Args:
            graph: nx.Graph; The observed graph G0, for which we do link prediction.
            spurious: bool, optional; Whether to predict the spuriousness of edges, ie. whether to remove edgess. Defaults to False.

        Raises:
            NotImplementedError: spurious link prediction is not currently supported

        Returns:
            new_graph: nx.Graph; the output Graph, for which we have done link prediction.
        """
        new_graph = nx.Graph()
        # Add all nodes from G0 to the new graph
        new_graph.add_nodes_from(graph.nodes)
        if spurious:
            raise NotImplementedError("No support for spurious link prediction (yet)...")
            return new_graph
        # Since there are no spurious edges, add all edges from G0
        new_graph.add_edges_from(graph.edges)
        # Iterate over the nodes, and check if any node pairs (u,v) with distance d(u,v) > 1
        # If they're similar -> likely to have a missing edge
        nodes_set = set(graph.nodes)
        for node in nodes_set:
            # Iterate over all nodes that are not adjacent to `node`
            for other_node in nodes_set.difference(graph.adj[node]).difference({node}):
                # Compute similarity score R(u,v)
                similarity_score = self.similarity(graph, node, other_node)
                # Check if R(u,v) > r, if so: add the edge to the new graph
                if similarity_score >= self.threshold:
                    new_graph.add_edge(node, other_node)
                    
        return new_graph
    
    def add_top_k_links(self, graph: nx.Graph, k:int) -> nx.Graph:
        """
        Add edges between the top `k` most similar node pairs that currently do not have an edge in G0.

        Args:
            graph: nx.Graph; The observed graph G0, for which we do link prediction.
            k: int; Number of missing edges to add.

        Returns:
            new_grapH: nx.Graph; the output Graph, for which we have done link prediction.
        """
        new_graph = nx.Graph()
        # Add all nodes from G0 to the new graph
        new_graph.add_nodes_from(graph.nodes)
        new_graph.add_edges_from(graph.edges)
        # Iterate over the nodes, and check if any node pairs (u,v) with distance d(u,v) > 1
        # If they're similar -> likely to have a missing edge
        nodes_set = set(graph.nodes)
        edge_scores = []
        for node in nodes_set:
            # Iterate over all nodes that are not adjacent to `node`
            for other_node in nodes_set.difference(graph.adj[node]).difference({node}):
                # Compute similarity score R(u,v)
                similarity_score = self.similarity(graph, node, other_node)
                # Store the edge and similarity score
                edge_scores.append((similarity_score, node, other_node))
        # Add top k edges to the new graph        
        top_k_edges = sorted(edge_scores, lambda x: x[0])[:k]
        for score, source, target in top_k_edges:
            new_graph.add_edge(source, target)
        
        return new_graph
                