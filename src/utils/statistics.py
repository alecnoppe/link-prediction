import networkx as nx


class Stats:
    """
    A class to compute various network statistics for a given graph.
    """

    def __init__(self):
        pass

    def global_clustering_coefficient(self, graph: nx.Graph):
        """
        Computes the global clustering coefficient of the graph.

        Args:
            graph: nx.Graph; The input graph.

        Returns:
            float: The global clustering coefficient.
        """
        return self.gcc(graph=graph)

    def gcc(self, graph: nx.Graph):
        """
        Computes the global clustering coefficient.

        Args:
            graph: nx.Graph; The input graph.

        Returns:
            float: The global clustering coefficient.
        """
        # Count the number of triangles and wedges
        total_triangles: int = 0
        total_wedges: int = 0

        # Iterate over adjacency lists
        for current_node in graph.nodes:
            neighbors = set(graph.adj[current_node].keys())  # Get neighbors of the current node
            for neighbor in neighbors:
                # Get neighbors of the neighbor, excluding the current node
                neighbor_neighbors = set(graph.adj[neighbor].keys()).difference({current_node})
                # Find the intersection of neighbors to count triangles
                intersection = neighbors.intersection(neighbor_neighbors)
                total_triangles += len(intersection)
                total_wedges += len(neighbor_neighbors)

        # Divide by the number of possible permutations to mitigate double counting
        total_triangles = int(total_triangles / 6)
        total_wedges = int(total_wedges / 2)

        # Compute the global clustering coefficient
        return (3 * total_triangles) / total_wedges

    def node_clustering_coefficient(self, graph: nx.Graph, node: int):
        """
        Computes the (local) clustering coefficient for a specific node.

        Args:
            graph: nx.Graph; The input graph.
            node: int; The node for which to compute the clustering coefficient.

        Returns:
            float: The clustering coefficient of the node.
        """
        return self.c_i(graph=graph, node=node)

    def c_i(self, graph: nx.Graph, node: int):
        """
        Computes the (local) clustering coefficient for a specific node.

        Args:
            graph: nx.Graph; The input graph.
            node: int; The node for which to compute the clustering coefficient.

        Returns:
            float: The clustering coefficient of the node.
        """
        total_triangles = 0
        neighbors = set(graph.adj[node].keys())  # Get neighbors of the node
        for neighbor in neighbors:
            # Get neighbors of the neighbor, excluding the current node
            neighbor_neighbors = set(graph.adj[neighbor].keys()).difference({node})
            # Find the intersection of neighbors to count triangles
            intersection = neighbors.intersection(neighbor_neighbors)
            total_triangles += len(intersection)

        d_i = graph.degree(node)  # Degree of the node
        # Compute the clustering coefficient
        return (total_triangles) / (d_i * (d_i - 1)) if d_i > 1 else 0

    def average_node_clustering_coefficient(self, graph: nx.Graph):
        """
        Computes the average clustering coefficient of all nodes in the graph.

        Args:
            graph: nx.Graph; The input graph.

        Returns:
            float: The average clustering coefficient of the graph.
        """
        sum_c_i = 0
        for node in graph.nodes:
            sum_c_i += self.c_i(graph, node)

        # Compute the average clustering coefficient
        return sum_c_i / len(graph.nodes)

    def degree_dependent_clustering_coefficient(self, graph: nx.Graph, degree: int):
        """
        Computes the average clustering coefficient for nodes with a specific degree.

        Args:
            graph: nx.Graph; The input graph.
            degree: int; The degree for which to compute the clustering coefficient.

        Returns:
            float: The average clustering coefficient for nodes with the given degree.
        """
        Nk = 0  # Number of nodes with the given degree
        sum_c_i = 0  # Sum of clustering coefficients for nodes with the given degree
        for node in graph.nodes:
            if graph.degree[node] == degree:
                Nk += 1
                sum_c_i += self.c_i(graph, node)

        # Compute the average clustering coefficient for the given degree
        return sum_c_i / Nk if Nk > 0 else 0
