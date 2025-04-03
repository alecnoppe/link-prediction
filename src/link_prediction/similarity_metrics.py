"""
All similarity metrics used here were adapted from:
- Lü, L., & Zhou, T. (2011). Link prediction in complex networks: A survey. Physica A: statistical mechanics and its applications, 390(6), 1150-1170.

In the docstrings, I mention the Index of the similarity-based method and a brief summary of the metric.
For a more in-depth analysis, see the survey paper listed above (and of course the original papers from which
the metrics were adapted).

The following notation is used:
- u: source node
- v: target node
- k(u): the degree of u
- N(u): the neighborhood of u
- S: the similarity score

For any unclear notation, see the survey paper.
"""

import networkx as nx
from math import sqrt, log
import numpy as np


"""Local similarity metrics"""
def similarity_common_neighbors(graph: nx.Graph, source: int, target: int) -> float:
    """
    Common Neighbors (CN) similarity Index (1) is the cardinality of the intersection between the neighborhoods;
    the number of shared neighbors between two nodes
    
    `S = CN(u,v) = | N(u) ∩ N(v) |`
     
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return len(set(graph.adj[source]).intersection(graph.adj[target]))

def similarity_salton_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Salton Index (2): common neighbors divided by the sqrt of the degree product; penalizing popular vertices
    
    `S = CN(u,v) / sqrt( k(u) * k(v) )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return similarity_common_neighbors(graph, source, target) / sqrt(graph.degree[source] * graph.degree[target])

def similarity_jaccard_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Jaccard Index (3): Common neighbors divided by the total number of unique neighbors of u,v; penalizes popular vertices
    (but less strongly than Salton)
    
    `S = CN(u,v) / | N(u) ∪ N(v) |`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return similarity_common_neighbors(graph, source, target) / len(set(graph.adj[source]).union(graph.adj[target]))

def similarity_sorenson_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Sorenson Index (4): Scaled common neighbors divided by the sum of the degrees of u, v
    
    `S = CN(u,v) / ( k(u) + k(v) )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return 2 * similarity_common_neighbors(graph, source, target) / (graph.degree[source] + graph.degree[target])

def similarity_hub_promoted_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Hub Promoted Index (5): Common neighbors divided by the minimum degree of u, v
    
    `S = CN(u,v) / min( k(u), k(v) )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return similarity_common_neighbors(graph, source, target) / min(graph.degree[source], graph.degree[target])

def similarity_hub_depressed_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Hub Depressed Index (6): Common neighbors divided by the maximum degree of u, v
    
    `S = CN(u,v) / max( k(u), k(v) )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return similarity_common_neighbors(graph, source, target) / max(graph.degree[source], graph.degree[target])

def similarity_leicht_holme_newman_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    LHN1 Index (7): Common neighbors divided by the product of the degrees of u, v
    
    `S = CN(u,v) / ( k(u) * k(v) )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return similarity_common_neighbors(graph, source, target) / (graph.degree[source] * graph.degree[target])

def similarity_preferential_attachment_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Preferential Attachment Index (8): Product of the degrees of u, v; promotes connecting high degree vertices.
    
    `S = k(u) * k(v)`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return (graph.degree[source] * graph.degree[target])

def similarity_adamic_adar_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Adamic Adar Index (9): Sum of the inverse log degree of each common neighbor; assigns more weight to common neighbors with
    fewer connections
    
    `S = Sum[  1 / ( log k(i))  ]` for i in N(u) ∩ N(v)
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return sum([1/log(graph.degree[shared_neighbor]) for shared_neighbor in set(graph.adj[source]).intersection(graph.adj[target])])

def similarity_resource_allocation_index(graph: nx.Graph, source: int, target: int) -> float:
    """
    Resource Allocation Index (10): Sum of the inverse degree of each common neighbor; assigns more weight to common neighbors with
    fewer connections
    
    `S = Sum[  1 / ( k(i))  ]` for i in N(u) ∩ N(v)
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return sum([1/graph.degree[shared_neighbor] for shared_neighbor in set(graph.adj[source]).intersection(graph.adj[target])])

"""Global similarity metrics"""

def similarity_katz_index(graph: nx.Graph, source: int, target: int, alpha: float) -> float:
    """
    Katz Index (11): Based on the notion that similar neighbors are similar; that neighbors with many short paths 
    between them are similar.
    
    `S = (I - a * A)^-1 - I`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        alpha: float; between 0 and 1, decay factor
        
    Returns:
        float: similarity according to the metric
    """
    I:np.ndarray = np.identity(n=graph.number_of_nodes(), dtype=int)
    A:np.ndarray = np.array(nx.adjacency_matrix(graph).toarray(), dtype=int)
    similarity = np.linalg.inv(I - (alpha * A))-I
    return similarity[source,target]
    
def similarity_LHN2(graph: nx.Graph, source: int, target: int, alpha: float) -> float:
    """
    LHN2 (12): Variant of Katz index, based on the notion that nodes with similar neighbors are similar
    
    `S = 2 * |E| * l_1 * D^-1 * ( I -  (a * A) / l_1)^-1 @ D^-1
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        alpha: float; between 0 and 1, decay factor
        
    Returns:
        float: similarity according to the metric
    """
    I:np.ndarray = np.identity(n=graph.number_of_nodes(), dtype=int)
    A:np.ndarray = np.array(nx.adjacency_matrix(graph).toarray(), dtype=int)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    l = max(eigenvalues)
    E = graph.number_of_edges()
    return (2 * E * l) / (graph.degree[source] * graph.degree[target]) * np.linalg.inv( (I - ((alpha)/l * A)))[source, target]
    
def similarity_average_commute_time(graph: nx.Graph, source: int, target: int) -> float:
    """
    Average Commute Time (13): Average number of steps of a random walker between u, v
    
    `S = 1 / (l_xx + l_yy - 2 * l_xy) `
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    A:np.ndarray = np.array(nx.adjacency_matrix(graph).toarray(), dtype=int)
    D:np.ndarray = np.identity(n=graph.number_of_nodes())
    for node in graph.nodes:
        D[node,node] = graph.degree[node]
    Lplus = np.linalg.pinv(D - A)
    lxx = Lplus[source,source]
    lyy = Lplus[target,target]
    lxy = Lplus[source,target]
    return 1 / (lxx + lyy - 2 * lxy)

def similarity_cosine_pinv_laplacian(graph: nx.Graph, source: int, target: int) -> float:
    """
    Cosine Based on Partial-Inverted Laplacian (14): Inner-product based measure.
    
    `S = l_xy / sqrt( lxx + lyy )`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    A:np.ndarray = np.array(nx.adjacency_matrix(graph).toarray(), dtype=int)
    D:np.ndarray = np.identity(n=graph.number_of_nodes())
    for node in graph.nodes:
        D[node,node] = graph.degree[node]
    Lplus = np.linalg.pinv(D - A)
    lxx = Lplus[source,source]
    lyy = Lplus[target,target]
    lxy = Lplus[source,target]
    return (lxy) / sqrt(lxx * lyy)

def similarity_random_walk_with_restart(graph: nx.Graph, source: int, target: int, c:float) -> float:
    """
    Random Walk with Restart (15): Application of the PageRank algorithm. 
    
    `S = q_xy + q_yx`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        c: float: between 0 and 1, probability of restarting
        
    Returns:
        float: similarity according to the metric
    """
    I:np.ndarray = np.identity(n=graph.number_of_nodes(), dtype=int)
    N = graph.number_of_nodes()
    # Compute the transition matrix (P)
    transition_matrix = np.zeros(shape=(N,N),dtype=float)
    for node in graph.nodes:
        k_x = graph.degree[node]
        for neighbor in graph.adj[node]:
            transition_matrix[node,neighbor] = 1/k_x
    # Initialize starting densities
    e_x = np.zeros(shape=N)
    e_x[source] = 1
    e_y = np.zeros(shape=N)
    e_y[target] = 1
    # Compute the probability that the random walker arrives at the other node in the steady state.
    q_x = (1-c) * np.linalg.inv(I-c*transition_matrix.T) @ e_x
    q_y = (1-c) * np.linalg.inv(I-c*transition_matrix.T) @ e_y
    
    return q_x[target] + q_y[source]
    
def similarity_simrank(graph: nx.Graph, source: int, target: int) -> float:
    """
    SimRank (16): Similar to LHN2, nodes are similar if they are connected to similar nodes
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        
    Returns:
        float: similarity according to the metric
    """
    return nx.simrank_similarity(graph, source, target)            

def similarity_matrix_forest_index(graph: nx.Graph, source: int, target: int, alpha: float) -> float:
    """
    Matrix Forest Index (17)
    
    `S = (I + a * L)^-1`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        alpga: float; parameter to steer the similarity
        
    Returns:
        float: similarity according to the metric
    """
    A:np.ndarray = np.array(nx.adjacency_matrix(graph).toarray(), dtype=int)
    I:np.ndarray = np.identity(n=graph.number_of_nodes(), dtype=int)
    D:np.ndarray = np.identity(n=graph.number_of_nodes())
    for node in graph.nodes:
        D[node,node] = graph.degree[node]
    L = D - A
    return np.linalg.inv(I + alpha*L)[source, target]

"""Quasi-local similarity metrics""" 
    
def similarity_local_path_index(graph: nx.Graph, source: int, target: int, alpha: float, n:int) -> float:
    """
    Local Path Index (18): Interpolation between common neighbors and Katz Index.
    
    `S = A^2 + a * A^3 + a^2 * A^4 ... `
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        alpha: float; between 0 and 1, decay factor
        n: int; max path distance
        
    Returns:
        float: similarity according to the metric
    """
    def get_number_of_paths_xy(distance) -> int:
        # Compute number of paths of length `distance`
        n_paths_xy = 0
        stack = [(source, 0)]
        while stack:
            node, depth = stack.pop()
            if depth == distance and node == target:
                n_paths_xy += 1
                continue
            elif depth == distance:
                continue
            elif node == target:
                continue
            neighbors = graph.adj[node]
            stack += [(neighbor, depth+1) for neighbor in neighbors]
        return n_paths_xy
    # Iterate over the allowed path lengths and compute the scaled number of paths
    # Then sum all of these values
    similarity_score = 0
    for i in range(2,n+1):
        c = i - 2
        n_paths_xy = get_number_of_paths_xy(i)
        similarity_score += alpha ** c * n_paths_xy
    return similarity_score
    
def similarity_local_random_walk(graph: nx.Graph, source: int, target: int, t:int) -> float:
    """
    Local Random Walk (19): Scored based on the probabilities that a random walker would arrive at the other node after
    t timesteps.
    
    `S = q_x * pi_xy(t) + q_y * pi_yx(t)`
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        t: int; timesteps for the random walk
        
    Returns:
        float: similarity according to the metric
    """
    # Initialize starting densities
    N = graph.number_of_nodes()
    E = graph.number_of_edges()
    pi_source = np.zeros(shape=N)
    pi_source[source] = 1
    q_source = graph.degree[source] / E
    pi_target = np.zeros(shape=N)
    pi_target[target] = 1
    q_target = graph.degree[target] / E
    # Compute transition matrix (P)
    transition_matrix = np.zeros(shape=(N,N),dtype=float)
    for node in graph.nodes:
        k_x = graph.degree[node]
        for neighbor in graph.adj[node]:
            transition_matrix[node,neighbor] = 1/k_x
    P_T = transition_matrix.T
    # Do `t` timesteps of multiplying the transition matrix with the current densities
    for _ in range(t):
        pi_source = P_T @ pi_source
        pi_target = P_T @ pi_target

    return (q_source * pi_source)[target] + (q_target * pi_target)[source]

def similarity_superposed_random_walk(graph: nx.Graph, source: int, target: int, t: int) -> float:
    """
    Superposed Random Walk (20): Sum of LRWs up to time t.
    
    `S = Sum[ LRW(t) ]
    
    Args:
        graph: nx.Graph; Graph on which to compute similarity-based metric
        source: int; Source node
        target: int; Target node
        t: int; number of timesteps
        
    Returns:
        float: similarity according to the metric
    """
    return sum([similarity_local_random_walk(graph, source, target, i) for i in range(t)])