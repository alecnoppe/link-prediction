import networkx as nx
import numpy as np
from typing import List
from math import comb


from src.link_prediction.LinkPredictor import LinkPredictor

class Partition:
    def __init__(self, nodes:List[int], graph:nx.Graph):
        """
        Create instance of the Partition class. A Partition is an assignment of the nodes of an observed graph G0,
        to |V| groups. The initialization is done randomly. 
        
        Args:
            nodes: List[int]; nodes to add to the partition
            graph: nx.Graph; observed graph G0
        """
        self.nodes = nodes
        self.graph = graph
        self.partition = [set() for _ in self.nodes]
        # Randomly assign nodes to groups in the partition
        assignment = np.random.choice(len(self.nodes), len(self.nodes))
        self.node_groups = dict()
        for node_idx, group_idx in enumerate(assignment): 
            self.partition[group_idx].add(node_idx)
            self.node_groups[node_idx] = group_idx

    def links_between_groups(self, alpha:int, beta:int) -> int:
        """
        Count the links between nodes in group a to nodes in group b
        
        Args:
            alpha: int; first group idx
            beta: int; second group idx
            
        Returns:
            int; number of observed links between a, b
        """
        n_observed_links = 0
        for u in self.partition[alpha]: 
            for v in self.partition[beta]:
                n_observed_links += 1 if self.graph.has_edge(u,v) else 0
        return n_observed_links
        
    def H(self) -> float:
        """
        H(P) from [2], essentially the log likelihood of a partition.
        
        Returns:
            float; H(P)
        """
        out = 0
        non_empty_partition_indices = [c for c, p_id in enumerate(self.partition) if len(p_id) > 0]
        for c, alpha in enumerate(non_empty_partition_indices):
            for beta in non_empty_partition_indices[c:]:
                alpha_nodes = len(self.partition[alpha])
                beta_nodes = len(self.partition[beta])
                n_possible_links = alpha_nodes * beta_nodes
                n_observed_links = self.links_between_groups(alpha, beta)
                out += np.log(n_possible_links + 1) + np.log(comb(n_possible_links, n_observed_links))
        return out
    
    def get_group(self, node_idx: int) -> int:
        """
        Get the group for a node.

        Args:
            node_idx: int; node to retrieve group for

        Returns:
            int; group idx
        """
        return self.node_groups[node_idx]

    def link_reliability(self, source:int, target:int) -> float:
        """
        Compute reliability Rij for a link between i, j based on only the current Partition.
        
        Args:
            source: int; source node i
            target: int; target node j
            
        Returns:
            float; reliability Rij
        """
        alpha = self.get_group(source)
        beta = self.get_group(target)
        alpha_nodes = len(self.partition[alpha])
        beta_nodes = len(self.partition[beta])
        n_possible_links = alpha_nodes * beta_nodes
        n_observed_links = self.links_between_groups(alpha, beta)
        return ((n_observed_links + 1)/(n_possible_links + 2)) * np.exp(-self.H())
    
    def compute_reliability_table(self) -> np.ndarray:
        """
        Compute all reliabilities Rij, and return them as a 2D numpy array.
        
        Returns:
            np.ndarray; Rij table
        """
        N = len(self.nodes)
        R = np.zeros(shape=(N,N))
        for source in range(N):
            for target in range(source+1, N):
                R[source,target]=self.link_reliability(source,target)
                
        return R + R.T
    
    def move_node(self, node: int, new_group: int):
        """
        Move a node to a new group.

        Args:
            node: int; node to move
            new_group: int; new group index
        """
        current_group = self.get_group(node)
        self.partition[new_group].add(node)
        self.partition[current_group].remove(node)
        self.node_groups[node] = new_group


class SBM(LinkPredictor):
    """
    Implementation of the Stochastic Block Model Maximum Likelihood method (HSM), introduced in:
    [2]     R. Guimerà, & M. Sales-Pardo, Missing and spurious interactions and the reconstruction of complex networks, Proc. Natl. Acad. Sci. U.S.A. 106 (52) 22073-22078, https://doi.org/10.1073/pnas.0908366106 (2009). 
    
    SBM is a maximum-likelihood based method, which uses a MCMC algorithm to sample block models according to their likelihood.
    Using these sampled block models, we can infer the probability that two unconnected nodes should be connected.
    """
    WARMUP_STEPS = 10000
    def __init__(self, threshold: float=0.5, m:int = 25, n: int = 100):
        """
        Create instance of SBM link predictor. 
        
        Instructions:
            Specify threshold for methods `predict_link` and `predict_all_links`. Defaults to 0.5
            Specify m (number of Partitions to sample) for all prediction methods. Defaults to 25
            Specify n (number of steps of MCMC between sampled Partitions) for all prediction methods. Defaults to 100

        Args:
            threshold: float; threshold above which to classify a link as missing
            m: int; number of Partitions to sample to determine the probability of a missing link
            n: int; number of MCMC steps between sampled Partitions
        """
        self.threshold = threshold
        self.m = m
        self.n = n
    
    def create_partition(self, graph: nx.Graph, warmup = True) -> Partition:
        """
        Create (and warm-up) a Partition based on G0.

        Args:
            graph: nx.Graph; observed graph G0
            warmup: bool, optional; Whether to warm up the Partition with `R` MCMC steps. Defaults to True.

        Returns:
            Partition: (warmed-up) Partition based on G0
        """
        partition = Partition(list(graph.nodes), graph)
        if not warmup:
            return partition
        
        for _ in range(SBM.WARMUP_STEPS):
            self.markov_chain_monte_carlo_step(partition)
            
        return partition
    
    def sample_reliability_table(self, partition: Partition, m: int, n: int) -> np.ndarray:
        """
        Sample `m` Partitions with `n` steps between each sample. Returns the corresponding reliability table as a
        2D array.
        
        Args:
            partition: Partition; initial Partition P
            m: int; number of Partitions to sample
            n: int; number of steps between each sample
        
        Returns:
            Rij: np.ndarray; 2D array (|V|, |V|) of reliabilities Rij
        """
        assert m > 0 and n > 0, "Only positive m,n allowed"
        Rij = np.zeros((len(partition.nodes), len(partition.nodes)))
        Z = 0
        for _ in range(m):
            for _ in range(n):
                self.markov_chain_monte_carlo_step(partition)
            Rij += partition.compute_reliability_table()
            Z += np.exp(- partition.H())
        return Rij / Z
    
    def markov_chain_monte_carlo_step(self, partition: Partition):
        """
        Single step of the MCMC algorithm described in [2]. Updates the partition P if P' has higher likelihood.
        
        Args:
            partition: Partition; partition describing G0
        """
        # Compute log likelihood of current partition P
        current_log_likelihood = partition.H()
        # Select random node to change to new group in partition P'
        random_node: int = np.random.randint(0, len(partition.nodes))
        current_group: int = partition.get_group(random_node)
        new_group: int = current_group
        while new_group == current_group: new_group = np.random.randint(len(partition.partition))
        partition.move_node(random_node, new_group)
        # Compute log likelihood of alternate partition P'
        next_log_likelihood = partition.H()
        # Compute delta log likelihood log L(P') - log L(P)
        delta_log_likelihood = next_log_likelihood - current_log_likelihood
        # Accept the alternate partition if delta log likelihood is nonnegative
        # otherwise accept with probability exp[ - log L(P') + log L(P)]
        if delta_log_likelihood < 0 and np.random.rand() > (np.exp(-1 * delta_log_likelihood)):    
            # REJECT the transition by swapping back to the current partition P
            partition.move_node(random_node, current_group)
        
    def predict_link(self, graph: nx.Graph, source: int, target: int) -> bool:
        """
        Predict whether (u,v) is missing from G0. Ie, R_ij >= t where t is some threshold specified in the __init__ call.
        
        Args:
            graph: nx.Graph; observed graph G0
            source: int; source node u label
            target: int; target node v label
            
        Returns:
            bool; whether (u,v) is probable enough to be classified as belonging to G
        """
        partition: Partition = self.create_partition(graph)
        reliability_table = self.sample_reliability_table(partition, m=self.m, n=self.n)
        return reliability_table[source, target] >= self.threshold
        
    def predict_all_links(self, graph: nx.Graph, spurious: bool=False) -> nx.Graph:
        """
        Predict whether (u,v) is missing from G0. Ie, R_uv >= t where t is some threshold specified in the __init__ call.
        Evaluates all unconnected (u,v) in G0
        
        Args:
            graph: nx.Graph; observed graph G0
            spurious: bool; whether to predict spurious links as well (NOTE: NOT SUPPORTED)
            
        Returns:
            new_graph: nx.Graph; new graph G' with all predicted missing links added
        """
        new_graph: nx.Graph = nx.Graph(graph)
        partition: Partition = self.create_partition(graph)
        reliability_table = self.sample_reliability_table(partition, m=self.m, n=self.n)
        if spurious:
            raise NotImplementedError("No support for spurious link prediction (yet)...")
        missing_links = reliability_table >= self.threshold
        for i in range(graph.number_of_nodes()):
            for j in range(i+1, graph.number_of_nodes()):
                if not graph.has_edge(i, j) and missing_links[i,j]:
                    new_graph.add_edge(i, j)
        return new_graph
    
    def add_top_k_links(self, graph: nx.Graph, k:int) -> nx.Graph:
        """
        Predict whether (u,v) is missing from G0. Ie, R_uv >= t where t is some threshold specified in the __init__ call.
        Evaluates all unconnected (u,v) in G0, and selects the top k most likely edges.
        
        Args:
            graph: nx.Graph; observed graph G0
            k: int; how many links to add
            
        Returns:
            new_graph: nx.Graph; new graph G' with all top-k predicted missing links added
        """
        new_graph: nx.Graph = nx.Graph(graph)
        partition: Partition = self.create_partition(graph)
        reliability_table = self.sample_reliability_table(partition, m=self.m, n=self.n)
        top_k_links = []
        for i in range(graph.number_of_nodes()):
            for j in range(i+1, graph.number_of_nodes()):
                if not graph.has_edge(i, j) and (len(top_k_links) < k or reliability_table[i,j] > top_k_links[k-1][0]):
                    top_k_links.append(
                        (reliability_table[i,j], i, j)
                    )
                    top_k_links = sorted(top_k_links, key=lambda x: x[0], reverse=True)[:min(len(top_k_links), k)]
                    
        for link in top_k_links: new_graph.add_edge(link[1], link[2])
        return new_graph
