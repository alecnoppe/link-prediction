from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import log

from src.link_prediction.LinkPredictor import LinkPredictor


class Node:
    def __init__(self, label: int=None, is_leaf=False, parent:'Node'=None, children: List['Node']=None):
        """
        Initialize Node class to be used in the Dendrogram.
        
        Args:
            label: int | None=None; Specify label if the Node is a leaf
            is_leaf: bool=False; Whether the Node is a leaf
            parent: Node=None; The parent Node
            children: List[Node]=[]; List of children Nodes.
        """
        self.label: int = label
        self.is_leaf = is_leaf
        self.parent: Node = parent
        self.children: List[Node] = children if children else []
        self.probability = 0
        
    def get_leaves_left(self) -> List['Node']:
        """
        Traverse left subtree and return the leaves.
        
        Return:
            List[Node]: leaf Nodes
        """
        if not self.children:
            return []
        queue = [self.children[0]]
        left_leaves = []
        while queue:
            current = queue.pop(0)
            if not current.children:
                left_leaves.append(current)
            else:
                queue += current.children
                
        return left_leaves
    
    def get_leaves_right(self) -> List['Node']:
        """
        Traverse right subtree and return the leaves.
        
        Return:
            List[Node]: leaf Nodes
        """
        if not self.children:
            return []
        queue = [self.children[1]]
        right_leaves = []
        while queue:
            current = queue.pop(0)
            if not current.children:
                right_leaves.append(current)
            else:
                queue += current.children
                
        return right_leaves
    
    def num_leaves_left(self) -> int:
        """Count number of leaf nodes in the left subtree"""
        return len(self.get_leaves_left())
    
    def num_leaves_right(self) -> int:
        """Count number of leaf nodes in the right subtree"""
        return len(self.get_leaves_right())
    
    def num_crossing_edges(self, graph: nx.Graph) -> int:
        """
        Count number of edges between nodes u, v where u is in the left subtree and v is in the right subtree.
        """
        num_crossing_edges = 0
        left_leaves = self.get_leaves_left()
        right_leaves = self.get_leaves_right()
        for l in left_leaves:
            for r in right_leaves:
                if graph.has_edge(l.label,r.label):
                    num_crossing_edges += 1
                    
        return num_crossing_edges
                
    def get_lowest_common_ancestor(self, other_node:'Node') -> 'Node':
        """
        Returns the lowest common ancestor of the current Node u, and another Node v. The lowest common ancestor is the
        shared ancestor of u, v with the largest depth.
        
        Args:
            other_node: Node;
        
        Returns:
            Node; lowest common ancestor
        """
        assert self.is_leaf and other_node.is_leaf, "This only makes sense for leaf nodes."
        self_ancestors: List[Node] = []
        current_node = self
        while current_node.parent:
            current_node = current_node.parent
            self_ancestors.append(current_node)
        
        current_node = other_node
        other_ancestors: List[Node] = []
        while current_node.parent:
            current_node = current_node.parent
            other_ancestors.append(current_node)
            
        while True:
            if self_ancestors[0] == other_ancestors[0]:
                break
            if len(self_ancestors) > len(other_ancestors):
                self_ancestors.pop(0)
            else:
                other_ancestors.pop(0)
            
        return self_ancestors[0]

    def get_sibling(self) -> 'Node':
        """
        Get sibling Node: the other child of the parent Node.
        """
        parent = self.parent
        if parent.children[0] == self:
            return parent.children[1]
        return parent.children[0]

class Dendrogram:
    RANDOM_INIT_SHUFFLE_STEPS = 50
    def __init__(self, nodes:List[int], graph: nx.Graph):
        """
        Initialize Dendrogram instance D
        
        Args:
            nodes: List[int]; List of leaf node labels
            graph: nx.Graph; observed graph G0
        """
        self.leaves: List[Node] = [Node(node, is_leaf=True) for node in nodes]
        self.graph: nx.Graph = graph
        self.internal_nodes: List[Node] = [Node() for _ in range(len(nodes)-1)]
        self.init_internal_nodes()
        
    def init_internal_nodes(self):
        """
        Randomly place the internal Nodes such that:
        - Each internal node has exactly two children
        - The Nodes form a Binary Tree
        - All paths from the root terminate in a leaf Node, representing a Node in the observed graph G0
        """
        internal_nodes_list: List[Node] = [] + self.internal_nodes
        all_nodes_list: List[Node] = [] + self.internal_nodes + self.leaves
        
        for i, parent in enumerate(internal_nodes_list):
            internal_nodes_list[i].children = [all_nodes_list[2*(i+1)-1], all_nodes_list[2*(i+1)]]
            all_nodes_list[2*(i+1)-1].parent = parent
            all_nodes_list[2*(i+1)].parent = parent
        self.shuffle_internal_nodes()
        self.update_probabilities()
        
    def shuffle_internal_nodes(self):
        """
        Shuffle the internal Nodes of D, by:
        - Selecting an internal node r UaR
        - Swapping of one the children of r with the sibling of r
        - Repeat `X` times
        """
        for _ in range(Dendrogram.RANDOM_INIT_SHUFFLE_STEPS):
            random_internal_node: Node = np.random.choice(self.internal_nodes)
            if not random_internal_node.parent:
                continue
            sibling: Node = random_internal_node.get_sibling()
            if np.random.rand() > 0.5:
                child: Node = random_internal_node.children[0]
            else:
                child: Node = random_internal_node.children[1]
                
            self.swap_nodes_stu(internal_node=random_internal_node, child=child, sibling=sibling)
        
    def swap_nodes_stu(self, internal_node: Node, child: Node, sibling: Node):
        """
        Swap nodes s/t, u where s is a child of r, and u is a sibling of r.
        
        Args:
            internal_node: Node; internal Node r
            child: Node; child of r
            sibling: Node; sibling of r
        """
        child.parent = internal_node.parent
        sibling.parent = internal_node
        child_idx = 0 if internal_node.children[0] == child else 1
        sibling_idx = 0 if internal_node.parent.children[1] == internal_node else 1
        internal_node.children[child_idx] = sibling
        internal_node.parent.children[sibling_idx] = child
        self.update_probabilities_after_swap(child, sibling)
        
    def update_probability(self, internal_node):
        """
        Update probability of an internal node r
        
        Args:
            internal_node: Node; internal node r
        """
        Lr = internal_node.num_leaves_left()
        Rr = internal_node.num_leaves_right()
        internal_node.probability = internal_node.num_crossing_edges(self.graph) / (Lr * Rr)
        
    def update_probabilities(self):
        """Update all internal node probabilities in D"""
        for internal_node in self.internal_nodes:
            self.update_probability(internal_node)
            
    def update_probabilities_after_swap(self, u:Node, v:Node):
        """
        If the Dendrogram is updated via swapping, we do not need to update all internal nodes. This method
        updates all nodes on a path from the swapped Nodes u, v.
        
        Args:
            u: Node; node 1
            v: Node; node 2 
        """
        nodes_u_to_root = set()
        while u.parent != None:
            u = u.parent
            nodes_u_to_root.add(u)
        nodes_v_to_root = set()
        while v.parent != None:
            v = v.parent
            nodes_v_to_root.add(v)
            
        # NOTE: Could probably do a set difference, but I'm not 100% sure. Union works for sure
        nodes_to_update = nodes_u_to_root.union(nodes_v_to_root) 
        
        for internal_node in nodes_to_update:
            self.update_probability(internal_node)
            
    def compute_probability_table(self) -> np.ndarray:
        """
        Compute the probabilities Pij for possible edges (i,j) between leaf nodes i, j. NOTE: Assumes undirected edges
        Pij is defined as the probability associated with the lowest common ancestor of i and j.
        
        Returns:
            P: np.ndarray; the probabilities Pij
        """
        N = len(self.leaves)
        P = np.zeros((N, N), dtype=float)
        
        for internal_node in self.internal_nodes:
            for i in internal_node.get_leaves_left():
                for j in internal_node.get_leaves_right():
                    P[i.label, j.label] = internal_node.probability
                    P[j.label, i.label] = internal_node.probability
                    
        return P
    
    def compute_likelihood(self) -> float:
        """
        Compute likelihood of D
        
        L(D) = Product[ (p^p * (1-p)^(1-p))^(Lr * Rr) ]
        
        Returns:
            likelihood: float;
        """
        likelihood = 1
        for internal_node in self.internal_nodes:
            p = internal_node.probability
            Lr = internal_node.num_leaves_left()
            Rr = internal_node.num_leaves_right()
            likelihood *= (p**p * (1 - p)**(1-p)) ** (Lr * Rr)
        return likelihood
        
    def compute_log_likelihood(self) -> float:
        """
        Compute log likelihood of D
        
        L(D) = log Product[ (p^p * (1-p)^(1-p))^(Lr * Rr) ]
        
        Returns:
            log_likelihood: float;
        """
        log_likelihood = 0
        for internal_node in self.internal_nodes:
            p = internal_node.probability
            Lr = internal_node.num_leaves_left()
            Rr = internal_node.num_leaves_right()
            log_likelihood += Lr * Rr * np.log((p**p * (1 - p)**(1-p)))
        return log_likelihood
        
    def plot(self):
        """
        Plots the dendrogram.
        
        NOTE: This was a pain to make and does not work perfectly for medium-sized graphs or larger (with leafs >= 15)
            - Mainly useful for debugging
        """
        G = nx.Graph()
        depth_nodes = dict()
        queue = [internal_node for internal_node in self.internal_nodes if internal_node.parent == None]
        current_depth = 0
        node_labels = dict()
        
        while True:
            depth_nodes[current_depth] = queue.copy()
            next_nodes = [None] * (2**(current_depth+1))
            for i in range(len(queue)):
                current = queue.pop(0)
                
                if current != None and current.children:
                    next_nodes[i*2] = current.children[0]
                    next_nodes[i*2+1] = current.children[1]
                    G.add_edge(current, current.children[0])
                    G.add_edge(current, current.children[1])
                    node_labels[current] = round(current.probability, 2)  
                elif current != None:
                    node_labels[current] = current.label
                
            current_depth += 1
            if all([next_node == None for next_node in next_nodes]): break
            queue = next_nodes

        pos={}
        WIDTH = 2 * current_depth
        for depth in depth_nodes.keys():
            offset_x = WIDTH / (2**depth+1) 
            for i, node in enumerate(depth_nodes[depth],1):
                if node != None:
                    pos[node] = (offset_x*i, -depth)
        
        nx.draw_networkx_nodes(G, pos, nodelist=self.internal_nodes, node_shape="s", node_color="white", linewidths=2, edgecolors="black", node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=self.leaves, node_shape="o", node_color="white", linewidths=2, edgecolors="black", node_size=500)
        nx.draw_networkx_edges(G, pos, width=2)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        plt.show()

class HSM(LinkPredictor):
    """
    Implementation of the Hierarchical Structure Model (HSM), introduced in:
    [1] Clauset, A., Moore, C., & Newman, M. E. (2008). Hierarchical structure and the prediction of missing links in networks. Nature, 453(7191), 98-101.
    
    HSM is a maximum-likelihood based method, which uses a MCMC algorithm to sample Dendrograms according to their likelihood.
    Using these sampled Dendrograms, we can infer the probability that two unconnected nodes should be connected.
    """
    WARMUP_STEPS = 10000
    def __init__(self, threshold: float=0.5, m:int = 25, n: int = 100):
        """
        Create instance of HSM link predictor. 
        
        Instructions:
            Specify threshold for methods `predict_link` and `predict_all_links`. Defaults to 0.5
            Specify m (number of Dendrograms to sample) for all prediction methods. Defaults to 25
            Specify n (number of steps of MCMC between Dendrograms) for all prediction methods. Defaults to 100

        Args:
            threshold: float; threshold above which to classify a link as missing
            m: int; number of Dendrograms to sample to determine the probability of a missing link
            n: int; number of MCMC steps between sampled Dendrograms
        """
        self.threshold = threshold
        self.m = m
        self.n = n
    
    def create_dendrogram(self, graph: nx.Graph, warmup = True) -> Dendrogram:
        """
        Create (and warm-up) a Dendrogram based on G0.

        Args:
            graph: nx.Graph; observed graph G0
            warmup: bool, optional; Whether to warm up the Dendrogram with `R` MCMC steps. Defaults to True.

        Returns:
            Dendrogram: (warmed-up) Dendrogram based on G0
        """
        dendrogram = Dendrogram(list(graph.nodes), graph)
        if not warmup:
            return dendrogram
        
        for _ in range(HSM.WARMUP_STEPS):
            self.markov_chain_monte_carlo_step(dendrogram)
            
        return dendrogram
    
    def sample_m_probability_tables_every_n_steps(self, dendrogram: Dendrogram, m: int, n: int) -> np.ndarray:
        """
        Sample `m` Dendrograms with `n` steps between each sample. Returns the corresponding probability tables as a
        3D array.
        
        Args:
            dendrogram: Dendrogram; dendrogram D
            m: int; number of Dendrograms to sample
            n: int; number of steps between each sample
        
        Returns:
            pij_samples: np.ndarray; 3D array (M, |V|, |V|) of probabilities Pij
        """
        assert m > 0 and n > 0, "Only positive m,n allowed"
        pij_samples = np.zeros((m, len(dendrogram.leaves), len(dendrogram.leaves)))
        for i in range(m):
            for _ in range(n):
                self.markov_chain_monte_carlo_step(dendrogram)
            pij_samples[i] = dendrogram.compute_probability_table()
        return pij_samples
    
    def markov_chain_monte_carlo_step(self, dendrogram: Dendrogram):
        """
        Single step of the MCMC algorithm described in [1]. Updates the dendrogram D if D' has higher likelihood.
        
        Args:
            dendrogram: Dendrogram; dendrogram describing G0
        """
        # Compute log likelihood of current dendrogram D
        current_log_likelihood = dendrogram.compute_log_likelihood()
        current_likelihood = dendrogram.compute_likelihood()
        # Select a random internal node r
        random_internal_node: Node = np.random.choice(dendrogram.internal_nodes)
        while random_internal_node.parent == None:
            random_internal_node: Node = np.random.choice(dendrogram.internal_nodes)
        # Select which child to swap with its sibling
        sibling: Node = random_internal_node.get_sibling()
        if np.random.rand() > 0.5:
            child_idx = 0
        else:
            child_idx = 1
        child = random_internal_node.children[child_idx]
        # Create alternate dendrogram D'
        dendrogram.swap_nodes_stu(random_internal_node, child, sibling)
        # Compute log likelihood of alternate dendrogram D'
        next_log_likelihood = dendrogram.compute_log_likelihood()
        next_likelihood = dendrogram.compute_likelihood()
        # Compute delta log likelihood log L(D') - log L(D)
        delta_log_likelihood = next_log_likelihood - current_log_likelihood
        # Accept the alternate dendrogram if delta log likelihood is nonnegative
        # otherwise accept with probability L(D') / L(D)
        if delta_log_likelihood < 0 and np.random.rand() > (next_likelihood/current_likelihood):    
            # REJECT the transition by swapping back to the current dendrogram D
            child: Node = random_internal_node.children[child_idx]
            sibling: Node = random_internal_node.get_sibling()
            dendrogram.swap_nodes_stu(random_internal_node, child, sibling)
        
    def predict_link(self, graph: nx.Graph, source: int, target: int) -> bool:
        """
        Predict whether (u,v) is missing from G0. Ie, P_ij >= t where t is some threshold specified in the __init__ call.
        
        Args:
            graph: nx.Graph; observed graph G0
            source: int; source node u label
            target: int; target node v label
            
        Returns:
            bool; whether (u,v) is probable enough to be classified as belonging to G
        """
        dendrogram: Dendrogram = self.create_dendrogram(graph)
        probability_tables = self.sample_m_probability_tables_every_n_steps(dendrogram, m=self.m, n=self.n)
        mean_probability_table = np.mean(probability_tables, axis=0)
        return mean_probability_table[source, target] >= self.threshold
        
    def predict_all_links(self, graph: nx.Graph, spurious: bool=False) -> nx.Graph:
        """
        Predict whether (u,v) is missing from G0. Ie, P_uv >= t where t is some threshold specified in the __init__ call.
        Evaluates all unconnected (u,v) in G0
        
        Args:
            graph: nx.Graph; observed graph G0
            spurious: bool; whether to predict spurious links as well (NOTE: NOT SUPPORTED)
            
        Returns:
            new_graph: nx.Graph; new graph G' with all predicted missing links added
        """
        new_graph: nx.Graph = nx.Graph(graph)
        dendrogram: Dendrogram = self.create_dendrogram(graph)
        probability_tables = self.sample_m_probability_tables_every_n_steps(dendrogram, m=self.m, n=self.n)
        mean_probability_table = np.mean(probability_tables, axis=0)
        if spurious:
            raise NotImplementedError("No support for spurious link prediction (yet)...")
        missing_links = mean_probability_table >= self.threshold
        for i in range(graph.number_of_nodes()):
            for j in range(i+1, graph.number_of_nodes()):
                if not graph.has_edge(i, j) and missing_links[i,j]:
                    new_graph.add_edge(i, j)
        return new_graph
    
    def add_top_k_links(self, graph: nx.Graph, k:int) -> nx.Graph:
        """
        Predict whether (u,v) is missing from G0. Ie, P_uv >= t where t is some threshold specified in the __init__ call.
        Evaluates all unconnected (u,v) in G0, and selects the top k most likely edges.
        
        Args:
            graph: nx.Graph; observed graph G0
            k: int; how many links to add
            
        Returns:
            new_graph: nx.Graph; new graph G' with all top-k predicted missing links added
        """
        new_graph: nx.Graph = nx.Graph(graph)
        dendrogram: Dendrogram = self.create_dendrogram(graph)
        probability_tables = self.sample_m_probability_tables_every_n_steps(dendrogram, m=self.m, n=self.n)
        mean_probability_table = np.mean(probability_tables, axis=0)
        top_k_links = []
        for i in range(graph.number_of_nodes()):
            for j in range(i+1, graph.number_of_nodes()):
                if not graph.has_edge(i, j) and (len(top_k_links) < k or mean_probability_table[i,j] > top_k_links[k-1][0]):
                    top_k_links.append(
                        (mean_probability_table[i,j], i, j)
                    )
                    top_k_links = sorted(top_k_links, key=lambda x: x[0])[:min(len(top_k_links), k)]
                    
        for link in top_k_links: new_graph.add_edge(link[1], link[2])
        return new_graph
