from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import log

from src.link_prediction.LinkPredictor import LinkPredictor


class Node:
    def __init__(self, label: int=None, is_leaf=False, parent:'Node'=None, children: List['Node']=None):
        self.label: int = label
        self.is_leaf = is_leaf
        self.parent: Node = parent
        self.children: List[Node] = children if children else []
        self.probability = 0
        
    def get_leaves_left(self) -> List['Node']:
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
        # Lr
        return len(self.get_leaves_left())
    
    def num_leaves_right(self) -> int:
        # Rr
        return len(self.get_leaves_right())
    
    def num_crossing_edges(self, graph: nx.Graph) -> int:
        # Er
        num_crossing_edges = 0
        left_leaves = self.get_leaves_left()
        right_leaves = self.get_leaves_right()
        for l in left_leaves:
            for r in right_leaves:
                if graph.has_edge(l.label,r.label):
                    num_crossing_edges += 1
                    
        return num_crossing_edges
                
    def get_lowest_common_ancestor(self, other_node:'Node') -> 'Node':
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
        parent = self.parent
        if parent.children[0] == self:
            return parent.children[1]
        return parent.children[0]

class Dendrogram:
    RANDOM_INIT_SHUFFLE_STEPS = 50
    def __init__(self, nodes:List[int], graph: nx.Graph):
        # Leaves
        self.leaves: List[Node] = [Node(node, is_leaf=True) for node in nodes]
        self.graph: nx.Graph = graph
        self.internal_nodes: List[Node] = [Node() for _ in range(len(nodes)-1)]
        self.init_internal_nodes()
        
    def init_internal_nodes(self):
        internal_nodes_list: List[Node] = [] + self.internal_nodes
        all_nodes_list: List[Node] = [] + self.internal_nodes + self.leaves
        
        for i, parent in enumerate(internal_nodes_list):
            internal_nodes_list[i].children = [all_nodes_list[2*(i+1)-1], all_nodes_list[2*(i+1)]]
            all_nodes_list[2*(i+1)-1].parent = parent
            all_nodes_list[2*(i+1)].parent = parent
        self.shuffle_internal_nodes()
        self.update_probabilities()
        
    def shuffle_internal_nodes(self):
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
        child.parent = internal_node.parent
        sibling.parent = internal_node
        child_idx = 0 if internal_node.children[0] == child else 1
        sibling_idx = 0 if internal_node.parent.children[1] == internal_node else 1
        internal_node.children[child_idx] = sibling
        internal_node.parent.children[sibling_idx] = child
        self.update_probabilities_after_swap(child, sibling)
        
    def update_probability(self, internal_node):
        Lr = internal_node.num_leaves_left()
        Rr = internal_node.num_leaves_right()
        internal_node.probability = internal_node.num_crossing_edges(self.graph) / (Lr * Rr)
        
    def update_probabilities(self):
        for internal_node in self.internal_nodes:
            self.update_probability(internal_node)
            
    def update_probabilities_after_swap(self, u:Node, v:Node):
        nodes_u_to_root = set()
        while u.parent != None:
            u = u.parent
            nodes_u_to_root.add(u)
        nodes_v_to_root = set()
        while v.parent != None:
            v = v.parent
            nodes_v_to_root.add(v)
            
        nodes_to_update = nodes_u_to_root.union(nodes_v_to_root)
        
        for internal_node in nodes_to_update:
            self.update_probability(internal_node)
            
    def compute_probability_table(self) -> np.ndarray:
        N = len(self.leaves)
        P = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i+1, N):
                P[i, j] = self.leaves[i].get_lowest_common_ancestor(self.leaves[j]).probability
                
        return P + P.T
    
    def compute_likelihood(self) -> float:
        likelihood = 1
        for internal_node in self.internal_nodes:
            p = internal_node.probability
            Lr = internal_node.num_leaves_left()
            Rr = internal_node.num_leaves_right()
            likelihood *= (p**p * (1 - p)**(1-p)) ** (Lr * Rr)
        return likelihood
        
    def compute_log_likelihood(self) -> float:
        log_likelihood = 0
        for internal_node in self.internal_nodes:
            p = internal_node.probability
            Lr = internal_node.num_leaves_left()
            Rr = internal_node.num_leaves_right()
            log_likelihood += Lr * Rr * np.log((p**p * (1 - p)**(1-p)))
        return log_likelihood
        
    def plot(self):
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
    WARMUP_STEPS = 10000
    def __init__(self):
        pass
    
    def create_dendrogram(self, graph: nx.Graph, warmup = True) -> Dendrogram:
        dendrogram = Dendrogram(list(graph.nodes), graph)
        if not warmup:
            return dendrogram
        
        for _ in range(HSM.WARMUP_STEPS):
            self.markov_chain_monte_carlo_step(dendrogram)
            
        return dendrogram
    
    def sample_m_probability_tables_every_n_steps(self, dendrogram: Dendrogram, m: int, n: int) -> np.ndarray:
        assert m > 0 and n > 0, "Only positive m,n allowed"
        pij_samples = np.zeros((m, len(dendrogram.leaves), len(dendrogram.leaves)))
        for i in range(m):
            for _ in range(n):
                self.markov_chain_monte_carlo_step(dendrogram)
            pij_samples[i] = dendrogram.compute_probability_table()
        return pij_samples
    
    def markov_chain_monte_carlo_step(self, dendrogram: Dendrogram):
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
        raise NotImplementedError("Not implemented")
    
    def predict_all_links(self, graph: nx.Graph, spurious: bool=False) -> nx.Graph:
        raise NotImplementedError("Not implemented")
                
    def add_top_k_links(self, graph: nx.Graph, k:int) -> nx.Graph:
        raise NotImplementedError("Not implemented")

    
"""
TODO:
- Do the link prediction thing from paper
- Sampling multiple dendrograms?
- idk see tmrw <3
"""