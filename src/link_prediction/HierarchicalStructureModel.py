from typing import List
import networkx as nx
import numpy as np


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
            self_ancestors.append(current_node)
            current_node = current_node.parent
        
        current_node = other_node
        other_ancestors: List[Node] = []
        while current_node.parent:
            other_ancestors.append(current_node)
            current_node = current_node.parent
            
        while True:
            if self_ancestors[0] == other_ancestors[0]:
                break
            if len(self_ancestors) > len(other_ancestors):
                self_ancestors.pop(0)
            else:
                other_ancestors.pop(0)
            
        return self_ancestors[0]
        

class Dendrogram:
    RANDOM_INIT_SHUFFLE_STEPS = 50
    def __init__(self, nodes, graph):
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
        
    def shuffle_internal_nodes(self):
        pass
    
    def update_probabilities(self):
        for internal_node in self.internal_nodes:
            Lr = internal_node.num_leaves_left()
            Rr = internal_node.num_leaves_right()
            internal_node.probability = internal_node.num_crossing_edges(self.graph) / (Lr * Rr)
            print(internal_node.probability)
        

class HSM:
    def __init__(self):
        pass
    
if __name__ == "__main__":
    G = nx.erdos_renyi_graph(5, 0.9)
    D = Dendrogram([0,1,2,3,4], G)
    l0 = D.leaves[0]
    l1 = D.leaves[1]
    r0 = D.internal_nodes[0]
    print(r0.num_crossing_edges(G))
    D.update_probabilities()
    
"""
TODO:
- Random shuffling with s,t,u system from paper
- MCMC for altering D' and selecting the next state with some probability
- See if D.update_probabilities can be faster
- Do the link prediction thing from paper
- Sampling multiple dendrograms?
- idk see tmrw <3
"""