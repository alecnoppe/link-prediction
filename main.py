import networkx as nx
from src.link_prediction.HierarchicalStructureModel import *


if __name__ == "__main__":
    N=5
    G:nx.Graph = nx.erdos_renyi_graph(N, 0.9)
    D = HSM().create_dendrogram(G)
    Gp = HSM().add_top_k_links(G, 3)
    print(len(G.edges), len(Gp.edges))
    # print(list(G.nodes))
    # D = Dendrogram(list(range(0,N)), G)
    # l0 = D.leaves[0]
    # l1 = D.leaves[1]
    # r0 = D.internal_nodes[0]
    # print(r0.num_crossing_edges(G))
    # D.update_probabilities()
    # print(D.compute_probability_table())
    # D.plot()
    # print("Likelihood: ", D.compute_likelihood())
    # print("Log-Likelihood: ", D.compute_log_likelihood())
    # root = [internal_node for internal_node in D.internal_nodes if internal_node.parent == None][0]
    # print(root.probability)
    # for node in D.leaves:
    #     print(f"Leaf {node.label} has path to root:", end=" ")
    #     while node.parent != None:
    #         node = node.parent
    #     print(node == root)
        
    # queue = [root]
    # visit_count =  {node: 0 for node in [] + D.internal_nodes + D.leaves}
    # while queue:
    #     current = queue.pop(0)
    #     visit_count[current] += 1
    #     if current.is_leaf:
    #         print(f"Found path from root to {current.label}")
    #     else:
    #         queue += current.children
            
    # for node in visit_count.keys():
    #     print(node.label, "->", visit_count[node])
