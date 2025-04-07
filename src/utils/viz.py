import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(graph: nx.Graph):
    """Plots a graph"""
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos=pos, node_color="#55D6BE", linewidths=2, edgecolors="black", node_size=500) # turquoise
    nx.draw_networkx_edges(graph, pos=pos, width=2, edge_color="#7D5BA6")
    nx.draw_networkx_labels(graph, pos=pos, font_size=10)
    plt.show()
    
    
def plot_predicted_graph(graph: nx.Graph, new_graph: nx.Graph):
    """Plots the new graph, and highlights the added edges"""
    pos = nx.spring_layout(new_graph)
    new_edges = [edge for edge in new_graph.edges if not edge in graph.edges]
    nx.draw_networkx_nodes(new_graph, pos=pos, node_color="#55D6BE", linewidths=2, edgecolors="black", node_size=500) # turquoise
    nx.draw_networkx_edges(new_graph, edgelist=graph.edges, pos=pos, width=2, edge_color="#7D5BA6") # purple
    nx.draw_networkx_edges(new_graph, edgelist=new_edges, pos=pos, width=2, edge_color="#FC6471", style="--") # pink
    nx.draw_networkx_labels(new_graph, pos=pos, font_size=10)
    plt.show()
