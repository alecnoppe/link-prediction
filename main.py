import networkx as nx
import argparse

from src.link_prediction.HierarchicalStructureModel import *
from src.link_prediction.StochasticBlockModel import *
from src.utils.viz import *
from src.link_prediction.SimilarityPredictor import SimilarityPredictor
from src.link_prediction.similarity_metrics import *


def main():
    """
    Argparse command line interface for the link prediction algorithms.
    
    Example usage:
    python main.py --graph data/karate_club.txt --algorithm HSM --top_k 10 --plot --output predicted_graph.txt
    python main.py --graph data/stochastic_block_model.txt --algorithm SBM --top_k 5 --plot --output predicted_graph.txt
    python main.py --graph data/watts_strogatz.txt --algorithm Similarity --similarity similarity_common_neighbors --top_k 10 --plot --output predicted_graph.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--graph", required=True, help="Path to the graph file")
    ap.add_argument("-a", "--algorithm", required=True, help="Algorithm to use")
    ap.add_argument("-s", "--similarity", required=False, help="Similarity metric to use")
    ap.add_argument("-t", "--threshold", required=False, type=float, default=0.5, help="Threshold for selecting missing links")
    ap.add_argument("-m", "--mcmc_samples", required=False, type=int, default=25, help="Number of MCMC samples")
    ap.add_argument("-ss", "--mcmc_step_size", required=False, type=float, default=100, help="MCMC step size")
    ap.add_argument("-k", "--top_k", required=False, type=int, default=10, help="Number of links to add")
    ap.add_argument("-p", "--plot", required=False, action="store_true", help="Plot the predicted graph")
    ap.add_argument("-o", "--output", required=False, help="Output file for the predicted graph")
    args = ap.parse_args()
    
    G = nx.read_edgelist(args.graph, nodetype=int)
    if args.algorithm == "HSM":
        model = HSM()
        nG = model.add_top_k_links(G, args.top_k)
    elif args.algorithm == "SBM":
        model = SBM()
        nG = model.add_top_k_links(G, args.top_k)
    elif args.algorithm == "Similarity":
        if args.similarity is None:
            raise ValueError("Similarity metric must be provided for Similarity algorithm")
        similarity_metric = globals()[args.similarity]
        predictor = SimilarityPredictor(args.threshold, similarity_metric)
        nG = predictor.add_top_k_links(G, args.top_k)
    else:
        raise ValueError("Invalid algorithm specified")
    
    if args.plot:
        plot_predicted_graph(G, nG)
        
    if args.output:
        nx.write_edgelist(nG, args.output, data=False)
        print(f"Predicted graph saved to {args.output}")
    
if __name__ == "__main__":
    main()