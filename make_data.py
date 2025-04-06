import networkx as nx

G = nx.karate_club_graph()
nx.write_edgelist(G, "data/karate_club.txt", data=False)

G = nx.stochastic_block_model([25, 25], [[0.15, 0.005], [0.005, 0.15]], seed=12345)
nx.write_edgelist(G, "data/stochastic_block_model.txt", data=False)

G = nx.watts_strogatz_graph(100, 10, 0.1, seed=12345)
nx.write_edgelist(G, "data/watts_strogatz.txt", data=False)