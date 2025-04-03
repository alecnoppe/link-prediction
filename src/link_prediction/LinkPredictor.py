from abc import ABC, abstractmethod
import networkx as nx


class LinkPredictor(ABC):
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def predict_link(self, graph: nx.Graph, source: int, target: int) -> bool:
        raise NotImplementedError("Not implemented")
    
    @abstractmethod
    def predict_all_links(self, graph: nx.Graph, spurious: bool=False) -> nx.Graph:
        raise NotImplementedError("Not implemented")
                