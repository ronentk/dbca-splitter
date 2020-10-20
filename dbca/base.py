from typing import Set, Tuple, Iterable
import networkx as nx

class DAG:
    """
    Inheriting classes must be directed and a-cyclic graphs
    """
    def __init__(self, G: nx.DiGraph):
        assert(nx.is_directed_acyclic_graph(G)), f"{G.edges()} not DAG"
        self.dag = G
    
    def visualize(self):
        nx.draw_networkx(self.dag, with_labels=True, pos=nx.planar_layout(self.dag))

class Compound(DAG):
    """
    Represents a recurring compound (sub-graph) across the sample set.
    """
    def __init__(self, sG: nx.DiGraph, G: nx.DiGraph, sample_id: str):
        super(Compound, self).__init__(sG)
        self._sG = sG
        self._G = G
        self._sid = sample_id
        
    
    def __repr__(self):
        return self.__repr__() 
    
    def __str__(self):
        raise NotImplementedError()
        
    def __hash__(self):
        return hash(self.__repr__())
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.__hash__() == other.__hash__()
            )

    @property
    def sample_id(self) -> str:
        """
        Return the id of the sample containing this compound.
        """
        return self._sid
    
    def is_super_edge(self, edge: Tuple[str,str]) -> bool:
        """
        Return True if edge contained in compound's supergraph.
    
        """
        source, target = edge
        return (source, target) in self._G.edges()
    
    def super_edges(self) -> Set[Tuple[str,str]]:
        """
        Return the set of super edges of this compound - edges between a node in the compound sub-graph and one
        outside the compound sub-graph.

        Returns
        -------
        Set[Tuple[str,str]]
            Set of super edges.

        """
        # super graphs represented by super edges, or union of incoming, outgoing edges from sub graph.
        in_edges = set([e for e in self._G.in_edges(self._sG.nodes()) if not e in self._sG.edges()])
        out_edges = set([e for e in self._G.out_edges(self._sG.nodes()) if not e in self._sG.edges()])
        return in_edges | out_edges
        
        
            
    
    
    
class Atom:
    """
    Represents a recurring node in samples across the sample set.
    """
    def __init__(self):
        pass
        # TODO
    
