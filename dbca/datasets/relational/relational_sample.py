from typing import List, Set, Tuple, Iterable
import networkx as nx
from collections import defaultdict
from itertools import product

from dbca.base import Compound
from dbca.sample import Sample

class RelationalCompound(Compound):
    """
    Compound representation for RelationalSample. 
    # TODO Currently supporting linear sub-graphs, need some graph to string 
    # linearization for more general sub-graphs.
    """
    def __init__(self, sub_graph: nx.DiGraph, sample_graph: nx.DiGraph = None, 
                 sid: str = None):
        super(RelationalCompound, self).__init__(sub_graph, sample_graph, sid)
        self.G = sub_graph
        self.sample_G = sample_graph if sample_graph else sub_graph

        # create unique ordering for atoms
        self.sorted_atoms = list(nx.algorithms.dag.lexicographical_topological_sort(self.G))
        self._repr = "_".join(self.sorted_atoms)
        
    def __repr__(self):
        return self._repr
    
    def __str__(self):
        return self._repr
    
    def __hash__(self):
        return hash(self.__repr__())
    
    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.__hash__() == other.__hash__()
            )
    
    @classmethod
    def from_edges(cls, edges: Iterable[Tuple[str, str]]):
        g = nx.DiGraph()
        g.add_edges_from(edges)
        return cls(g)
        
        

        

class RelationalSample(Sample):
    """
    Toy sample type, graph representing relations between unique entities.
    """
    def __init__(self, graph: nx.DiGraph, name: str = ""):
        super(RelationalSample, self).__init__(graph, name)
        self.G = graph
        self.compounds_by_type = defaultdict(list)
        
        
    @property
    def atoms(self) -> List[str]:
        """
        Return list of atoms. 
        Assuming atoms are simply strings.

        """
        return list(self.G.nodes())
    
    
    @property
    def compounds(self) -> List[str]:
        """
        For this example, we define compounds as linear sub-graphs (paths) of any length.
        """
        if hasattr(self, "_compounds"):
            return self._compounds
        else:
            self._compounds = []
            for a1, a2 in product(self.atoms, self.atoms):
                new_compounds = [RelationalCompound(self.G.edge_subgraph(p), self.G, self.id) 
                                 for p in list(nx.all_simple_edge_paths(
                                         self.G, source=a1, target=a2)) if p]
                self._compounds += new_compounds
            
            # otherwise we will double count subgraphs
            self._compounds = set(self._compounds)
            for c in self._compounds:
                self.compounds_by_type[str(c)].append(c)
            return self._compounds
        
        
    def get_occurrences(self, compound_type: str) -> Iterable[Compound]:
        """
        Return all occurences of compounds of type `compound_type`.

        """
        return self.compounds_by_type[compound_type]
    
    def compounds_types(self) -> List[str]:
        return list(self.compounds_by_type.keys())
    
                
                
            
        
        
    
    