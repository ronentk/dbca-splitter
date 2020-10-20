
from typing import Iterable, Set
from shortuuid import uuid
import networkx as nx
from dbca.base import Atom, Compound, DAG

class Sample(DAG):
    """
    Class for representing single samples in the dataset. A sample is a 
    directed, acyclic graph (DAG).
    """
    def __init__(self, G: nx.DiGraph, name: str = ""):
        """
        Creates a new sample.

        Args:
            G (nx.DiGraph): DAG representation of the sample.
            name (str, optional): Optional name, for human-readability.
        """
        super(Sample, self).__init__(G)
        self._name = name
        # create unique ids for sample
        self._uid = uuid()
        self._sid = self._name + "_" + self._uid
    
    
    @property
    def id(self) -> str:
        """
        Return unique id of sample.

        """
        return self._sid
    
    
    @property
    def compounds(self) -> Iterable[Compound]:
        """
        Return all occurrences of all compound types in the sample.

        """
        raise NotImplementedError()
    
    @property
    def atoms(self) -> Iterable[Atom]:
        """
        Return atoms comprising the sample.

        """
        raise NotImplementedError()
        
    def get_occurrences(self, compound_type: str) -> Iterable[Compound]:
        """
        Return all occurences of compounds of type `compound_type` in sample.
        """
        raise NotImplementedError()
        