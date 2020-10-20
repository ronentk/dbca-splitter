from typing import List, Set
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

from dbca.utils import normalize, are_arrays_close, remove_non_positive, are_counters_close

class FrequencyDistribution:
    """
    Represents a frequency distribution over elements, in un-normalized and normalized form. 
    Allows access by element name as well as vector representation.
    """
    def __init__(self, element_freqs: Counter):
        # remove zero weight elements as they don't affect distribution and shouldn't
        # affect eq operator
        self._element_freqs = remove_non_positive(element_freqs.copy())
        self.el2id = {e: i for (i,e) in enumerate(sorted(self._element_freqs.keys()))}
        self.id2el = {i: e for (i,e) in enumerate(sorted(self._element_freqs.keys()))}
        
        self._unnormalized = np.array([self._element_freqs[self.id2el[i]] for i in range(self.size)])
        self._normalized = normalize(self._unnormalized)
        
    def __eq__(self, o):
        return(
            are_counters_close(self._element_freqs, o._element_freqs) and
            (self.el2id == o.el2id) and
            (self.id2el == o.id2el) and
            are_arrays_close(self._unnormalized, o._unnormalized) and
            are_arrays_close(self._normalized, o._normalized)
            )
    
    @property
    def size(self) -> int:
        return len(self._element_freqs)
    
    @property
    def element_freqs(self):
        return OrderedDict(self._element_freqs.most_common())
    
    def element_prob(self, element: str) -> float:
        """ 
        Return probability of `element` in distribution (0 if doesn't exist).
        """
        if element in self.el2id:
            return self._normalized[self.el2id[element]]
        else:
            return 0
    
    @property
    def elements(self) -> Set[str]:
        """
        Return the set of elements with non-zero probability in this distribution.
        """
        return set(self._element_freqs.keys())
        
    
    @property
    def normalized(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray.
            Normalized probability distribution over elements.

        """
        return self._normalized
        
        
        
    @property    
    def unnormalized(self) -> np.ndarray:
        """
        

        Returns
        -------
        np.ndarray.
            Unnormalized distribution over elements.

        """
        return self._unnormalized
    
