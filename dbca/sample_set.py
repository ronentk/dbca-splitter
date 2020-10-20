from typing import List
from functools import partial
from collections import Counter, defaultdict
import logging

from dbca.freq_distribution import FrequencyDistribution

logger = logging.getLogger(__name__)

class SampleSet:
    """
    Represents common properties and operations for a collection of samples.
    """
    def __init__(self):
        
        # record relevant information on atoms and compounds
        self.atom_weights_by_sample = defaultdict(Counter)
        self.atom_weights = Counter()
        
        # TODO remove redundant storage
        self.local_compounds_by_type = defaultdict(partial(defaultdict, list))
        self.local_compounds_by_samp = defaultdict(partial(defaultdict, list))
        self.compound_weights_by_type = defaultdict(Counter)
        self.compound_weights_by_sample = defaultdict(Counter)
        self.compound_weights = Counter()
        
        self._atom_distribution = None
        self._compound_distribution = None
           
    
    @property
    def size(self) -> int:
        """
        Return size of sample set.
        """
        return len(self.atom_weights_by_sample)
    
    @property
    def sample_ids(self) -> List[str]:
        """
        Return list of sample_ids in the SampleSet.

        """
        return list(self.atom_weights_by_sample.keys())
    
    @property
    def atom_distribution(self) -> FrequencyDistribution:
        """
        

        Returns
        -------
        FrequencyDistribution
            Current atom distribution.

        """
        if not self._atom_distribution:
            self._atom_distribution = self.get_atom_distribution()
        return self._atom_distribution
    
    @property
    def compound_distribution(self) -> FrequencyDistribution:
        """
        Returns current compound distribution.

        Returns
        -------
        FrequencyDistribution
            Current compound distribution.

        """
        if not self._compound_distribution:
            self._compound_distribution = self.get_compound_distribution()
        return self._compound_distribution

    
    def compound_weight_in_set(self, compound_type: str) -> float:
        """ 
        Return weight of `compound_type` in sample set.
        """
        return self.compound_weights[compound_type]
        
    def compound_weight_in_sample(self, sample_id: str, compound_type: str) -> float:
        """ 
        Return weight of `compound_type` in sample `sample_id`.
        """
        return self.compound_weights_by_sample[sample_id][compound_type]
    
    def sample_compound_weights(self, sample_id: str) -> Counter:
        """
        Return compound weights by type, for sample `sample_id`.

        Args:
            sample_id (str): sample to look up.

        Returns:
            Counter: Compound weights by type for sample `sample_id`.
        """
        return self.compound_weights_by_sample[sample_id]
    
    def sample_atom_counts(self, sample_id: str) -> Counter:
        return self.atom_weights_by_sample[sample_id]
        

    
    def get_atom_distribution(self) -> FrequencyDistribution:
        """
        Re-calculate atom distribution.

        Returns:
            FrequencyDistribution: New atom distribution.
        """
        return FrequencyDistribution(self.atom_weights)
        
    
    def get_compound_distribution(self) -> FrequencyDistribution:
        """
        Re-calculate compound distribution.

        Returns:
            FrequencyDistribution: New compound distribution.
        """
        return FrequencyDistribution(self.compound_weights)
    
