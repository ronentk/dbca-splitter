
from typing import List, Tuple
from enum import Enum
import logging
from tqdm import tqdm

from dbca.sample_set import SampleSet
from dbca.full_sample_set import FullSampleSet
from dbca.freq_distribution import FrequencyDistribution

logger = logging.getLogger(__name__)

class SplitType(str, Enum):
    TRAIN = "train"
    TEST = "test"

class SplitSampleSet(SampleSet):
    """
    Represents a sample split (train/test). For efficiency/multi-processing,
    doesn't store any actual sample objects, just the references in key form.
    """
    def __init__(self, split: str = "train"):
        super(SplitSampleSet, self).__init__()
        self.split_type = SplitType(split)

    
    @property
    def is_train(self) -> bool:
        return self.split_type == SplitType.TRAIN
    

    def add_sample_from_fullset(self, sample_id: str, full_sample_set: FullSampleSet):
        """
        Adds a new sample `sample_id` to split from `full_sample_set`. 
        From computational considerations, sample not actually added- 
        just extracting the relevant stats.

        Args:
            sample_id (str): id of sample to add.
            full_sample_set (FullSampleSet): Full sample set actually containing the Sample object.
        """
        # update atom stats
        sample_atom_weights = full_sample_set.atom_weights_by_sample[sample_id]
        self.atom_weights_by_sample[sample_id] = sample_atom_weights
        self.atom_weights += sample_atom_weights

        # update compound stats
        sample_compound_weights = full_sample_set.compound_weights_by_sample[sample_id]
        self.compound_weights_by_sample[sample_id] = sample_compound_weights
        
        for c_type in sample_compound_weights.keys():
            self.compound_weights_by_type[c_type][sample_id] = full_sample_set.compound_weights_by_type[c_type][sample_id]
        
        self.compound_weights += sample_compound_weights

    
    def add_samples(self, sample_ids: List[str], full_sample_set: FullSampleSet):
        """
        Adds list of samples by id from `full_sample_set`.

        Args:
            sample_ids (List[str]): ids of sample to add.
            full_sample_set (FullSampleSet): Full sample set actually containing the Sample objects.
        """
        for sample_id in tqdm(sample_ids, total=len(sample_ids)):
            self.add_sample_from_fullset(sample_id, full_sample_set)
        
            
    
    def update(self, sample_id: str, full_sample_set: FullSampleSet, inplace: bool = False) -> Tuple[FrequencyDistribution,
                                                                                            FrequencyDistribution]:
        """
        Add sample `sample_id` from `full_sample_set` to split and return new atom and compound distributions.
        If inplace is False, update won't affect original split.

        Args:
            sample_id (str): id of sample to add.
            full_sample_set (FullSampleSet): Full sample set actually containing the Sample object.
            inplace (bool, optional): If inplace is False, update won't affect original split. If True, 
            new sample will be added to split. Defaults to False.

        Returns:
            FrequencyDistribution: Updated atom distribution after adding new sample.
            FrequencyDistribution]: Updated compound distribution after adding new sample. 
        """
         
        if sample_id in self.atom_weights_by_sample:
            # sample already in set, return current distributions
            return self.atom_distribution, self.compound_distribution
        
        self.add_sample_from_fullset(sample_id, full_sample_set)
        
        # compute new frequency distributions
        new_atom_dist = self.get_atom_distribution()
        new_compound_dist = self.get_compound_distribution()
        
        if inplace:
            # update distributions
            self._atom_distribution = new_atom_dist
            self._compound_distribution = new_compound_dist
            
        # not inplace update- revert changes
        else:
            self.pop_sample(sample_id)
                        
        return new_atom_dist, new_compound_dist
        
    def pop_sample(self, sample_id: str):
        """
        Remove sample `sample_id` from split.

        """
        popped_sample_atom_weights = self.atom_weights_by_sample.pop(sample_id)
        self.atom_weights -= popped_sample_atom_weights
        
        # update counters with removed sample
        popped_sample_compound_weights = self.compound_weights_by_sample.pop(sample_id)
        for c_type in popped_sample_compound_weights.keys():
            self.compound_weights_by_type[c_type].pop(sample_id)
        
        self.compound_weights -= popped_sample_compound_weights
        