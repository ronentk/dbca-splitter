from typing import Iterable, List, Mapping
from collections import Counter
import logging
from tqdm import tqdm

from dbca.base import Compound
from dbca.sample_set import SampleSet
from dbca.storage import SampleStore

logger = logging.getLogger(__name__)


class FullSampleSet(SampleSet):
    """
    Represents the full pool of Samples from which the train and test splits will be created.
    """
    def __init__(self, sample_store: SampleStore, top_n_compounds: int = 100000):
        """[summary]

        Args:
            sample_store (SampleStore): Store containing sample objects.
            top_n_compounds (int, optional): Number of top weighted compounds to track, the rest will have weight 0. Defaults to 100000.
        """
        super(FullSampleSet, self).__init__()
        # TODO merge sample store with FullSampleSet
        # 
        self._sample_store = sample_store
        self.top_n_compounds = top_n_compounds
                
        logger.info(f"Loading {sample_store.size} samples...")
        self.load_samples(sample_store.sample_ids)
        filtered = self.filter_compounds(top_n=top_n_compounds) # keeping for debugging purposes
        total_compounds = len(filtered) + len(self.compound_weights)
        logger.info(f"Keeping {min(top_n_compounds, len(self.compound_weights))}/{total_compounds}"
                    " highest weighted compounds.")
        self._atom_distribution = self.get_atom_distribution()
        self._compound_distribution = self.get_compound_distribution()
            
  
    def get_compounds_by_uids(self, c_uids: List[str]) -> List[Compound]:
        """ 
        Retrieve Compound objects keyed by compound unique ids.
        """
        return [self._sample_store.get_compound_by_uid(c_uid) for c_uid in c_uids]
        


    def _add_sample(self, sample_id: str):
        """
        Add sample with id `sample_id` from storage to set and update atom weights
        based on sample atoms. Compound weights not computed here, only after all samples are loaded.
        """

        # update atom weight records based on sample atoms
        sample_atom_weights = Counter(self._sample_store.get_sample_atoms(sample_id))
        self.atom_weights_by_sample[sample_id] = sample_atom_weights
        self.atom_weights += sample_atom_weights

        # update compound records - weight not computed here, only after all samples are loaded
        sample_c_refs_dict = self._sample_store.get_sample_compounds_key_sample(sample_id)
        self.local_compounds_by_samp[sample_id] = sample_c_refs_dict
        by_compound_types = self._sample_store.get_sample_compounds_key_type(sample_id)
        
        updated_c_types = list(sample_c_refs_dict.keys())
        
        for compound_type in updated_c_types:
            self.local_compounds_by_type[compound_type].update(by_compound_types[compound_type])
            
        

    
    def load_samples(self, sample_ids: List[str]):
        """
        Load samples by id and compute atom and compound weights for full sample set.

        Parameters
        ----------
        List[str] :
            Sample ids to load.

        """
        # compute all atom weights and record all compounds.
        for sample_id in tqdm(sample_ids, total=len(sample_ids)):
            self._add_sample(sample_id)
            
        # compute weight for all compound types
        num_compounds = len(self.local_compounds_by_type)
        logger.info(f"Computing weight for all {num_compounds} compounds...")
        # TODO can parallelize this
        for compound_type in tqdm(self.local_compounds_by_type.keys(),
                                  total=num_compounds):
            self.compound_weights[compound_type] = self.calc_compound_weight_in_sample_set(compound_type)
    
    

    
    def calc_compound_weight_in_sample_set(self, compound_type: str) -> float:        
        """
        
        Calculate the weight of compound `compound_type` in sample set, by summing over weights of compound in
        all samlpes in which it is present.
        """
        weight = 0
        samples_with_compound = self.local_compounds_by_type.get(compound_type)

        for sample_id in samples_with_compound:
            sample_weight = self.calc_compound_weight_in_sample(compound_type, sample_id)
            self.compound_weights_by_type[compound_type][sample_id] = sample_weight
            self.compound_weights_by_sample[sample_id][compound_type] = sample_weight
            weight += sample_weight
            
        return weight
    
            
    def get_c_uids_by_type(self, compound_type: str) -> List[str]:
        """
        Get compound uids for all compounds of type `compound_type`. The c_uid keys the actual
        Compound object.
        """
        c_uids = []
        for sample_c_uids in self.local_compounds_by_type[compound_type].values():
            c_uids += sample_c_uids
        return c_uids

    def calc_max_occur_supergraph_prob(self, c_uid: str) -> float:
        """ 
        For a given compound occurrence `c_uid`, find the supergraph co-occuring
        with it the most often across all samples, and return the co-occurrence probability.
        The higher this probability, the less interesting the occurrence.
        """
        compound = self._sample_store.get_compound_by_uid(c_uid)
        # get all occurences for compound of this type
        occs_c_uids = self.get_c_uids_by_type(str(compound))
        occs_compounds = self.get_compounds_by_uids(occs_c_uids)
        max_co_occur_prob = 0.0

        # to find maximal co-occurring super-graph, we can just consider all "super-edges" w.r.t
        # the compound.
        for edge in compound.super_edges():
            co_occur_prob = sum([c.is_super_edge(edge) for c in occs_compounds]) / len(occs_compounds)
            if co_occur_prob > max_co_occur_prob:
                max_co_occur_prob = co_occur_prob
        return max_co_occur_prob
            
    
    
    def calc_compound_weight_in_sample(self, compound_type: str, sample_id: str) -> float:
        """
        Calculate the weight of compound of type `compound_type` in `sample_id`, taking weight of the occurrence
        with maximal weight if there exist multiple occurrences.

        The weight of the occurrence is the complement of the maximum empirical probability from 
        `calc_max_occur_supergraph_prob`.

        Args:
            compound_type (str): Compound type to calcualte weight for.
            sample_id (str): id of sample to consider.

        Returns:
            float: Compound weight in sample.
        """
        sample_compound_occs = self.local_compounds_by_samp[sample_id][compound_type]
        max_weight = 0.0
        for c_uid in sample_compound_occs:
            # taking complement of maximum co-occurrence probability
            weight = (1 - self.calc_max_occur_supergraph_prob(c_uid))
            max_weight = weight if weight > max_weight else max_weight
        
        return max_weight
    
    
    def pop_compound(self, compound_type: str) -> Mapping:
        """ 
        Remove compound of type `compound_type` from records, and set its weight to 0.
        """
        
        # remove from all counters
        popped_compound = self.compound_weights_by_type[compound_type]
        
        for sample_id in popped_compound.keys():
            self.compound_weights_by_sample[sample_id].pop(compound_type)
        
        self.compound_weights.pop(compound_type)
        
        return popped_compound
    
    def filter_compounds(self, top_n: int = 10000) -> Mapping:
        """
        Keep only top_n compound types with highest weights.

        Parameters
        ----------
        top_n : int, optional
            Number of top weight compound types to keep. The default is 10000.

        Returns
        -------
        Mapping
            Weights of removed compounds.

        """
        
        sorted_counts = sorted(self.compound_weights.items(), key=lambda x: x[1], reverse=True)
        compounds_to_remove = sorted_counts[top_n:]
        for c, count in compounds_to_remove:
            self.pop_compound(c)
        return dict(compounds_to_remove)

    
     
    
    

    
