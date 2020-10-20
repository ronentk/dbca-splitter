from typing import List, Set, Dict
from functools import partial
from collections import defaultdict
from tqdm import tqdm
import logging

from dbca.sample import Sample
from dbca.base import Compound

logger = logging.getLogger(__name__)

class SampleStore:
    """
    Helper class for handling storage of Samples.
    #TODO resolve redundancy with FullSampleSet.
    """
    def __init__(self, samples: List[Sample]):
        
        # samples keyed by sample id
        self._samples = {}
        
        # set of each sample's atoms keyed by sample_id
        self._sample_atoms = {}
        
        # key each compound by compound sig and then all sample_ids in which 
        # it's present, then by unique id for each occurence per sample (if multiple occs. in
        # single sample)
        self._compounds_by_type = defaultdict(partial(defaultdict, partial(defaultdict, Compound)))
        
        # key each compound by sample ids in which it's present 
        # and then compound id, then by unique id for each occurence per sample
        # (if multiple occs. in single sample)
        self._compounds_by_samp = defaultdict(partial(defaultdict, partial(defaultdict, Compound)))

        self._compounds_by_uid = {}
        
        logger.info("Loading samples into storage...")
        self.load_samples(samples)
        logger.info("Done!")
        
    def load_samples(self, samples: List[Sample]):
        """ 
        Load samples to storage.
        """            
        self._samples.update({s.id : s for s in samples})
        
        for s in tqdm(samples, total=len(samples)):
            self._sample_atoms.update({s.id: s.atoms})
            for i, c in enumerate(s.compounds):
                # get unique id for each compound occurence in sample set
                c_uid = f"{s.id}_{str(c)}_{i}" 
                self._compounds_by_type[str(c)][s.id][c_uid] = c
                self._compounds_by_samp[s.id][str(c)][c_uid] = c
                self._compounds_by_uid[c_uid] = c
        
    @property
    def size(self) -> int:
        return len(self._samples)
    
    @property
    def samples(self) -> List[Sample]:
        """
        Return list of contained samples.
        """
        return list(self._samples.values())
    
    
    @property
    def sample_ids(self) -> List[str]:
        """
        Return list of contained samples.
        """
        return list(self._samples.keys())


    def sample_by_id(self, sample_id: str) -> Sample:
        """
        Return Sample object with id `sample_id`.
        """
        return self._samples.get(sample_id)

    def get_sample_atoms(self, sample_id: str) -> List[str]:
        """
        Return list of atoms for given sample.
        """
        return list(self._sample_atoms[sample_id])

    def get_compound_by_uid(self, compound_uid: str) -> Compound:
        """
        Return Compound object with uid `compound_uid`.
        """
        return self._compounds_by_uid[compound_uid]


    def get_compound_types_by_sample(self, sample_id: str) -> Set[str]:
        """
        Return set of compound types contained in this sample.
        """
        return set(self._compounds_by_samp[sample_id].keys())


    def get_sample_compounds_key_sample(self, sample_id: str) -> Dict:
        """
        Return nested dict of form by c_type -> list[c_uid]
        for all compound types <c_type> in sample <sample_id>
        """
        refs_dict = {}
        for c_type, occs_dict in self._compounds_by_samp[sample_id].items():
            refs_dict[c_type] = list(occs_dict.keys())
        return refs_dict

    
    def get_sample_compounds_key_type(self, sample_id) -> Dict:
        """
        Return nested dict of form by c_type, sample_id -> list[c_uid]
        for all compound types <c_type> in sample <sample_id>
        """
        sample_compound_types = self.get_compound_types_by_sample(sample_id)
        refs_dict = {c_type: {sample_id: list(self._compounds_by_type[c_type][sample_id].keys()) } for 
                        c_type in sample_compound_types}
        return refs_dict
    
    def get_samples_with_compound(self, compound_type: str) -> Set[str]:
        """
        Return list of sample ids with compound `compound_type`.
        """
        return set(self._compounds_by_type.get(compound_type).keys())
