from typing import List
import logging
import numpy as np
from tqdm import tqdm
import ray

from dbca.sample import Sample
from dbca.full_sample_set import FullSampleSet
from dbca.split_sample_set import SplitSampleSet
from dbca.dbca_splitter import DBCASplitter
from dbca.storage import SampleStore
from dbca.utils import chunks
from dbca.dist_funcs import chernoff_divergence
from dbca.freq_distribution import FrequencyDistribution
from dbca.config import DBCASplitterConfig

logger = logging.getLogger(__name__)


        
        

class DBCASplitterRay(DBCASplitter):
    """
    Processes a sample set to generate training and test splits according to 
    specified compositionality settings, which are controlled by:
    
    * D_A: target atom divergence
    * D_C: target compound divergence
    
    A compositional generalization setting features high compound divergence 
    and low atom divergence.
    
    """
    def __init__(self, samples: List[Sample], config: DBCASplitterConfig = None):
        """
        """
        super(DBCASplitterRay, self).__init__(samples, config)        
                
        ray.init(num_cpus=self.config.num_processes)
        
        # for future processing of full sample set by ray workers
        self.full_sample_set_id = ray.put(self.full_sample_set)
        
        
    def find_best_sample(self, sample_set_to_update: SplitSampleSet,
                         other_sample_set: SplitSampleSet) -> Sample:
        """
        

        Parameters
        ----------
        sample_set_to_update : SampleSet
            DESCRIPTION.
        other_sample_set : SampleSet
            DESCRIPTION.

        Returns
        -------
        Sample
            DESCRIPTION.

        """
        
        all_scores = []
        logger.info(f"Searching for new sample using {self.config.num_processes} processes...")
        update_split = sample_set_to_update.split_type.value
        other_split = other_sample_set.split_type.value
        all_scores = _peek_sample_ray(self, update_split, other_split)

        
        sorted_scores = sorted(all_scores, key=lambda x: (x[1], x[0]))
        best_id, best_score = sorted_scores[0]
        debug_infos = {"best_id": best_id,
                       "best_score": best_score,
                       "all_scores": all_scores}
        return best_id, debug_infos
        
    
    def add_sample_to_set(self, sample_id: Sample, sample_set: SplitSampleSet):
        """ 
        Add new sample to sample set.
        """
        split = sample_set.split_type
        self.sample_splits[sample_id] = split
        self.unused_sample_ids.remove(sample_id)
        
        sample_set.update(sample_id, self.full_sample_set, inplace=True)
        
    def end_generation(self):
        logger.info(f"Saving to {self.config.save_dir}...")
        self.save_splits(self.config.save_dir_path / "data.json")
        
        logger.info("Shutting down ray...")
        ray.shutdown()
        
            
# using https://docs.ray.io/en/master/auto_examples/plot_hyperparameter.html
def _peek_sample_ray(split_generator: DBCASplitter, split_to_update: str,
                         other_split: str) -> float:
    pool_size = split_generator.config.num_processes
    s_id_list = [sid for sid in split_generator.unused_sample_ids]
    configs = [split_generator.config for i in range(pool_size)]
    if split_to_update == "train":
        update_split_copy = [split_generator.train_set for i in range(pool_size)]
        other_split_copy = [split_generator.test_set for i in range(pool_size)]
    else:
        update_split_copy = [split_generator.test_set for i in range(pool_size)]
        other_split_copy = [split_generator.train_set for i in range(pool_size)]
    
    
    
    
    score_mapping = {}
    remaining_ids = []
    all_scores = []
    logger.info("Submitting ray jobs...")
    for i,batch in tqdm(enumerate(list(chunks(s_id_list, pool_size))), total=pool_size):
        config = configs[i]
        update_split_single = update_split_copy[i]
        other_split_single = other_split_copy[i]
        # time1 = time.perf_counter()
        peek_score_id = peek_ray_sid_list.remote(batch, update_split_single, other_split_single,
                                       config, split_generator.full_sample_set_id)
        # time2 = time.perf_counter()
        # logger.debug(f"Running ray remote on batch {i} in {time2 - time1:0.4f} seconds")
        remaining_ids.append(peek_score_id)
        # score_mapping[peek_score_id] = sample_id
        
    while remaining_ids:
        # Use ray.wait to get the object ref of the first task that completes.
        done_ids, remaining_ids = ray.wait(remaining_ids)
        # There is only one return result by default.
        result_id = done_ids[0]
        
        # sample_id = score_mapping[result_id]
        scores = ray.get(result_id)
        all_scores += scores
        
    

    return all_scores


            


@ray.remote
def peek_ray_sid_list(sample_ids: List[str], sample_set_to_update: SplitSampleSet,
                         other_sample_set: SplitSampleSet, 
                         dbca_config: DBCASplitterConfig, full_sample_set: FullSampleSet) -> float:
    scores = []
    for sample_id in tqdm(sample_ids, total=len(sample_ids)):
        
        scores.append(peek_ray(sample_id, sample_set_to_update,
                            other_sample_set,
                            dbca_config,
                            full_sample_set))
    results = list(zip(sample_ids, scores))
    return results


def peek_ray(sample_id: str, sample_set_to_update: SplitSampleSet,
                         other_sample_set: SplitSampleSet, 
                         dbca_config: DBCASplitterConfig, full_sample_set: FullSampleSet) -> float:
    """ 
    
    """
    a_dist, c_dist = sample_set_to_update.update(sample_id,
                                                    full_sample_set,
                                                    inplace=False)
    if sample_set_to_update.is_train:
        train_a_dist = a_dist
        train_c_dist = c_dist
        test_a_dist = other_sample_set.atom_distribution
        test_c_dist = other_sample_set.compound_distribution
    else:
        test_a_dist = a_dist
        test_c_dist = c_dist
        train_a_dist = other_sample_set.atom_distribution
        train_c_dist = other_sample_set.compound_distribution
    return score(train_a_dist, test_a_dist, train_c_dist, test_c_dist, 
                 dbca_config)

def score(train_a_dist: FrequencyDistribution, 
             test_a_dist: FrequencyDistribution,
             train_c_dist: FrequencyDistribution,
             test_c_dist: FrequencyDistribution,
             dbca_config: DBCASplitterConfig
             ) -> float:
       """
       

       Parameters
       ----------
       train_a_dist : FrequencyDistribution
           DESCRIPTION.
       test_a_dist : FrequencyDistribution
           DESCRIPTION.
       train_c_dist : FrequencyDistribution
           DESCRIPTION.
       test_c_dist : FrequencyDistribution
           DESCRIPTION.

       Returns
       -------
       float
           DESCRIPTION.

       """
       atom_score = chernoff_divergence(train_a_dist, 
                                         test_a_dist, dbca_config.alpha_a)
       if atom_score < dbca_config.D_A:
           atom_score = 0
           
       compound_score = np.abs(chernoff_divergence(train_c_dist,
                                                 test_c_dist, dbca_config.alpha_c) - dbca_config.D_C)
       return compound_score + atom_score
    