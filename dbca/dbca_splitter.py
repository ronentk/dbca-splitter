from typing import List, Tuple
import logging
import numpy as np
from tqdm import tqdm
import time
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


from dbca.sample import Sample
from dbca.sample_set import SampleSet
from dbca.full_sample_set import FullSampleSet
from dbca.split_sample_set import SplitSampleSet
from dbca.storage import SampleStore
from dbca.config import DBCASplitterConfig
from dbca.dist_funcs import chernoff_divergence
from dbca.freq_distribution import FrequencyDistribution

logger = logging.getLogger(__name__)





def get_weights_df(sample_set: SampleSet, dist_type: str = "compound") -> pd.DataFrame:
    """
    Return DataFrame of atom/compound weights in a sample set.

    Args:
        sample_set (SampleSet): Sample set from which to extract weights.
        dist_type (str, optional): "atom" or "compound" to designate which weights to extract. Defaults to "compound".

    Returns:
        pd.DataFrame: DataFrame of atom/compound weights in a sample set.
    """
    assert(dist_type in ["compound", "atom"])
    weights = sample_set.compound_weights if dist_type == "compound" else sample_set.atom_weights
    df = pd.DataFrame([{"name": k, "weight": v} for k,v in weights.items()])
    df = df.set_index('name')
    df.sort_values(by="weight", ascending=False, inplace=True)
    return df




class DBCASplitter:
    """
    Processes a sample set to generate training and test splits according to 
    specified compositionality settings, which are controlled by:
    
    * D_A: target atom divergence
    * D_C: target compound divergence
    
    A compositional generalization setting features high compound divergence 
    and low atom divergence.

    For full detail of generation settings, see DBCASplitterConfig.
    
    """
    def __init__(self, samples: List[Sample], config: DBCASplitterConfig = None):
        """
        Create new DBCASplitter.

        Args:
            samples (List[Sample]): Full set of samples to create splits from.
            config (DBCASplitterConfig, optional): Optional settings for split generation.
        """
        self.sample_store = SampleStore(samples)
        self.full_sample_set = FullSampleSet(sample_store=self.sample_store)
        self.sample_splits = {s_id: None for s_id in self.full_sample_set.sample_ids}
        self.unused_sample_ids = set(self.sample_splits.keys())
        
        
        self.config = config if config else DBCASplitterConfig()
        self.logger = logging.getLogger(__name__)
        self.train_set = SplitSampleSet(split="train")
        self.test_set = SplitSampleSet(split="test")
        
        # set seed for reproduceability
        np.random.seed(self.config.seed)
        
    
        
    def init_build(self):
        self.logger.info(f"Creating output directory at {self.config.save_dir}...")
        self.config.save_dir_path.mkdir(parents=True, exist_ok=True)
        self.save_config(self.config.save_dir_path)
        
            
    @property
    def step(self) -> int:
        """ 
        Current step of generation.
        """
        return self.train_set.size + self.test_set.size
    
    @property
    def total_steps(self) -> int:
        """ 
        Return total number of steps needed for generation of splits.
        """
        return self.config.n_train + self.config.n_test
    
    @property
    def steps_left(self) -> int:
        """ 
        Return number of steps left for generation process.
        """
        return self.total_steps - self.step
    
    @property
    def build_done(self) -> bool:
        """ 
        Return True if build complete, False o.w.
        """
        return ((self.train_set.size == self.config.n_train) and 
                          (self.test_set.size == self.config.n_test))
    
    @property
    def target_train_test_ratio(self):
        """ 
        Return target desired ratio between train and test set size.
        """
        return self.config.train_test_ratio
    
    @property
    def curr_train_test_ratio(self):
        """
        Return current ratio between train and test set size. 
        """
        if self.train_set.size == 0:
            return 0
        if self.test_set.size == 0:
            return np.inf
        
        return self.train_set.size / self.test_set.size
    
    @property
    def train_step(self) -> bool:
        """ Return True if current step should add sample to training set and False o.w. (should add to test set)"""
        if (self.train_set.size % self.target_train_test_ratio != 0):
            return True
        elif self.curr_train_test_ratio <= self.target_train_test_ratio:
            return True
        else:
            return False
            
    
    def get_sample(self, sample_id: str) -> Sample:
        """ 
        Return sample with id `sample_id`.
        """
        return self.sample_store.sample_by_id(sample_id)
    
    
    def calc_atom_divergence(self, dist_a: FrequencyDistribution, 
                        dist_b: FrequencyDistribution) -> float:
        return chernoff_divergence(dist_a, dist_b, alpha=self.config.alpha_a)
    
    def calc_compound_divergence(self, dist_a: FrequencyDistribution, 
                        dist_b: FrequencyDistribution) -> float:
        return chernoff_divergence(dist_a, dist_b, alpha=self.config.alpha_c)
    
    
    @property
    def atom_divergence(self)-> float:
        """
        

        Returns
        -------
        float
            DESCRIPTION.

        """
        return self.calc_atom_divergence(self.train_set.atom_distribution,
                                             self.test_set.atom_distribution)
    @property  
    def compound_divergence(self) -> float:
        """
        

        Returns
        -------
        float
            DESCRIPTION.

        """
        return self.calc_compound_divergence(self.train_set.compound_distribution,
                                             self.test_set.compound_distribution)
        
  
    def score(self, train_a_dist: FrequencyDistribution, 
              test_a_dist: FrequencyDistribution,
              train_c_dist: FrequencyDistribution,
              test_c_dist: FrequencyDistribution) -> float:
        """
        Calculate score of input train/test splits.

        Parameters
        ----------
        train_a_dist : FrequencyDistribution
            Train atom distribution.
        test_a_dist : FrequencyDistribution
            Test atom distribution.
        train_c_dist : FrequencyDistribution
            Train compound distribution.
        test_c_dist : FrequencyDistribution
            Test compound distribution.

        Returns
        -------
        float
            Score of input train/test splits (in [0,1]).

        """
        atom_score = self.calc_atom_divergence(train_a_dist, 
                                          test_a_dist)
        # exact atom divergence less important than it being below maximal threshold
        if atom_score < self.config.D_A:
            atom_score = 0
            
        compound_score = np.abs(self.calc_compound_divergence(train_c_dist,
                                                  test_c_dist) - self.config.D_C)
        

        return compound_score + atom_score
        
        
        
        
    
    def find_best_sample(self, sample_set_to_update: SplitSampleSet,
                         other_sample_set: SplitSampleSet) -> str:
        """
        Greedily select the best sample to add to `sample_set_to_update` w.r.t target
        atom and compound divergence measures.

        Parameters
        ----------
        sample_set_to_update : SplitSampleSet
            Sample set to be updated with chosen sample.
        other_sample_set : SplitSampleSet
            The other sample set (not updated).

        Returns
        -------
        str
            id of found sample.

        """
        all_scores = []
        self.logger.debug("Searching for new sample...")
        
        for s_id in self.unused_sample_ids:
            all_scores.append((s_id, self.peek_sample(s_id, 
                                              sample_set_to_update, other_sample_set)))
        sorted_scores = sorted(all_scores, key=lambda x: (x[1], x[0]))
        best_id, best_score = sorted_scores[0]
        debug_infos = {"best_id": best_id,
                       "best_score": best_score,
                       "all_scores": all_scores}
        return best_id, debug_infos

        
    def peek_sample(self, sample_id: str, sample_set_to_update: SplitSampleSet,
                         other_sample_set: SplitSampleSet) -> float:
        """
        Check score for adding sample `sample_id` to `sample_set_to_update` without actually
        making the update (not in-place).

        Args:
            sample_id (str): id of sample to check update for.
            sample_set_to_update (SplitSampleSet): Sample set to be updated with chosen sample.
            other_sample_set (SplitSampleSet): The other sample set (not updated).

        Returns:
            float: Split score if we had added `sample_id` to `sample_set_to_update`
        """
        a_dist, c_dist = sample_set_to_update.update(sample_id, self.full_sample_set, 
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
            
        return self.score(train_a_dist, test_a_dist, train_c_dist, test_c_dist)

    
    def add_sample_to_set(self, sample_id: str, split_sample_set: SplitSampleSet):
        """ 
        Add new sample to sample set (update in-place).
        """
        split = split_sample_set.split_type
        self.sample_splits[sample_id] = split
        self.unused_sample_ids.remove(sample_id)
        
        split_sample_set.update(sample_id, full_sample_set=self.full_sample_set,
                          inplace=True)
        
        
    def save_config(self, save_dir: Path):
        """ 
        Save configuration file as JSON.
        """
        config_path = save_dir / 'config.json'
        config_path.write_text(self.config.to_json())
    
    def save_splits(self, save_path: Path):
        """ 
        Save split information to `save_path`.
        """
        splits = {s_id: split.value for s_id, split in self.sample_splits.items() if split}
        save_data = {"splits": splits,
                     "train_size": self.train_set.size,
                     "test_size": self.test_set.size,
                     "compound_divergence": self.compound_divergence,
                     "atom_divergence": self.atom_divergence
                     }
        json.dump(save_data, save_path.open(mode="w"))
        
    
    def end_iteration(self):
        if self.config.save_progress:
            self.save_splits(Path(self.config.save_dir) / "data.json")
            
    
    def end_generation(self):
        """ 
        End generation process.
        """
        self.logger.info("Finished creating dataset!")
        self.logger.info(f"Train set size: {self.train_set.size}, test set size: {self.test_set.size}")
        self.logger.info(f"D_A: {self.atom_divergence}, D_C: {self.compound_divergence}")
            
        self.logger.info(f"Saving to {self.config.save_dir}...")
        self.save_splits(Path(self.config.save_dir) / "data.json")
    
    def generate_splits(self) -> Tuple[SplitSampleSet, SplitSampleSet]:
        """
        At each iteration, a new sample u is selected such that D_C(V||W) and 
        D_A(V||W) are kept as closely as possible to the desired values.

        Returns
        -------
        SplitSampleSet
            Generated train set split.
        SplitSampleSet
            Generated test set split.

        """
        
        self.init_build()
        
        # add random init to initialize train set
        chosen_sample_id = np.random.choice(list(self.unused_sample_ids))
        self.logger.info(f"Choosing random first sample: {chosen_sample_id}...")
        self.add_sample_to_set(chosen_sample_id, self.train_set)
        
        
        # main split generation loop
        self.logger.info("Starting to create splits...")
        for i in tqdm(range(self.steps_left), total=self.steps_left):
            tic = time.perf_counter()
            if self.train_step:
                best_id, debug_infos = self.find_best_sample(self.train_set, self.test_set)
                self.add_sample_to_set(best_id, self.train_set)
                split = self.train_set.split_type.value
            else:
                best_id, debug_infos = self.find_best_sample(self.test_set, self.train_set)
                self.add_sample_to_set(best_id, self.test_set)
                split = self.test_set.split_type.value
            
            toc = time.perf_counter()
        
            
            self.logger.debug(f"Found new {split} sample ({best_id})! Score: {debug_infos.get('best_score')}, Time: {toc - tic:0.4f}")
            self.logger.debug(f"Train set size: {self.train_set.size}, test set size: {self.test_set.size}")
            self.logger.debug(f"D_A: {self.atom_divergence}, D_C: {self.compound_divergence}")
            
            self.end_iteration()
        
        
        self.end_generation()
        

        self.logger.info("Done!")
        
        
    @classmethod
    def measure_sample_sets(cls, train_set: List[Sample], test_set: List[Sample]):
        """
        Measure atom and compound divergence between two existing sample sets.
    
        Parameters
        ----------
        train_set : List[Sample]
            Train set samples.
        test_set : List[Sample]
            Test set samples.
    
        Returns
        -------
        float:
            atom divergence
        float:
            compound divergence
        DBCASplitter:
            DBCASplitter object containing full split details.
        
        """
        dbca_splitter = cls(train_set + test_set)
        for sample in train_set:
            dbca_splitter.add_sample_to_set(sample.id, dbca_splitter.train_set)
        
        for sample in test_set:
            dbca_splitter.add_sample_to_set(sample.id, dbca_splitter.test_set)
            
        return dbca_splitter.atom_divergence, dbca_splitter.compound_divergence, dbca_splitter
    
    
    def get_merged_weights_df(self, dist_type: str = "compound") -> pd.DataFrame:
        tr_df = get_weights_df(self.train_set, dist_type=dist_type)
        te_df = get_weights_df(self.test_set, dist_type=dist_type)
        com_df = pd.concat((tr_df, te_df), axis=1)
        com_df.columns = ["train", "test"]
        com_df.fillna(0, inplace=True)
        com_df["train_norm"] = com_df.train / com_df.train.sum()
        com_df["test_norm"] = com_df.test / com_df.test.sum()
        return com_df
    
    def visualize_dists(self, dist_type: str = "compound"):
        """
        Plot the train and test atom/compound distributions.

        Parameters
        ----------
        dist_type : str, optional
            "atom" or "compound". The default is "compound".

        Returns
        -------
        Distribution plot.

        """
        assert(dist_type in ["compound", "atom"])
        com_df = self.get_merged_weights_df(dist_type)
    
        indices = np.arange(len(com_df))
        width = 1
        plt.bar(indices, list(com_df.train_norm), width=width, 
                color='cyan', label='Train')
        plt.bar(indices, list(com_df.test_norm),
                width=width, color='orange', alpha=0.5, label='Test')
        
                   
        if dist_type == "compound":
            div = self.compound_divergence
        else:
            div = self.atom_divergence

        plt.title(f"{dist_type} frequencies, divergence = {div:0.3f}")
        
        # Set logscale for y-axis
        plt.yscale('log')

        plt.legend()
        plt.show()

        return plt
            
        
