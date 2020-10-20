
from typing import List
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from collections import Counter
import numpy as np
import networkx as nx
import itertools
import logging
from ast import literal_eval

logger = logging.getLogger(__name__)

def setup_logger(default_level=logging.INFO):
    logger.info(f"Setting logging level to {default_level}")
    logging.basicConfig(level=default_level)
    
    
@dataclass_json
@dataclass
class JSONConfig:
    """ 
    Base config file.
    """
    
                
    def to_file(self, path: Path) -> None:
        path.write_text(self.to_json())
        
    @classmethod
    def from_file(cls, path: Path):
        return cls.from_json(path.read_text())
        
def load_config(path: Path, config_type: JSONConfig) -> JSONConfig:
    return config_type.from_json(path.read_text())



    
@dataclass_json
@dataclass
class DBCASplitterConfig(JSONConfig):
    """
    Attributes
    ----------
    d_c (float): Desired compound divergence between 0 and 1. [default: 1.0]
    d_a (float): Maximal allowed atom divergence between 0 and 1.  [default: 0.02]
    alpha_a (float): Chernoff co-efficient for atom divergence [default: 0.5]
    alpha_c (float): Chernoff co-efficient for compound divergence [default: 0.1]
    pool_size (int): Full dataset size to generate splits from 1. [default: 1000]
    max_compounds (int): take top max_compounds with highest weights
    n_train (int): Desired train set size. 
    n_test (int): Desired test set size (will be rounded if doesn't divide by train size).
    save_dir (str): Directory to save output dataset to.
    seed (int): Random seed for reproduceability. 
    num_processes (int): How many processes to use. Will need Ray library for `num_processes` > 1. 
    save_progress: bool = False
    verbose (bool): Verbosity level.

    """
    d_c: float = 0.02
    d_a: float = 1.0
    alpha_a: float = 0.5
    alpha_c: float = 0.1
    pool_size: int = 10000
    max_compounds: int = 10000 # take top max_compounds with highest weights
    n_train: int = 1000
    n_test: int = 1000
    save_dir: str = "."
    seed: int = 1234
    num_processes: int = 1
    save_progress: bool = False
    verbose: bool = False
    
    def __post_init__(self):
        # type checking hack based on https://stackoverflow.com/questions/58992252/how-to-enforce-dataclass-fields-types
        for (name, field_type) in self.__annotations__.items():
            if not isinstance(self.__dict__[name], field_type):
                current_type = type(self.__dict__[name])
                self.__dict__[name] = literal_eval(self.__dict__[name])
                
        self._ratio = np.round(self.n_train / self.n_test)
        # fix train / test ratio to be whole number
        if self.n_train % self.n_test != 0:
            logger.warning(f"train/test set ratio = {self.n_train / self.n_test}, rounding to {self._ratio}")
            self.n_test = int(self.n_train // self._ratio)
                        
            
        # randomize seed if not defined
        if not self.seed:
            self.seed = np.random.randint(0, high=np.iinfo(np.int32).max)

        # setup logging
        if self.verbose:
            setup_logger(default_level=logging.DEBUG)
        else:
            setup_logger(default_level=logging.INFO)
    
    @property
    def D_C(self) -> float:
        return self.d_c
    
    @property
    def D_A(self) -> float:
        return self.d_a
    
    @property
    def save_dir_path(self) -> Path:
        return Path(self.save_dir)
        
    @property
    def train_test_ratio(self) -> float:
        return self._ratio