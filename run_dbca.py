"""
Distribution Based Compositionality Assessment (DBCA). Un-official implementation
of DBCA, a method presented by Keysers et. al, (2020) to generate train/test splits with
controllable compositionality.

Usage:
    run_dbca.py SAVE_DIR [--d_c=<d_c> --d_a=<d_a>]
    [--n_train=<n_train> --n_test=<n_test> --pool_size=<pool_size>]
    [--max_compounds=<max_compounds>]
    [--seed=<seed>]
    [--num_processes=<num_processes> --verbose]


Options:
   -h --help     Show this screen.
   --version     Show version.
   --verbose    Verbosity.
   --d_c=<d_c>  Desired compound divergence between 0 and 1.  [default: 1]
   --d_a=<d_a>  Maximal allowed atom divergence between 0 and 1.  [default: 0.02]
   --pool_size=<pool_size>  Full dataset size to generate splits from 1. [default: 1000]
   --n_train=<n_train>  Desired train set size.  [default: 50]
   --n_test=<n_test>   Desired test set size (will be rounded if doesn't divide by train size).  [default: 50]
   --seed=<seed>  Random seed for reproduceability [default: 1234]
   --max_compounds=<max_compounds>  Consider only top max_comp compounds with highest weights. [default: 10000]
   --num_processes=<num_processes>  How many processes to use. Will need Ray library for num_processes > 1. [default: 1]



"""

from docopt import docopt
from pathlib import Path
import logging

from dbca.config import DBCASplitterConfig, setup_logger
from dbca.dbca_splitter import DBCASplitter
from dbca.dbca_splitter_ray import DBCASplitterRay
from dbca.datasets.relational.utils import get_samples

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
    arguments = {k.lower().replace("--", ""): v for k,v in arguments.items()}

    config = DBCASplitterConfig(**arguments)

    logger.info(f"Creating {config.pool_size} samples...")

    # toy sample source
    samples = get_samples(num_samples=config.pool_size, seed=1234, num_entities=10, fixed_scale=False)


    if config.num_processes > 1:
        dbca_splitter = DBCASplitterRay(samples, config)
    else:
        dbca_splitter = DBCASplitter(samples, config)

    dbca_splitter.generate_splits()
