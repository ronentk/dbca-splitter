# dbca-splitter

Independent implementation of the Distribution-Based Compositionality Assessment (DBCA) method presented in the ICLR2020 paper [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://arxiv.org/abs/1912.09713).

Official [authors repo](https://github.com/google-research/google-research/tree/master/cfq) contains Compositional Freebase Questions (CFQ) dataset generated by DBCA method, but not the actual code to create Maximum Compound Divergence (MCD) splits for arbitrary DAG-structured data, hence this implementation.

See [here](https://ronentk.github.io/post/dbca/) for a blog post walk-through of the paper I wrote up.

## Installation

From repo root, run `pip install -r requirements.txt`.

Tested with Ubuntu 18.04 and Python 3.7.

## Usage

### Generating data
To generate a dataset with desired compositionality settings, you'll need to provide your own sample set with samples represented in the required directed-acyclic-graph (DAG) format. The current implementation just uses [simple toy data](dbca/datasets/relational/relational_sample.py) for testing/research purposes.

### Demo notebook
See [demo notebook](notebooks/visualize_dists.ipynb) for sample usage.

### Split generation settings

From repo root, run `python run_dbca.py --h`  to see the various possible split generation settings (or see source [here](dbca/config.py)).
