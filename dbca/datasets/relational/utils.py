import networkx as nx
import numpy as np
import pytest
from tqdm import tqdm
import time
import dill

from dbca.storage import SampleStore
from dbca.split_sample_set import SplitSampleSet
from dbca.full_sample_set import FullSampleSet
from dbca.datasets.relational import make_generated_story, RelationalSample

CONST_SEED = 1234

def create_graph():
    edges = [('e0', 'e1'),
         ('e1', 'e2'),
         ('e2', 'e3'),
         ('e2', 'e4'),
         ('e0', 'e5')]
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def get_samples(num_entities: int = 20, num_edges: int = None, num_samples: int = 100, seed: int = CONST_SEED, identical_graphs: bool = False, fixed_scale=False):
    samples = []
    samples = []
    stories_rand_state = np.random.RandomState(seed)
    seeds = stories_rand_state.randint(low=0, high=np.iinfo(np.int32).max, size=num_samples)
    for i in tqdm(range(num_samples), total=num_samples):
        curr_seed = seed if identical_graphs else seeds[i]
        np_random = np.random.RandomState(curr_seed)
        num_edges = num_entities // 2 if not num_edges else num_edges
        story, gen_story = make_generated_story(num_entities, num_edges, num_edges, np_random_state=np_random, fixed_scale=fixed_scale)
        samples.append(RelationalSample( gen_story.graph, f"s_{i}"))
    return samples


def create_sample_store(num_entities: int = 20, num_edges: int = 6, num_samples: int = 100, seed: int = CONST_SEED, identical_graphs: bool = False, fixed_scale=False):
    samples = []
    stories_rand_state = np.random.RandomState(seed)
    seeds = stories_rand_state.randint(low=0, high=np.iinfo(np.int32).max, size=num_samples)
    for i in tqdm(range(num_samples), total=num_samples):
        curr_seed = seed if identical_graphs else seeds[i]
        np_random = np.random.RandomState(curr_seed)
        story, gen_story = make_generated_story(num_entities, num_edges, num_edges, np_random_state=np_random, fixed_scale=fixed_scale)
        samples.append(RelationalSample(gen_story.graph, f"s_{i}"))

    return SampleStore(samples)

def create_full_sample_set(num_entities: int = 20, num_edges: int = 6, num_samples: int = 100, seed: int = CONST_SEED, identical_graphs: bool = False, fixed_scale=False):
    sample_store = create_sample_store(num_samples=num_samples, num_edges = num_edges, seed=seed, num_entities=num_entities, identical_graphs=identical_graphs, fixed_scale=fixed_scale)
    ss_full = FullSampleSet(sample_store)
    return ss_full

def pickle_test(obj):
    tic = time.perf_counter()
    pickled = dill.dumps(obj)
    toc = time.perf_counter()
    print(f"Pickled in {toc - tic:0.4f} seconds")
    obj = dill.loads(pickled)
    toc2 = time.perf_counter()
    print(f"Unpickled in {toc2 - toc:0.4f} seconds")
