from typing import List
from collections import Counter
import numpy as np
import networkx as nx
import itertools



def flatten_lists(l):
    """
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    """
    return [item for sublist in l for item in sublist]

def are_counters_close(a, b):
    equal_keys = a.keys() == b.keys()
    if not equal_keys:
        return False
    a_vals = [v for k,v in sorted(a.items(), key=lambda x: x[0])]
    b_vals = [v for k,v in sorted(b.items(), key=lambda x: x[0])]
    return np.all(np.isclose(a_vals, b_vals))
    

def are_arrays_close(a, b):
    return np.all(np.isclose(a,b))

def normalize(x: np.ndarray, ord=1):
    """
    Normalize array by given norm. Default = 1
    """
    return x / np.linalg.norm(x, ord=ord)

def remove_non_positive(c: Counter):
    return Counter({k: v for k,v in c.items() if v > 0})
    

def get_all_subgraphs(G: nx.DiGraph) -> List[nx.DiGraph]:
    """
    Return all connected subgraphs that have at least 2 nodes incl. input graph

    """
    all_connected_subgraphs = []
    for nb_nodes in range(2, G.number_of_nodes() + 1):
        for SG in (G.subgraph(selected_nodes) for selected_nodes in itertools.combinations(G, nb_nodes)):
            if nx.is_connected(SG.to_undirected()):
                # print(SG.nodes)
                all_connected_subgraphs.append(SG)
                
    return all_connected_subgraphs


        
# https://stackoverflow.com/questions/24483182/python-split-list-into-n-chunks
def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]