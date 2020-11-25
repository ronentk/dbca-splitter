from pathlib import Path
import string
import networkx as nx
import numpy as np
import ast
from tqdm import tqdm

from dbca.datasets.relational import RelationalSample
from dbca.sample_set import SampleSet
from dbca.storage import SampleStore

CONST_SEED = 1234


class GeneratedStory:
    """
    Helper class to represent story information in structured format
    """
    def __init__(self):
        self.edges = []
        self.sentences = []
        self.graph = nx.DiGraph()
        self.question = ""
        self.answer = False
        
    def process_line(self, sentence: str):
        if "?" in sentence:
            self.add_question(sentence)
        elif "is" in sentence:
            self.add_story_line(sentence)
        else:
            pass
        
    def add_story_line(self, sentence: str):
        sent = sentence.strip("\n")
        story_sentence, edge, all_edges = sent.split(" ; ")
        self.sentences.append(story_sentence)
        edge = ast.literal_eval(edge)
        self.edges.append(edge)
        self.graph.add_edge(edge[0], edge[1])
        
    def add_question(self, sentence: str):
        sent = sentence.strip("\n")
        q, a = sent.split(" ; ")
        self.question = q
        self.answer = ast.literal_eval(a)
        
class RelationalSampleGenerator:
    def __init__(self, num_entities: int, num_edges: int, seed: int):
        self.num_entities = num_entities
        self.num_edges = num_edges
        self.np_random = np.random.RandomState(seed)
        
    def generate(self):
        story_rand_state = np.random.RandomState(self.np_random.randint(0, high=np.iinfo(np.int32).max))
        story, gen_story = make_generated_story(self.num_entities, self.num_edges, 
                                                self.num_edges, np_random_state=story_rand_state)
        yield gen_story.graph
        
        
        


def write_stories_to_file(n, train_path, test_path, seed, k, r, L, H):
    """
    Generates and write stories to a given file
    :param n: number of stories
    :param path: output file path
    :param seed: seed that initializes the pseudo-random number generator
    :param k: number of object types
    :param r: number of relation types (currently just 1 is supported)
    :param L: min number of edges
    :param H: max number of edges
    """
    np_random_state = np.random.RandomState(seed)
    objects = get_objects_list(k, np_random_state)
    train_fpath = Path(train_path)
    train_fpath.parent.mkdir(parents=True, exist_ok=True)
    f = open(train_path, "a")
    for i in range(n):
        f.write(generate_story(objects, L, H, np_random_state)[0])
    f.close()
    test_fpath = Path(test_path)
    test_fpath.parent.mkdir(parents=True, exist_ok=True)
    f = open(test_path, "a")
    for i in range(n):
        f.write(generate_story(objects, L, H, np_random_state)[0])
    f.close()
    
def make_generated_story(k, L, H, np_random_state = None, fixed_scale: bool = False,
                         seed: int = 1234):
    if not np_random_state:
        np_random_state = np.random.RandomState(np.random.randint(np.iinfo(np.int32).max))
    if fixed_scale: # fixed
        objects = get_objects_list(k, None)
    else: # randomize scale
        objects = get_objects_list(k, np_random_state)
        
    story, gen_story = generate_story(objects, L, H, np_random_state, seed, fixed_scale)
    gen_story.total_num_entities =k
    return story, gen_story

def generate_story(objects, L, H, np_random_state, seed, fixed_scale):
    """
    Generates and prints a new story with k object types, r relation types and L < edges < H
    :param objects: a list of (object, weight) tuples
    :param L: min number of edges
    :param H: max number of edges
    :param np_random_state: a random state
    :param: seed: Random integer seed used for initialization (for reproduceability purposes).
    :return: story - a sequence of sentences and a question (string)
    """

    G = nx.DiGraph()
    sentence_index = 1
    story = ""
    gen_story = GeneratedStory()
    gen_story.max_edges = H
    gen_story.min_edges = L
    gen_story.seed = int(seed) # to convert from int64 which gives JSON serialization trouble
    gen_story.fixed_scale = fixed_scale
    
    bigger_sentence = "{} {} is bigger than {} ; {} ; {}\n"
    smaller_sentence = "{} {} is smaller than {} ; {} ; {}\n"
    question_sentence = "{} Is {} bigger than {}? ; {}\n"
    edge_sentence_mapping = {}

    while True:
        if len(G.edges) < H:
            curr_num_of_edges = len(G.edges)
            idx = np_random_state.choice(len(objects), size=2, replace=False)
            (h, t) = (objects[idx[0]], objects[idx[1]])
            if h[1] > t[1]:
                G.add_edge(t[0], h[0])
                if curr_num_of_edges < len(G.edges):
                    sentence = bigger_sentence.format(sentence_index, h[0], t[0], (t[0], h[0]), G.edges)
                    story += sentence
                    gen_story.process_line(sentence)
                    edge_sentence_mapping[(t[0], h[0])] = sentence_index
                    sentence_index += 1
            else:
                G.add_edge(h[0], t[0])
                if curr_num_of_edges < len(G.edges):
                    sentence = smaller_sentence.format(sentence_index, h[0], t[0], (h[0], t[0]), G.edges)
                    story += sentence
                    gen_story.process_line(sentence)
                    edge_sentence_mapping[(h[0], t[0])] = sentence_index
                    sentence_index += 1

        if len(G.edges) >= L:
            G_comp = nx.complement(G)
            answer = None

            edge_list = [e for e in G_comp.edges]
            while answer is None and len(edge_list) > 0:
                idx = np_random_state.randint(len(edge_list))
                (h, t) = (edge_list[idx])
                if (t, h) not in G.edges:
                    if nx.has_path(G, t, h):
                        answer = True
                    elif nx.has_path(G, h, t):
                        answer = False
                    if answer is not None:
                        q_sentence = question_sentence.format(sentence_index, h, t, answer)
                        gen_story.process_line(q_sentence)
                        
                        story += q_sentence
                        return story, gen_story
                edge_list.remove((h, t))
            return generate_story(objects, L, H, np_random_state, seed, fixed_scale)


def get_objects_list(n, np_random_state = None):
    """
    Generates a list of (object, weight) tuples of size n
    :param n: list size
    :return: (object, weight) tuples list of size n
    """
    alphabet_string = string.ascii_uppercase
    weights = list(range(1, n + 1))
    if np_random_state:
        np_random_state.shuffle(weights)
    letters = [f'e{i}' for i,c in  enumerate(list(alphabet_string)[0: n])]
    return list(zip(letters, weights))
