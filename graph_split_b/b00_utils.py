'''
some tools here
'''

import pickle
import networkx as nx

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()

def create_graph(d):
    G = nx.from_dict_of_lists(d)
    return G