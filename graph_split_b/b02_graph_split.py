'''
greedy_modularity_communities:
Find communities in graph using Clauset-Newman-Moore greedy modularity maximization. This method currently supports the
Graph class and does not consider edge weights.

Greedy modularity maximization begins with each node in its own community and joins the pair of communities that most
increases modularity until no such pair exists.

Parameters

    G (NetworkX graph)
Returns

Return type

    Yields sets of nodes, one for each community.

References

1 M. E. J Newman ‘Networks: An Introduction’, page 224 Oxford University Press 2011.
2 Clauset, A., Newman, M. E., & Moore, C. “Finding community structure in very large networks.” Physical Review E 70(6), 2004.



append:
    it is impossible to calculate hmm matrix for too large graph
    just split once
'''

import networkx as nx
from graph_split_b import b00_utils
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import copy


def graph_cut(G):
    r = greedy_modularity_communities(G)
    c = list(r)
    return c


def complete_neighbors(G, node):
    neighbors = []
    for n in node:
        neighbor = list(nx.neighbors(G,n))
        for i in neighbor:
            if i not in node:
                neighbors.append(i)
    return neighbors + node


if __name__ == '__main__':
    '''
    split graph to subgraph
    re_dicts.pkl: whole graph with dict 
    dict_processed.pkl:part graph with vehicles passing by
    sub_nodes.npy: subgraph nodes , two dimension array
    '''
    data = b00_utils.load_pickle('./data/dict_processed.pkl')
    G = b00_utils.create_graph(data)
    # greedy_modularity_communities cut the whole
    nodes = graph_cut(G)
    nodes = list(nodes)
    print(len(nodes))
    np.save('./graph_splited/sub_nodes.npy',np.array(nodes))
    # complete nodes complete neighborhoods
    for i in range(len(nodes)):
        nodes[i] = complete_neighbors(G, list(nodes[i]))
    np.save('./graph_splited/sub_nodes_neighbors.npy',np.array(nodes))