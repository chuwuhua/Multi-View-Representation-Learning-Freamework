'''
1、 remove nodes from G where node is not in trips
re_dict 是data1.0嵌入的原始数据，不包括cross的路段
dict_process 所有路段都有车辆经过
'''

from graph_split_b import b00_utils
import numpy as np
import networkx as nx

'''
record:
    G node : 1018421   ->  634966 ( easier to split )
    trip node : 634966
    # 所有路段都有车辆订单经过
'''


def pre_graph_trip(G, trips):
    trips = np.unique(trips)
    G_ = G.subgraph(trips)
    sub = nx.to_dict_of_lists(G_)
    b00_utils.save_pickle('./data/dict_processed.pkl', sub)


if __name__ == '__main__':
    d = b00_utils.load_pickle('./data/re_dicts.pkl')
    G = b00_utils.create_graph(d)
    trips = np.load('../data/trips.npy', allow_pickle=True)
    pre_graph_trip(G, trips)
