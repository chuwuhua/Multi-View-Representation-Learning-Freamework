# 可视化路网分割前后的大小变化
import os
import sys
import numpy as np
import networkx as nx
import pickle
import psutil


def calculatMemory(func):
    def wrap(*args, **kwargs):
        pid = os.getpid()
        process = psutil.Process(pid)
        memory1 = process.memory_info().rss
        res = func(*args, **kwargs)
        memory2 = process.memory_info().rss
        memory = memory2 - memory1
        # print('memory1:{},memory2:{}'.format(memory1, memory2))
        # print('内存占用为{}B,{}KB,{}MB,{}GB'.format(memory, memory / 1024, memory / 1024 ** 2, memory / 1024 ** 3))
        return res, memory

    return wrap


@calculatMemory
def loadOriginalNetwork():
    path = '/DATA/CGW/RoadEmbedding2/graph_split_b/data/dict_processed.pkl'
    with open(path, 'rb') as f:
        nw = pickle.load(f)
    G = nx.from_dict_of_lists(nw)
    return G


@calculatMemory
def loadSubNetwork(num):
    path = '/DATA/CGW/RoadEmbedding2/graph_split_b/data/subGraph/subNetwork{}.pkl'.format(num)
    with open(path, 'rb') as f:
        nw = pickle.load(f)
    G = nx.from_dict_of_lists(nw)
    return G


@calculatMemory
def loadSubNetwork_path(path, num):
    temp_path = os.path.join(path, 'subNetwork{}.pkl'.format(num))
    with open(temp_path, 'rb') as f:
        nw = pickle.load(f)
    G = nx.from_dict_of_lists(nw)
    return G


def saveSubNetwork_path(G, path, nodeLists):
    for step, nodes in enumerate(nodeLists):
        sub = G.subgraph(nodes)
        temp_path = os.path.join(path, 'subNetwork{}.pkl'.format(step))
        with open(temp_path, 'wb') as f:
            pickle.dump(nx.to_dict_of_lists(sub), f)


@calculatMemory
def initMartix(N):
    matrix = np.ones((N, N))
    return matrix


if __name__ == '__main__':
    # 加载原始路网
    # G,m = loadOriginalNetwork()
    # 'original network 内存占用为831057920B,811580.0KB,792.55859375MB,0.7739830017089844GB'
    #
    # subList = np.load('/DATA/CGW/RoadEmbedding2/graph_split_b/graph_splited/sub_nodes.npy',
    #                   allow_pickle=True).tolist()
    # saveSubNetwork(G,subList)

    # calculate average memory usage
    _,memory = loadSubNetwork_path('/DATA/CGW/RoadEmbedding2/NewExperiment/result/temp/',0)

    memoryUsage = []
    matrixUsage = []
    for i in range(371):
        G, memory = loadSubNetwork(i)
        memoryUsage.append(memory)
        N = len(G.nodes)
        Matrix, memory = initMartix(N)
        matrixUsage.append(memory)

    memory = sum(memoryUsage) / len(memoryUsage)
    print('内存占用为{}B,{}KB,{}MB,{}GB'.format(memory, memory / 1024, memory / 1024 ** 2, memory / 1024 ** 3))
    memory = sum(matrixUsage) / len(matrixUsage)
    print('内存占用为{}B,{}KB,{}MB,{}GB'.format(memory, memory / 1024, memory / 1024 ** 2, memory / 1024 ** 3))
