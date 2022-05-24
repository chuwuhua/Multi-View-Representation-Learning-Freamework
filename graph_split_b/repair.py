import numpy as np
import networkx as nx
from graph_split_b import b00_utils
from data_visual import calculatMemory, initMartix
import copy
from tqdm import tqdm,trange


def repairNetwork(nodeLists, G):
    res = []
    count_edges = []
    for i in nodeLists:
        i = list(i)
        i_ = copy.deepcopy(i)
        for j in i:
            nei = list(G.neighbors(j))
            for k in nei:
                if k not in i_:
                    i_.append(k)
        s = G.subgraph(i_)
        res.append(i_)
        count_edges.append(len(s.edges))
    return res, count_edges


def visualizeRepairIncrease(repairResult):
    # 节点增长
    nodesLists = repairResult[0]
    maxNodeNum = []
    minNodeNum = []
    averageNodeNum = []
    for nodesList in nodesLists:
        nodeNum = [len(i) for i in nodesList]
        maxNodeNum.append(max(nodeNum))
        minNodeNum.append(min(nodeNum))
        averageNodeNum.append(int(sum(nodeNum) / len(nodeNum)))
    maxMatrixMemoryUsage = []
    minMatrixMemoryUsage = []
    averageMatrixMemoryUsage = []
    for i in range(len(maxNodeNum)):
        matrix, memoryUsage = initMartix(maxNodeNum[i])
        maxMatrixMemoryUsage.append(memoryUsage / 1024 ** 2)
        matrix, memoryUsage = initMartix(minNodeNum[i])
        minMatrixMemoryUsage.append(memoryUsage / 1024 ** 2)
        matrix, memoryUsage = initMartix(averageNodeNum[i])
        averageMatrixMemoryUsage.append(memoryUsage / 1024 ** 2)
    # print('maxNodeNum:', maxNodeNum)  # maxNodeNum: [5288, 5373, 5475, 5602, 5743, 5909, 6093, 6287, 6481, 6710]
    # [5288, 5373, 5475, 5602, 5743, 5909, 6093, 6287, 6481, 6710, 7165, 7669, 8239, 8836, 9397, 9992, 10671, 11381,
    #  12402, 13495, 14663, 15893, 17172, 18483, 19877, 21349, 22905, 24517, 26218, 27969]
    # print('matrixMemoryUsage(MB):', matrixMemoryUsage)
    # matrixMemoryUsage(MB): [213.34375, 220.2578125, 228.69921875, 239.4296875, 251.63671875, 266.390625, 283.2421875, 301.5625, 320.4609375, 343.5078125]
    # [213.34375, 220.2578125, 228.69921875, 239.4296875, 251.63671875, 266.390625, 283.2421875, 301.5625, 320.4609375,
    #  343.5078125, 391.67578125, 448.71484375, 517.89453125, 595.66796875, 673.70703125, 761.72265625, 868.76171875,
    #  988.21875, 1173.4765625, 1389.4296875, 1640.35546875, 1927.08984375, 2249.73828125, 2606.37109375, 3014.33984375,
    #  3477.32421875, 4002.67578125, 4561.90234375, 5244.3359375, 5968.22265625]

    print('maxNodeNum',
          maxNodeNum)  # maxNodeNum [5288, 5373, 5475, 5602, 5743, 5909, 6093, 6287, 6481, 6710, 7165, 7669, 8239, 8836, 9397, 9992, 10671, 11381, 12402, 13495, 14663, 15893, 17172, 18483, 19877, 21349, 22905, 24517, 26218, 27969]

    print('minNodeNum',
          minNodeNum)  # minNodeNum [107, 115, 120, 124, 131, 140, 144, 148, 152, 157, 164, 171, 178, 186, 195, 202, 211, 220, 228, 235, 243, 250, 257, 264, 271, 278, 286, 296, 308, 320]

    print('averageNodeNum',
          averageNodeNum)  # averageNodeNum [1775, 1828, 1897, 1982, 2083, 2199, 2331, 2480, 2644, 2823, 3017, 3224, 3445, 3677, 3921, 4177, 4444, 4721, 5009, 5310, 5623, 5948, 6283, 6633, 6998, 7374, 7763, 8166, 8584, 9015]

    print('maxMatrixMemoryUsage',
          maxMatrixMemoryUsage)  # maxMatrixMemoryUsage [213.34375, 220.2578125, 228.69921875, 239.4296875, 251.63671875, 266.390625, 283.2421875, 301.5625, 320.4609375, 343.5078125, 391.67578125, 448.71484375, 517.89453125, 595.66796875, 673.70703125, 761.72265625, 868.76171875, 988.21484375, 1173.4765625, 1389.4296875, 1640.3515625, 1927.08984375, 2249.73828125, 2606.3671875, 3014.33984375, 3477.32421875, 4002.68359375, 4585.90234375, 5244.3203125, 5968.21484375]

    print('minMatrixMemoryUsage',
          minMatrixMemoryUsage)  # minMatrixMemoryUsage [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0078125, 0.33984375, 0.03125, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.03125, 0.03515625, 0.04296875, 0.05859375, 0.05859375]

    print('averageMatrixMemoryUsage',
          averageMatrixMemoryUsage)  # averageMatrixMemoryUsage [24.0390625, 25.49609375, 27.45703125, 29.97265625, 33.10546875, 36.89453125, 41.45703125, 46.92578125, 53.3359375, 60.8046875, 69.4453125, 79.3046875, 90.546875, 103.15234375, 117.296875, 133.11328125, 150.67578125, 170.04296875, 191.42578125, 215.12109375, 241.23046875, 269.921875, 301.1796875, 335.671875, 373.62890625, 414.85546875, 459.78125, 508.7578125, 562.17578125, 620.04296875]


if __name__ == '__main__':
    file_path = './graph_splited/sub_nodes.npy'
    nodeLists = np.load(file_path, allow_pickle=True)
    d = b00_utils.load_pickle('./data/dict_processed.pkl')
    G = b00_utils.create_graph(d)
    node = []
    node.append(nodeLists)
    t = trange(30)
    for _ in t:
        t.set_description('repair {}:'.format(_))
        nodes, edgeNum = repairNetwork(nodeLists, G)
        node.append(nodes)
        nodeLists = nodes
    np.save('./graph_splited/repairResult_30.npy', node)
    # repairResult = np.load('graph_splited/repairResult_30.npy', allow_pickle=True)
    # visualizeRepairIncrease(repairResult)
    # initMartix(27969)
