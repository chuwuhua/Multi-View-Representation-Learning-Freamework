# 重新可视化-折线图、螺旋图

"""
x:拓扑距离（1-n)/转移概率(0-1)
y:average cosine similarity
统计所有的结果可能性比较低，但是可以随机选取一部分结果
"""
import networkx as nx
import pickle
from random import choices
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_road_embedding(path_road_embedding):
    road_embeddings = pd.read_csv(path_road_embedding, header=None).values
    road_embeddings = road_embeddings[np.argsort(road_embeddings[:, 0]), :]  # 升序排序
    road_embeddings_pad = np.zeros((int(np.max(road_embeddings[:, 0]) + 1), road_embeddings.shape[1] - 1))
    road_embeddings_pad[road_embeddings[:, 0].astype(np.int)] = road_embeddings[:, 1:]
    return road_embeddings_pad


def loadGraph(weight=False):
    if weight:
        path = r'/DATA/CGW/RoadEmbedding2/data/dict_processed_weight.pkl'
    else:
        path = r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\data\dict_processed.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if weight:
        G = nx.from_dict_of_dicts(data)
    else:
        G = nx.from_dict_of_lists(data)
    return G


def selectNodePairs(G, order):
    """
    随机选取两个路段，寻找最短路径，只要路径长度足够大，即包含了我们需要的所有拓扑距离的路段对
    """
    nodePairs = []
    result = [[] for _ in range(order)]
    markov = [[] for _ in range(order)]
    nodeList = list(G.nodes)
    while len(nodePairs) < 500:
        pair = choices(nodeList, k=2)
        if pair[0] == pair[1]:
            continue
        if pair in nodePairs:
            continue
        path = list(nx.shortest_path(G, pair[0], pair[1]))
        if len(path) < order + 1:
            continue
        nodePairs.append(pair)
        dis = 0
        for step in range(order):
            result[step].append([path[0], path[step + 1]])
            neighbor = list(G.neighbors(path[step]))
            weights = [G.get_edge_data(path[step], i)['weight'] for i in neighbor]
            weight = G.get_edge_data(path[step], path[step+1])['weight']
            if weight==0:
                weight = 1
            dis += sum(weights)/weight
            markov[step].append(dis)
    return result,markov


def calculateTopologyCosine(data, vec):
    cos = []
    for pairs in data:
        cos_similarity = []
        for pair in pairs:
            cos_similarity.append(cosine_similarity(vec[pair[0]].reshape(1, -1), vec[pair[1]].reshape(1, -1))[0][0])
        cos.append(sum(cos_similarity) / len(cos_similarity))
    return cos


def calculateMarkovCosine(G, vec):
    # 0,1,0.05
    nodeList = list(G.nodes)
    result = [[] for i in range(20)]
    for node in tqdm(nodeList):
        neighbor = list(G.neighbors(node))
        weights = [G.get_edge_data(node, i)['weight'] for i in neighbor]
        s = sum(weights)
        markov = [i / s for i in weights]
        cos = [cosine_similarity(vec[node].reshape(1, -1), vec[i].reshape(1, -1))[0][0] for i in neighbor]
        for i in range(len(markov)):
            if markov[i] == 1:
                result[-1].append(cos[i])
                continue
            result[int(markov[i] // 0.05)].append(cos[i])
    result = [sum(i) / len(i) for i in result]
    return result


def visualTopologyCosineWithPlot(data):
    plt.figure(figsize=(5,5))
    x = [i for i in range(1, 21)]
    # plt.plot(x, data[0][0:20], color='#194f97',marker='^', lw=1)
    plt.plot(x, data[0][0:20], color='black',marker='^', lw=1)
    # plt.plot(x, data[1][0:20], color='#bd6b08',marker='x', lw=1)
    plt.plot(x, data[1][0:20], color='black',marker='D', lw=1)
    # plt.plot(x, data[2][0:20], color='#007f54',marker='o', lw=1)
    plt.plot(x, data[2][0:20], color='black',marker='o', lw=1)
    plt.plot(x, data[3][0:20], color='#c82d31',marker='s', lw=2)
    # plt.vlines(x=10,ymin=0,ymax=data[0][9],colors='gray')
    plt.ylabel('cos', {'size': 16})
    plt.xticks([1, 5, 10, 15, 20])
    plt.ylim(0,1)
    plt.xlim(1,21)
    plt.legend(['DeepWalk', 'Node2Vec', 'Road2Vec', 'Trans2Vec'], prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.savefig('/DATA/CGW/RoadEmbedding2/NewExperiment/img/cos_topology.png',dpi=1600)
    plt.show()


def visualTopologyCosineWithPolar(data):
    x = [i for i in range(1, 21)]
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")

    ax.plot([np.arccos(i) for i in data[0][0:20]], x, color='#194f97', lw=1.5)
    ax.plot([np.arccos(i) for i in data[1][0:20]], x, color='#bd6b08', lw=1.5)
    ax.plot([np.arccos(i) for i in data[2][0:20]], x, color='#007f54', lw=1.5)
    ax.plot([np.arccos(i) for i in data[3][0:20]], x, color='#c82d31', lw=1.5)
    # ax.ylabel('cos', {'size': 16})
    ax.set_yticks([10, 15, 20])
    ax.legend(['DeepWalk', 'Node2Vec', 'Road2Vec', 'Trans2Vec'], prop={'size': 14}, loc='lower left')
    ax.tick_params(labelsize=12)

    plt.show()
    pass

def visualTopologyCosineWithPCOLOR(data):
    fig = plt.figure()
    plt.pcolor(data[:,0:20],cmap='Blues')
    fig.tight_layout()
    plt.show()

def visualMarkovCosineWithPlot(data):
    x = [i / 20 for i in range(20)]
    fig = plt.figure(figsize=(5,5))
    # plt.plot(x, data[0][0:20], color='#194f97',marker='^', lw=1)
    plt.plot(x, data[0][0:20], color='black', marker='^', lw=1)
    # plt.plot(x, data[1][0:20], color='#bd6b08',marker='x', lw=1)
    plt.plot(x, data[1][0:20], color='black', marker='D', lw=1)
    # plt.plot(x, data[2][0:20], color='#007f54',marker='o', lw=1)
    plt.plot(x, data[2][0:20], color='black', marker='o', lw=1)
    plt.plot(x, data[3][0:20], color='#c82d31', marker='s', lw=1.5)
    plt.ylabel('cos', {'size': 16})
    plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.legend(['DeepWalk', 'Node2Vec', 'Road2Vec', 'Trans2Vec'], prop={'size': 14})
    plt.tick_params(labelsize=12)
    plt.savefig('/DATA/CGW/RoadEmbedding2/NewExperiment/img/cos_markov.png',dpi=1600)

    plt.show()


def visualMarkovCosineWithPolar(data):
    x = [i / 20 for i in range(20)]
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.plot([np.arccos(i) for i in data[0]], x, color='#194f97', lw=2)
    ax.plot([np.arccos(i) for i in data[1]], x, color='#bd6b08', lw=2)
    ax.plot([np.arccos(i) for i in data[2]], x, color='#007f54', lw=2)
    ax.plot([np.arccos(i) for i in data[3]], x, color='#c82d31', lw=2)
    ax.set_yticks([0.25,0.5,0.75,1])
    ax.legend(['DeepWalk', 'Node2Vec', 'Road2Vec', 'Trans2Vec'], prop={'size': 14}, loc='lower left')
    ax.tick_params(labelsize=12)
    plt.savefig('/DATA/CGW/RoadEmbedding2/NewExperiment/img/cos_markov.png',dpi=1600)
    plt.show()


if __name__ == '__main__':
    # topology vs cosine first
    # 500 node pairs at least for each topology distance
    # n=100
    # topologyDistance = 100
    # G = loadGraph(weight=True)
    # result,distance = selectNodePairs(G, topologyDistance)
    # np.save('./data/nodesTopologyDistance{}.npy'.format(topologyDistance),np.array(result))
    # np.save('./data/nodesTopologyDistance{}_2.npy'.format(topologyDistance),np.array(result))
    # np.save('./data/nodesMarkovDistance{}_2.npy'.format(topologyDistance),np.array(distance))
    # exit(0)
    # pairs = np.load('./data/nodesTopologyDistance100.npy', allow_pickle=True)
    # paths = [
    #     r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\Deepwalk\deepwalk_8.csv',
    #     r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\Node2Vec_\node2vec_8.csv',
    #     r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\Trans2Vec\markov2vec_8.csv',
    #     r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\Trans2Vec\trans2vec_8_7&3.csv'
    # ]
    # cos = []
    # for path in paths:
    #     roadEmbedding = load_road_embedding(path)
    #     cos.append(calculateTopologyCosine(pairs, roadEmbedding))
    # np.save(r'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\cos_topology\cos_topology.npy',np.array(cos))

    # G = loadGraph(weight=True)
    # for path in paths:
    #     roadEmbedding = load_road_embedding(path)
    #     res = calculateMarkovCosine(G, roadEmbedding)
    #     print(res)
    markov_cos = [
        [0.9964872718836478, 0.9953805565672138, 0.9952218224972428, 0.9947436484543483, 0.9951989983297777,
         0.9937176232211655, 0.9944347301513439, 0.9926888567798432, 0.99291642896195, 0.9934266302546715,
         0.9909950179786865, 0.9944697845316426, 0.9944635699634062, 0.995971779901327, 0.9959312676560325,
         0.9954296351902658, 0.9963619958846895, 0.9965963652270987, 0.9962058099284892, 0.9978728940467788]
        ,
        [0.9943727711318949, 0.9926826690472536, 0.9924075342888252, 0.9916211959299112, 0.9921599835562045,
         0.9899780093079952, 0.9910302440222066, 0.9885474678953704, 0.9886777360945664, 0.9888197454690674,
         0.9851552554392993, 0.9905056657645012, 0.9907035206438748, 0.9929511257694366, 0.9928541558631775,
         0.9924881107456448, 0.9934659916223008, 0.9940104826077526, 0.9935761423997207, 0.996493993839079]
        ,
        [0.8838886570190043, 0.9533185619056632, 0.9600340273256123, 0.9642928233783832, 0.9665342824095343,
         0.9714496173309776, 0.9708316541141179, 0.975293270254593, 0.9753295214804454, 0.9743049713357206,
         0.9758475570502853, 0.9776714311257467, 0.9820597591493171, 0.9775797766945485, 0.9793742917608597,
         0.9870214985887079, 0.9810796763035107, 0.9819048635756207, 0.9891812523671719, 0.9475739046538376]
        ,
        [0.9752627945676879, 0.9784837519271615, 0.9805786932015068, 0.9807190065860983, 0.9829962359402902,
         0.9820606201706475, 0.9841572264572066, 0.9829477503812121, 0.9841663143136495, 0.9841344082995482,
         0.9825894275692492, 0.9871739638992165, 0.9883616937816795, 0.9894527141199837, 0.9900223121614566,
         0.9910430296105953, 0.9912060927463567, 0.9915245396699939, 0.9920141465281094, 0.9920759248580626]
    ]

    topology_cos = np.load(r'/DATA/CGW/RoadEmbedding2/model/cos_topology/cos_topology.npy',
                           allow_pickle=True)
    # visualTopologyCosineWithPlot(topology_cos)
    # visualTopologyCosineWithPolar(topology_cos)
    # visualTopologyCosineWithPCOLOR(topology_cos)
    visualMarkovCosineWithPlot(markov_cos)
    # visualMarkovCosineWithPolar(markov_cos)