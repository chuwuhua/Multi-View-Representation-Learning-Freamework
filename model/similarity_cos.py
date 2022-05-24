# calculate cosine similarity of vec and markov

from tqdm import tqdm
import networkx as nx
import utils
import os
import numpy as np
import pandas as pd
from random import choices
import pickle
def Tirps(path):
    for i in os.listdir(path):
        data = np.load(os.path.join(path, i), allow_pickle=True)
        yield i, data

def load_road_embedding(path_road_embedding):
    road_embeddings = pd.read_csv(path_road_embedding, header=None).values
    road_embeddings = road_embeddings[np.argsort(road_embeddings[:, 0]), :]  # 升序排序
    road_embeddings_pad = np.zeros((int(np.max(road_embeddings[:, 0]) + 1), road_embeddings.shape[1] - 1))
    road_embeddings_pad[road_embeddings[:, 0].astype(np.int)] = road_embeddings[:, 1:]
    return road_embeddings_pad

def network_markov(G,path):
    for name, trips in Tirps(path):
        bar = tqdm(iterable=trips, desc=name)
        for trip in bar:  # search roads in single trip
            for i in range(len(trip) - 1):
                G.get_edge_data(trip[i], trip[i + 1])['weight'] += 1
    return G

def cos_similarity(v1,v2):
    return float(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_graph(path):
    data = load_pickle(path)
    G = nx.from_dict_of_dicts(data)
    return G

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    f.close()

def select_edges(G,nodes,vec,k):
    # G.get_edge_data(u,v)['weight']
    result_weight = []
    result_cos = []
    for node in tqdm(nodes):
        nei = list(G.neighbors(node))
        weight = []
        cos = []
        for n in nei:
            weight.append(G.get_edge_data(node,n)['weight'])
            cos.append(cos_similarity(vec[node],vec[n]))
        weight = np.array(weight)/sum(weight)
        result_weight += weight.tolist()
        result_cos += cos
    # save to visual easily
    np.save('/DATA/CGW/RoadEmbedding2/model/cos_weight/trans2vec_8_7&3.npy',np.array([result_weight,result_cos]))

def visual():
    pass

if __name__ == '__main__':
    trip_path = '/mnt/DATA2/CGW/data_1.0/npy/sub_segments'
    cos_path = '/DATA/CGW/RoadEmbedding2/model/Trans2Vec/trans2vec_8_7&3.csv'
    cos = load_road_embedding(cos_path)
    # G = utils.load_graph_with_weight('/DATA/CGW/RoadEmbedding2/data/dict_processed.pkl')
    # G = network_markov(G,trip_path)
    # save_pickle('/DATA/CGW/RoadEmbedding2/data/dict_processed_weight.pkl',nx.to_dict_of_dicts(G))
    # exit()
    G = load_graph('/DATA/CGW/RoadEmbedding2/data/dict_processed_weight.pkl')
    # select some of G edges to visualize image
    # select 500 nodes and their neighbors
    nodes = np.load('/DATA/CGW/RoadEmbedding2/model/selected_nodes.npy',allow_pickle=True)
    select_edges(G,nodes,cos,k=500)



