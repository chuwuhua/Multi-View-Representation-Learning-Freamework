import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm



def load_graph(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    G = nx.from_dict_of_lists(data)

    return G


def load_graph_with_weight(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    G = nx.from_dict_of_lists(data)
    for u, v in G.edges:
        G.add_edge(u, v, weight=0)
    return G


def load_road_embedding(path_road_embedding):
    road_embeddings = pd.read_csv(path_road_embedding, header=None).values
    road_embeddings = road_embeddings[np.argsort(road_embeddings[:, 0]), :]  # 升序排序
    road_embeddings_pad = np.zeros((int(np.max(road_embeddings[:, 0]) + 1), road_embeddings.shape[1] - 1))
    road_embeddings_pad[road_embeddings[:, 0].astype(np.int)] = road_embeddings[:, 1:]
    road_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(road_embeddings_pad))
    return road_embeddings


def all_segments():
    path = '/mnt/DATA2/CGW/data_1.0/npy/sub_segments'
    all_trip = []
    for num in range(0, 31):
        num = num + 1
        if num == 3:
            continue
        if num < 10:
            file = '2020080{}.npy'.format(num)
        else:
            file = '202008{}.npy'.format(num)
        trips = np.load(os.path.join(path, file), allow_pickle=True)
        all_trip += trips.tolist()
    return all_trip


def embedding_trip(path_road_embedding, path_index, save_path):
    road_embedding = load_road_embedding(path_road_embedding)
    index = np.load(path_index, allow_pickle=True)
    trips = np.array(all_segments())
    for step, i in enumerate(index):
        trip_batch = trips[i]
        trip_batch = [torch.tensor(i).long() for i in trip_batch]
        trip_batch = pad_sequence(trip_batch, batch_first=False)
        trip_batch = road_embedding(trip_batch)
        torch.save(trip_batch,
                   os.path.join(save_path, '{}.pt'.format(int(step))))



if __name__ == '__main__':
    embedding_trip('/DATA/CGW/RoadEmbedding2/model/Node2Vec_/node2vec_8.csv',
                   '/DATA/CGW/RoadEmbedding2/data/index_1024.npy',
                   '/DATA/CGW/RoadEmbedding2/data/Node2Vec_1024')
