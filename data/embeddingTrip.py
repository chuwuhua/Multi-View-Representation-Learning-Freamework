import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os


def extarctTrip():
    data = []
    index = np.load('/mnt/DATA2/CGW/data_1.0/npy/tuple_data/index_1024.npy',allow_pickle=True)
    index = [i[0:len(i):3] for i in index]
    for i in tqdm(range(1, 32)):
        if i == 3: continue
        if i < 10:
            file = '2020080{}.npy'.format(i)
        else:
            file = '202008{}.npy'.format(i)
        data += np.load(os.path.join('/mnt/DATA2/CGW/data_1.0/npy/sub_segments', file), allow_pickle=True).tolist()
    for step,i in enumerate(index):
        temp = []
        for j in i:
            temp.append(data[j])
        np.save(os.path.join('/DATA/CGW/graduate/data/trainingData/trip_batch','{}.npy'.format(step)),np.array(temp))

def load_road_embedding(path_road_embedding):
    road_embeddings = pd.read_csv(path_road_embedding, header=None).values
    road_embeddings = road_embeddings[np.argsort(road_embeddings[:, 0]), :]  # 升序排序
    road_embeddings_pad = np.zeros((int(np.max(road_embeddings[:, 0]) + 1), road_embeddings.shape[1] - 1))
    road_embeddings_pad[road_embeddings[:, 0].astype(np.int)] = road_embeddings[:, 1:]
    road_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(road_embeddings_pad))
    return road_embeddings


def embedding_trip2(road_embedding, save_path):
    road_embedding = load_road_embedding(road_embedding)
    trip_path = '/DATA/CGW/graduate/data/trainingData/trip_batch'
    for file in tqdm(os.listdir(trip_path)):
        name = file[0:-4]
        trip_batch = np.load(os.path.join(trip_path, file), allow_pickle=True)
        trip_batch = [torch.tensor(i).long() for i in trip_batch]
        trip_batch = pad_sequence(trip_batch, batch_first=False)
        trip_batch = road_embedding(trip_batch)
        torch.save(trip_batch,
                   os.path.join(save_path, '{}.pt'.format(name)))


if __name__ == '__main__':
    embedding_trip2('/DATA/CGW/graduate/data/roadEmbedding/Road2Vec/road2vec_12.csv',
                    '/DATA/CGW/graduate/data/trainingData/trip_embedding')

    # extarctTrip()