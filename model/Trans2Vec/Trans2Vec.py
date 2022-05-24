# struct weight graph
# struct markov matrix using all data
# random walk using markov matrix
# Trans2Vec
import os
import numpy as np
from tqdm import tqdm
import utils
from node2vec import Node2Vec
import gensim
import pkg_resources


def Tirps(path):
    for i in os.listdir(path):
        data = np.load(os.path.join(path, i), allow_pickle=True)
        yield i, data


def Markov(G, path):
    for name, trips in Tirps(path):
        bar = tqdm(iterable=trips, desc=name)
        for trip in bar:  # search roads in single trip
            for i in range(len(trip) - 1):
                G.get_edge_data(trip[i], trip[i + 1])['weight'] += 1
    return G


class Trans2Vec(Node2Vec):
    def fit(self, path, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameters for gensim.models.Word2Vec - do not supply 'size' / 'vector_size' it is
            taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        # Figure out gensim version, naming of output dimensions changed from size to vector_size in v4.0.0
        gensim_version = pkg_resources.get_distribution("gensim").version
        size = 'size' if gensim_version < '4.0.0' else 'vector_size'
        if size not in skip_gram_params:
            skip_gram_params[size] = self.dimensions

        if 'sg' not in skip_gram_params:
            skip_gram_params['sg'] = 1
        np.save(path, np.array(self.walks))
        exit(0)
        # return gensim.models.Word2Vec(self.walks, **skip_gram_params)


if __name__ == '__main__':
    '''
    # simulate trans2vec walks with markov
    trip_path = '/mnt/DATA2/CGW/data_1.0/npy/sub_segments'
    # create graph with weigth=0
    G = utils.load_graph_with_weight('/DATA/CGW/RoadEmbedding2/data/dict_processed.pkl')
    # add weight from trips
    G = Markov(G, trip_path)
    tv1 = Trans2Vec(graph=G, p=1, q=1, num_walks=10, walk_length=30, workers=6)
    tv1.fit(path = 'data/Trans2Vec_walks1.npy')  # save walks
    '''
    '''
    # simulate trans2vec walks without markov
    G = utils.load_graph('/DATA/CGW/RoadEmbedding2/data/dict_processed.pkl')
    # q>1 BFS
    tv2 = Trans2Vec(graph=G, p=4, q=4, num_walks=10, walk_length=30, workers=1)
    tv2.fit(path = 'data/Trans2Vec_walks2.npy')
    '''
    # fit walks
    data1 = np.load('data/Trans2Vec_walks1.npy', allow_pickle=True)
    
    data2 = np.load('./data/Trans2Vec_walks2.npy', allow_pickle=True).tolist()
    model = gensim.models.Word2Vec(data1 + data2, vector_size=8, window=5, batch_words=4, min_count=1)
    model.wv.save_word2vec_format('trans2vec_8.csv')

    # fit walks 2 use markov only
    # data1 = np.load('data/Trans2Vec_walks1.npy', allow_pickle=True).tolist()
    model = gensim.models.Word2Vec(data1, vector_size=8, window=5, batch_words=4, min_count=1)
    model.wv.save_word2vec_format('markov2vec_8.csv')

    # fit walk 3 use struct only
    # data2 = np.load('data/Trans2Vec_walks2.npy', allow_pickle=True).tolist()
    model = gensim.models.Word2Vec(data2, vector_size=8, window=5, batch_words=4, min_count=1)
    model.wv.save_word2vec_format('struct2vec_8.csv')