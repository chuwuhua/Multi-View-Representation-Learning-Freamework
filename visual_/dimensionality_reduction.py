# 利用pca对嵌入结果进行降维，便于二维或者三维可视化
'''
多种方法可用
pca: keep original infomation
lda: make final data easy to classify  有监督的计算方法
lle: keep original structure
laplacian: like lle, keep info and struct
'''

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import LocallyLinearEmbedding as LLE
import pandas as pd
import numpy as np

class DimensionReduce():
    def __init__(self, method='pca', vector_size=2):
        self.method = method
        self.vector_size = vector_size

    def prepare(self):
        if self.method == 'pca':
            self.model = PCA(n_components=self.vector_size)
        elif self.method == 'lle':
            self.model = LLE(n_components=self.vector_size)

    def fit(self, data):
        print('use model {} and dimension {}'.format(self.method, self.vector_size))
        self.prepare()
        data = self.model.fit_transform(data)
        return data


if __name__ == '__main__':
    DR = DimensionReduce(method='pca')
    path_road_embedding = '/DATA/CGW/RoadEmbedding2/model/Trans2Vec/trans2vec_8.csv'
    road_embeddings = pd.read_csv(path_road_embedding, header=None).values
    result = DR.fit(road_embeddings[:, 1:])
    np.save('/DATA/CGW/RoadEmbedding2/visual_/data/{}'.format('pca_trans2vec_8to2.npy'),result)
    print('finish')