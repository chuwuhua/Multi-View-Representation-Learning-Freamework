from node2vec import Node2Vec
import utils
G = utils.load_graph('/DATA/CGW/RoadEmbedding2/data/dict_processed.pkl')
model = Node2Vec(G,num_walks=10,p=1,q=4,walk_length=30,dimensions=8,workers=4)
model_ = model.fit()
model_.wv.save_word2vec_format('node2vec_8.csv')