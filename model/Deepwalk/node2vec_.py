from node2vec import Node2Vec
import utils
# deepwalk when p=1 and q=1
G = utils.load_graph('/DATA/CGW/RoadEmbedding2/data/dict_processed.pkl')
model = Node2Vec(G,num_walks=10,p=1,q=1,walk_length=30,dimensions=16,workers=4)
model_ = model.fit()
model_.wv.save_word2vec_format('deepwalk_16.csv')