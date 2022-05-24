# visualize cos similarity and weight
import numpy as np
import matplotlib.pyplot as plt
import os


def visual(data):
    """
    :param data: [[weight],[cos]]
    :return: none
    scatter and line with k=1,b=0
    """
    plt.scatter(data[0], data[1], c='black', s=8)
    plt.xlim(-0.02, 1.02)
    plt.ylim(0, 1.001)
    plt.title('Road2vec')
    plt.show()

def snap(data):
    ans = []
    for i in data:
        ans += i
    return ans

def visualOrder(data1,data2,data3,data4,order):
    # plot cosine similarity with order
    order1 = snap(data1[:,order])
    order2 = snap(data2[:,order])
    order3 = snap(data3[:,order])
    order4 = snap(data4[:,order])
    plt.boxplot([order1,order2,order3,order4],showfliers=True,flierprops={'marker':'+','linewidth':0.1})
    # plt.plot([1 for i in range(len(order1))],order1)
    plt.ylim(0,1.01)
    plt.title('order {}'.format(order))
    plt.show()


if __name__ == '__main__':
    # file_path = 'D:\Desktop\毕业相关\小论文\data\RoadEmbedding2\model\cos_weight'
    # weight_cos = np.load(os.path.join(file_path, 'markov2vec_8.npy'), allow_pickle=True)
    # visual(weight_cos)
    order_cos_deepwalk = np.load(r'/DATA/CGW/RoadEmbedding2/model/cos_order/deepwalk_8.npy',allow_pickle=True)
    order_cos_node2vec = np.load(r'/DATA/CGW/RoadEmbedding2/model/cos_order/node2vec_8.npy',allow_pickle=True)
    order_cos_road2vec = np.load(r'/DATA/CGW/RoadEmbedding2/model/cos_order/markov2vec_8.npy',allow_pickle=True)
    order_cos_trans2vec = np.load(r'/DATA/CGW/RoadEmbedding2/model/cos_order/trans2vec_8_7&3.npy',allow_pickle=True)
    visualOrder(order_cos_deepwalk,order_cos_node2vec,order_cos_road2vec,order_cos_trans2vec,order=4)
