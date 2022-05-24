# 利用折线图或柱状图来可视化时间旅行估计结果
# 结果顺序为 deepwalk node2vec road2vec trans2vec

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

result_d4 = [0.19262456799594638, 0.21642275800546393, 0.21512838450666422, 0.19659407661917308]
result_d8 = [0.20183301093176198, 0.19597851015173662, 0.24711848199606323, 0.19355594084407082]
result_d12 = [0.207088244243119, 0.19718117097344917, 0.20477958367421067, 0.18701263996533668]
result_d16 = [0.22416474061230054, 0.19489746039900452, 0.20182632685949406, 0.19626117062468368]

result_percent = [0.21224937720510823, 0.21067762858449265, 0.21339721594435665, 0.19355594084407082,
                  0.19583308401828012, 0.21535508916116294, 0.19374255444973898, 0.199180682703184, 0.20793057975765222,
                  0.19893064065119967, 0.19597851015173662]

result_window = [0.21438896192166096, 0.2002217830457271, 0.240775196830957, 0.21535508916116294, 0.2002753597409584,
                 0.19169523375323758, 0.21670125449338976, 0.19490299849675444]

nodeIncrease = [5288, 5373, 5475, 5602, 5743, 5909, 6093, 6287, 6481, 6710, 7165, 7669, 8239, 8836, 9397, 9992, 10671,
                11381, 12402, 13495, 14663, 15893, 17172, 18483, 19877, 21349, 22905, 24517, 26218, 27969]
memoryIncrease = [213.34375, 220.2578125, 228.69921875, 239.4296875, 251.63671875, 266.390625, 283.2421875, 301.5625,
                  320.4609375, 343.5078125, 391.67578125, 448.71484375, 517.89453125, 595.66796875, 673.70703125,
                  761.72265625, 868.76171875, 988.21875, 1173.4765625, 1389.4296875, 1640.35546875, 1927.08984375,
                  2249.73828125, 2606.37109375, 3014.33984375, 3477.32421875, 4002.67578125, 4561.90234375,
                  5244.3359375, 5968.22265625]


def visual1(x1, x2):
    x = ['DeepWalk', 'node2vec', 'Road2vec', 'Trans2vec']
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('dimension 8')
    ax1.set_ylim(0.18, 0.26)
    ax1.set_ylabel('MAPE', fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('dimension 16')
    ax2.set_ylim(0.18, 0.26)
    ax2.set_ylabel('MAPE', fontsize=10)

    ax1.bar(x, x1, width=0.7)
    ax2.bar(x, x2, width=0.7)
    plt.show()


def visual2(y_):
    x = ['10:0', '9:1', '8:2', '7:3', '6:4', '5:5', '4:6', '3:7', '2:8', '1:9', '0:10']
    plt.ylim(0.18, 0.24)
    plt.plot(x, y_, lw=3, c='black')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('MAPE', fontsize=13)
    # plt.xlabel('转移模式:拓扑结构',fontsize=13)
    plt.show()


def visual3(y_):
    x = [i for i in range(2, 10, 1)]
    plt.ylim(0.18, 0.26)
    plt.plot(x, y_, lw=3, c='black')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylabel('MAPE', fontsize=13)
    # plt.xlabel('window size',fontsize=13)
    plt.show()


# visual1(result_d8, result_d16)
# visual2(result_percent)
# visual3(result_window)

def visualDimensionMAPE():
    x = np.array([i for i in range(4)])
    result = np.array([result_d4, result_d8, result_d12, result_d16])
    width = 0.2
    plt.bar(x, result[:, 0], width=width, color='#194f97', edgecolor='black')
    plt.bar(x + width, result[:, 1], width=width, color='#bd6b08', edgecolor='black')
    plt.bar(x + width * 2, result[:, 2], width=width, color='#007f54', edgecolor='black')
    plt.bar(x + width * 3, result[:, 3], width=width, color='#da1f18', edgecolor='black')
    plt.ylabel('MAPE', fontsize=14)
    plt.legend(['DeeWalk', 'Node2Vec', 'Road2Vec', 'Trans2Vec'], fontsize=14)
    plt.xticks([],size=12)
    plt.yticks(size=12)
    plt.ylim(0.15, 0.28)
    plt.show()


def visual_k(data):
    plt.plot(data, lw=3, c='black')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(c='grey', linestyle='--')
    plt.show()


def visual_k_error():
    x = [_ for _ in range(1, 31)]
    maxNodeNum = [5288, 5373, 5475, 5602, 5743, 5909, 6093, 6287, 6481, 6710, 7165, 7669, 8239, 8836, 9397, 9992, 10671,
                  11381, 12402, 13495, 14663, 15893, 17172, 18483, 19877, 21349, 22905, 24517, 26218, 27969]
    minNodeNum = [107, 115, 120, 124, 131, 140, 144, 148, 152, 157, 164, 171, 178, 186, 195, 202, 211, 220, 228, 235,
                  243, 250, 257, 264, 271, 278, 286, 296, 308, 320]
    averageNodeNum = [1775, 1828, 1897, 1982, 2083, 2199, 2331, 2480, 2644, 2823, 3017, 3224, 3445, 3677, 3921, 4177,
                      4444, 4721, 5009, 5310, 5623, 5948, 6283, 6633, 6998, 7374, 7763, 8166, 8584, 9015]
    maxMatrixMemoryUsage = [213.34375, 220.2578125, 228.69921875, 239.4296875, 251.63671875, 266.390625, 283.2421875,
                            301.5625, 320.4609375, 343.5078125, 391.67578125, 448.71484375, 517.89453125, 595.66796875,
                            673.70703125, 761.72265625, 868.76171875, 988.21484375, 1173.4765625, 1389.4296875,
                            1640.3515625, 1927.08984375, 2249.73828125, 2606.3671875, 3014.33984375, 3477.32421875,
                            4002.68359375, 4585.90234375, 5244.3203125, 5968.21484375]
    minMatrixMemoryUsage = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01171875,
                            0.0234375, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.02734375, 0.02734375,
                            0.02734375, 0.02734375, 0.03125, 0.03515625, 0.04296875, 0.05859375, 0.05859375]
    averageMatrixMemoryUsage = [24.0390625, 25.49609375, 27.45703125, 29.97265625, 33.10546875, 36.89453125,
                                41.45703125, 46.92578125, 53.3359375, 60.8046875, 69.4453125, 79.3046875, 90.546875,
                                103.15234375, 117.296875, 133.11328125, 150.67578125, 170.04296875, 191.42578125,
                                215.12109375, 241.23046875, 269.921875, 301.1796875, 335.671875, 373.62890625,
                                414.85546875, 459.78125, 508.7578125, 562.17578125, 620.04296875]
    plt.plot(x, maxNodeNum, color='#0e2c82', marker='o', markersize=3)
    plt.plot(x, averageNodeNum, color='#e30039')
    plt.plot(x, minNodeNum, color='#00994e')
    plt.legend(['PYTHONRSB', 'PYTHONRSB', 'PYTHONRSB'], loc='center left', prop={'size': 14})
    # plt.ylabel('MB',{'size':16})
    plt.xlabel('k', {'size': 16})
    plt.xlim(1, 31)
    plt.ylim(0, 30000)
    plt.tick_params(labelsize=12)
    plt.hlines(10671, xmin=1, xmax=17, ls='--', color='gray')
    plt.hlines(27969, xmin=1, xmax=30, ls='--', color='gray')
    plt.vlines(17, ymin=-50, ymax=10761, ls='--', color='gray')
    plt.fill_between(x, averageNodeNum, maxNodeNum, color='C0', alpha=0.3,
                     interpolate=True)
    plt.fill_between(x, averageNodeNum, minNodeNum, color='C1', alpha=0.3,
                     interpolate=True)
    plt.show()


if __name__ == '__main__':
    # visualDimensionMAPE()
    visual_k_error()
