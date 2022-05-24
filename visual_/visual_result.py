import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def visual_result_sz():
    string_param = 'WideDeepNet'

    path = '/DATA/CGW/graduate/model/{}/result'.format(string_param)
    result = []
    model = ['deepwalk', 'node2vec', 'trans2vec']

    for i in model:
        data = np.load(os.path.join(path, 'link_num_{}.npy'.format(i)), allow_pickle=True)
        result.append(data)
    temp = np.append(result[0][:, 0:2], result[1][:, [0]], axis=1)
    temp = np.append(temp, result[2][:, [0]], axis=1)
    result_pd = pd.DataFrame(temp, columns=['deepwalk', 'link_num', 'node2vec', 'trans2vec'])
    result_pd['link_num'] = result_pd['link_num'].apply(lambda x: (x // 20) * 20)
    link_num_mape = result_pd.groupby('link_num')['deepwalk', 'node2vec', 'trans2vec'].mean()
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.grid(False)  # 不要背景
    l1, = ax1.plot(link_num_mape['deepwalk'], color='#F7A35C', marker='+', lw=1)
    l2, = ax1.plot(link_num_mape['node2vec'], color='#9BBB59', marker='x', lw=1)
    l3, = ax1.plot(link_num_mape['trans2vec'], color='#4BACC6', marker='^', lw=1.5)
    plt.legend(handles=[l1, l2, l3], labels=['deepwalk', 'node2vec', 'trans2vec'], loc='upper center',
               bbox_to_anchor=(0.3, 0.95), fontsize=14)
    plt.savefig('/DATA/CGW/graduate/visual/figure/{}.png'.format(string_param), dpi=1600)
    plt.show()


def prepareTravelFeature():
    travelFeature = []
    for i in range(1, 32):
        if i == 3: continue
        if i < 10:
            file = '2020080{}.npy'.format(i)
        else:
            file = '202008{}.npy'.format(i)
        data = np.load(os.path.join('/mnt/DATA2/CGW/data_1.0/npy/travel_feature', file), allow_pickle=True).tolist()
        travelFeature += data
    travelFeature = np.array(travelFeature)
    index = np.load('/mnt/DATA2/CGW/data_1.0/npy/tuple_data/index_1024.npy', allow_pickle=True)
    travelFeatureBatch = []
    for i in index:
        i = i[0:len(i):3]
        travelFeatureBatch.append(travelFeature[i])
    np.save('/DATA/CGW/graduate/visual/originalData/travelFeatureBathc.npy', np.array(travelFeatureBatch))


def prepareWeekDay():
    travelFeature = []
    for i in range(1, 32):
        if i == 3: continue
        if i < 10:
            file = '2020080{}.npy'.format(i)
        else:
            file = '202008{}.npy'.format(i)
        data = np.load(os.path.join('/mnt/DATA2/CGW/data_1.0/npy/date_week', file), allow_pickle=True).tolist()
        travelFeature += data
    travelFeature = np.array(travelFeature)
    index = np.load('/mnt/DATA2/CGW/data_1.0/npy/tuple_data/index_1024.npy', allow_pickle=True)
    travelFeatureBatch = []
    for i in index:
        i = i[0:len(i):3]
        travelFeatureBatch.append(travelFeature[i])
    np.save('/DATA/CGW/graduate/visual/originalData/DateWeekBatch.npy', np.array(travelFeatureBatch))


def process_label_out():
    string_param = 'GRU-DNN'
    result = []
    model = ['deepwalk', 'trans2vec']
    travelFeature = np.load('/DATA/CGW/graduate/visual/originalData/travelFeatureBathc.npy', allow_pickle=True)
    path = '/DATA/CGW/graduate/model/{}/result'.format(string_param)

    label_out = []
    for i in model:
        data = np.load(os.path.join(path, 'label_out_{}.npy'.format(i)), allow_pickle=True)
        label_out.append(data)
    df = []
    for i in range(8443):
        tf = travelFeature[i]
        deepwalk = label_out[0][i]
        trans2vec = label_out[1][i]
        for j in range(len(tf)):
            # date = int(tf[j][0])
            # week = int(tf[j][1])
            distance = tf[j][2]
            label1 = deepwalk[0][j][0]
            # out1 = deepwalk[1][j][0]
            out1 = deepwalk[1][j][0][0][0]
            loss1 = np.abs((label1 - out1) / out1)
            label2 = trans2vec[0][j][0]
            # out2 = trans2vec[1][j][0]
            out2 = trans2vec[1][j][0][0][0]
            loss2 = np.abs((label2 - out2) / out2)
            df.append([distance, loss1, loss2])
    df = np.array(df)
    np.save('/DATA/CGW/graduate/visual/originalData/distance_loss_{}.npy'.format(string_param), df)


def visualSlice():
    string_param = 'WideDeepNet'
    data = np.load('/DATA/CGW/graduate/visual/originalData/slice_loss_{}.npy'.format(string_param), allow_pickle=True)
    data = pd.DataFrame(data, columns=['slice', 'loss1', 'loss2'])
    loss = data.groupby('slice')['loss1', 'loss2'].mean()

    plt.figure(figsize=(5, 5))
    plt.plot(loss['loss2'], color='#F7A35C', marker='+', lw=1)
    plt.plot(loss['loss1'], color='#4BACC6', marker='x', lw=1)
    plt.legend(['ababababaaa', 'abababbabaaa'])
    plt.savefig('/DATA/CGW/graduate/visual/figure/slice_{}.png'.format(string_param))

    # plt.show()


def visualDistance():
    string_param = 'FMAETA'
    data = np.load('/DATA/CGW/graduate/visual/originalData/distance_loss_{}.npy'.format(string_param), allow_pickle=True)
    data = pd.DataFrame(data, columns=['distance', 'loss1', 'loss2'])
    data['distance'] = data['distance'].apply(lambda x: x // 1000)
    loss = data.groupby('distance')['loss1', 'loss2'].mean()

    plt.figure(figsize=(5, 5))
    plt.plot(loss['loss1'], color='#F7A35C', marker='+', lw=1)
    plt.plot(loss['loss2'], color='#4BACC6', marker='x', lw=1)
    plt.legend(['ababababaaa', 'abababbabaaa'])
    plt.savefig('/DATA/CGW/graduate/visual/figure/distance_{}.png'.format(string_param))
    plt.show()

def visualWeather():
    string_param = 'WideDeepNet'
    data = np.load('/DATA/CGW/graduate/visual/originalData/dateweek_loss_{}.npy'.format(string_param),
                   allow_pickle=True)
    data = pd.DataFrame(data, columns=['date', 'week', 'loss1', 'loss2'])
    weather = [5, 4, 3, 3, 5, 1, 1, 1, 1, 1, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 5]
    data['date'] = data['date'].apply(lambda x: weather[int(x)])
    loss = data.groupby('date')['loss2', 'loss1'].mean()
    plt.figure(figsize=(5, 5))
    # plt.plot(loss['loss2'], color='#F7A35C', lw=1)
    # plt.plot(loss['loss1'], color='#4BACC6', lw=1)
    loss.plot.bar()
    plt.legend(['ababababaaa', 'abababbabaaa'])
    plt.ylim(0, 0.3)
    plt.xticks([])
    # plt.savefig('/DATA/CGW/graduate/visual/figure/slice_{}.png'.format(string_param))
    plt.show()


if __name__ == '__main__':
    # 可视化初步结果
    # visual_result_sz()
    # prepareTravelFeature()
    process_label_out()
    # visualSlice()
    # prepareWeekDay()
    # visualWeather()
    # visualDistance()
