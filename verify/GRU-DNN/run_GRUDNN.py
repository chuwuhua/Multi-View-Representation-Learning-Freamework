import torch
import torch.optim as optim
from data.dataset import DiDiData,index
from data_xa.dataset_xa import XAData
from GRUDNNnet import GRUDNNnet,mape
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
import numpy as np

lr = 0.001
epoch = 50
data_length = 8443  # batch数量
# data_length = 305  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
string_param = 'road2vec_allfeature'


train_index, test_index = index(data_length)

train_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=train_index)
test_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=test_index)
# train_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=train_index)
# test_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=test_index)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                             persistent_workers=True)
net = GRUDNNnet(input_size=13,g_input=32,g_hidden=128,hidden_size=64,g_out=128)
net.reset_parameters()
net.to(device)
print('***** using device {} *****'.format(device))


criterion_regression = mape()

# optimizer
optimizer = optim.AdamW(net.parameters(), lr=lr)
best_loss = float('inf')  # 确定存储哪一组参数

for e in range(epoch):
    epoch_start = time.time()
    running_loss = 0
    net.train()
    start_time = time.time()
    for step, data in enumerate(train_dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = data
        global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        # sparse_feature = sparse_feature.to(device)
        # label = label.to(device)
        label = (label+0.01).to(device)
        optimizer.zero_grad()
        outs = net(trip_feature,global_feature)
        loss = criterion_regression(outs, label)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        l2 = rmse(outs, label)
        l3 = mae(outs, label)

        if step % 1000 == 999:
            end_time = time.time()
            print(
                "epoch: {}, step: {}, loss:{}, rmse: {}, mae: {},time:{}".format( e, step,
                                                                                         running_loss / 1000,
                                                                                         l2.item(),
                                                                                         l3.item(),

                                                                                         end_time - start_time))
            running_loss = 0
            running_loss_c = 0
            start_time = time.time()
    print('******** waiting test **********')

    with torch.no_grad():
        net.eval()
        test_loss = 0
        l2p = 0
        l3p = 0
        for step, j in enumerate(test_dataloader):
            global_feature, trip_feature, sparse_feature, lengthes, label = j
            global_feature = global_feature.to(device)
            trip_feature = trip_feature.to(device)
            sparse_feature = sparse_feature.to(device)
            label = (label+0.01).to(device)
            # label = label.to(device)
            outs = net(trip_feature,global_feature)

            t_loss = criterion_regression(outs, label)
            l2 = rmse(outs, label)
            l3 = mae(outs, label)
            test_loss += t_loss.item()
            l2p += l2.item()
            l3p += l3.item()
        final_loss = test_loss / len(test_index)
        final_rmse = l2p / len(test_index)
        final_mae = l3p / len(test_index)
        # writer.add_scalar('use_fold_{}'.format(use_fold),final_loss,e)
        epoch_end = time.time()
        if final_loss < best_loss:
            best_loss = final_loss

            torch.save(net.state_dict(),
                       r'/DATA/CGW/graduate/model/GRU-DNN/model_backup/{}'.format(string_param))
            np.save(r'/DATA/CGW/graduate/model/GRU-DNN/model_backup/best_loss_{}.npy'.format(string_param),
                    np.array([best_loss]))
        print('test loss: {}, best loss: {}, rmse:{}, mae:{},  epoch time: {}'.format(final_loss, best_loss, final_rmse,
                                                                                         final_mae,
                                                                                  epoch_end - epoch_start))
        # deepwalk all feature 12 test loss: 0.1403216538377962, best loss: 0.13933450084192875, rmse:151.9449765340547, mae:106.78003881027517,  epoch time: 62.07654023170471
        # node2vec all feature 12 test loss: 0.14839623601679927, best loss: 0.13887883473056792, rmse:151.80406771786733, mae:110.83652684889054,  epoch time: 62.415711879730225
        # trans2vec all feature 12 test loss: 0.13992780473478877, best loss: 0.13668551578076518, rmse:154.1374173742295, mae:107.8605235220649,  epoch time: 87.15341234207153

        # deepwalk xa test loss: 0.04613724608055275, best loss: 0.026918345404302945, rmse:48.87935639464337, mae:44.26142101702483,  epoch time: 1.1515703201293945
        # node2vec xa test loss: 0.02512629883890243, best loss: 0.020446090746428006, rmse:31.392211447591368, mae:27.71260813785636,  epoch time: 1.186863899230957
        # trans2vec xa test loss: 0.039706375328419, best loss: 0.01873707588073676, rmse:23.961566593336023, mae:19.428345980851546,  epoch time: 1.2834100723266602