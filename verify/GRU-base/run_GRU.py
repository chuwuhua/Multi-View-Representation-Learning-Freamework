import torch
import torch.optim as optim
from data.dataset import DiDiData,index
from data_xa.dataset_xa import XAData
from GRUnet import GRUnet,mape
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
import numpy as np

lr = 0.0001
epoch = 100
data_length = 8443  # batch数量
# data_length = 305  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
string_param = 'trans2vec_allfeature'


train_index, test_index = index(data_length)

train_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=train_index)
test_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=test_index)
# train_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=train_index)
# test_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=test_index)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                             persistent_workers=True)
net = GRUnet(input_size=13,hidden_size=128)
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
        # global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        # sparse_feature = sparse_feature.to(device)
        label = label.to(device)
        # label = (label+0.01).to(device)
        optimizer.zero_grad()
        outs = net(trip_feature)
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
            # global_feature = global_feature.to(device)
            trip_feature = trip_feature.to(device)
            # sparse_feature = sparse_feature.to(device)
            # label = (label+0.01).to(device)
            label = label.to(device)
            outs = net(trip_feature)

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
                       r'/DATA/CGW/graduate/model/GRU-base/model_backup/{}'.format(string_param))
            np.save(r'/DATA/CGW/graduate/model/GRU-base/model_backup/best_loss_{}.npy'.format(string_param),
                    np.array([best_loss]))
        print('test loss: {}, best loss: {}, rmse:{}, mae:{},  epoch time: {}'.format(final_loss, best_loss, final_rmse,
                                                                                         final_mae,
                                                                                  epoch_end - epoch_start))
        # deepwalk all feature 12 test loss: 0.15381757256439257, best loss: 0.1525599697727579, rmse:168.73838793200747, mae:119.98466390947839,  epoch time: 60.47373175621033
        # node2vec all feature 12 test loss: 0.1558075108625321, best loss: 0.1558075108625321, rmse:172.02697681919463, mae:122.18419834563161,  epoch time: 60.627262592315674
        # trans2vec all feature 12 test loss: 0.14689425549933527, best loss: 0.1459074190134264, rmse:158.97441059809827, mae:113.78500214707367,  epoch time: 83.21717023849487

        # deepwalk xa test loss: 0.09348032623529434, best loss: 0.035298973564868386, rmse:119.47641724088918, mae:102.49730403526969,  epoch time: 0.9799098968505859
        # node2vec xa test loss: 0.0578711124987382, best loss: 0.05356083846772495, rmse:104.46912912700488, mae:90.04676536891772,  epoch time: 0.9963569641113281
        # trans2vec xa test loss: 0.04067900220094168, best loss: 0.027622501284855862, rmse:54.348037605700284, mae:43.765156087668046,  epoch time: 1.109128713607788