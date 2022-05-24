import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader
# from DeepTTE.dataset import DiDiData
from data.dataset import DiDiData, index
from data_xa.dataset_xa import XAData
from utilis import fold_cross, rmse, mae
from rnndeepnet_d import edn, mape, NoamOpt
import time

fold = 6
lr = 0.0005
# lr = 0.00001
epoch = 50
data_length = 8443
# data_length = 305
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
string_param = 'road2vec12_allfeature'
train_index, test_index = index(data_length)

train_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=train_index)
test_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=test_index)
# train_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=train_index)
# test_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=test_index)



train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                             persistent_workers=True)

net = edn(in_feature_d=32, in_feature_s=13, hidden_unit_d=64, hidden_unit_s=128, out_feature_d=32)
# net = edn(in_feature_d=32, in_feature_s=33, hidden_unit_d=32, hidden_unit_s=64, out_feature_d=32)
# net = edn(in_feature_d=32, in_feature_s=17, hidden_unit_d=32, hidden_unit_s=64, out_feature_d=32)
net.reset_parameters()
net.to(device)
print('***** using device {} *****'.format(device))

criterion_regression = mape()

# optimizer
optimizer = optim.AdamW(net.parameters(), lr=lr)
optimizer_w = NoamOpt(d_model=64, factor=1, warmup=20 * data_length, optimizer=optimizer)

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
        sparse_feature = sparse_feature.to(device)
        # label = (label+0.001).to(device)
        label = label.to(device)
        optimizer_w.zero_grad()
        outs = net(global_feature, trip_feature, lengthes)

        loss = criterion_regression(outs, label)
        loss.backward()
        optimizer_w.step()

        c_step, c_lr = optimizer_w.qurey()

        running_loss = running_loss + loss.item()
        l2 = rmse(outs, label)
        l3 = mae(outs, label)

        if step % 1000 == 999:
            end_time = time.time()
            print(
                "epoch: {}, step: {}, loss:{}, rmse: {}, mae: {}, step:{}, lr:{}, time:{}".format(e, step,
                                                                                                  running_loss / 1000,
                                                                                                  l2,
                                                                                                  l3,
                                                                                                  c_step, c_lr,
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
            # label = (label+0.001).to(device)
            label = label.to(device)
            outs = net(global_feature, trip_feature, lengthes)

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
                       r'/DATA/CGW/graduate/model/DeepTTE/model_backup/{}'.format(string_param))
            np.save(
                r'/DATA/CGW/graduate/model/DeepTTE/model_backup/best_loss_{}.npy'.format(string_param), np.array(
                    [best_loss]))
        print(' test loss: {}, best loss: {}, rmse: {}, mae: {},  epoch time: {}'.format(final_loss, best_loss,
                                                                                         final_rmse, final_mae,
                                                                                         epoch_end - epoch_start))
# deepwalk 12 all_feature test loss: 0.14355550420187793, best loss: 0.1391509050084958, rmse: 159.95656593825757, mae: 112.3088238155517,  epoch time: 94.58967876434326
# node2vec 12 all feature test loss: 0.1387870551520452, best loss: 0.13840823549081627, rmse: 152.23468110799507, mae: 106.44560605548749,  epoch time: 97.37213492393494
# trans2vec 12 all feature test loss: 0.13540088895971578, best loss: 0.13540088895971578, rmse: 152.20404520648484, mae: 106.34732681708142,  epoch time: 95.49702382087708

# xa trans2vec all feature test loss: 0.1315277692578409, best loss: 0.1315277692578409, rmse: 86.54781820463097, mae: 68.03463270353234,  epoch time: 1.3303241729736328
# xa node2vec all feature test loss: 0.2765791516589082, best loss: 0.12520043288721985, rmse: 88.34747285428254, mae: 71.43195504727571,  epoch time: 1.2556703090667725
# xa deepwalk test loss: 0.27852499436425127, best loss: 0.1421854376954877, rmse: 154.01617473104724, mae: 133.50435273543647,  epoch time: 1.2400290966033936