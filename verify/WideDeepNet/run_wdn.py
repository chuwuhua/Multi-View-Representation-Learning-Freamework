import torch
import torch.optim as optim
from data.dataset import DiDiData,index
from data_xa.dataset_xa import XAData
from wdn import WideDeepNet, mape, NoamOpt
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
import numpy as np

lr = 0.0005
epoch = 50
# data_length = 8443  # batch数量
data_length = 305  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
string_param = 'trans2vec_allfeature_xa'


train_index, test_index = index(data_length)

# train_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=train_index)
# test_dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=test_index)
train_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=train_index)
test_dataset = XAData(source_data=r'/DATA/CGW/graduate/data_xa/data_processed_1024', index=test_index)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=True)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                             persistent_workers=True)


# net = WideDeepNet(in_wide=22, in_deep=32, in_rnn=33, v_wide=8, hidden_wide=16, hidden_deep=128, out_wide=32, out_deep=128, out_rnn=128)
# net = WideDeepNet(in_wide=22, in_deep=32, in_rnn=13, v_wide=8, hidden_wide=16, hidden_deep=128, out_wide=32, out_deep=128, out_rnn=128)
net = WideDeepNet(in_wide=25, in_deep=32, in_rnn=17, v_wide=8, hidden_wide=16, hidden_deep=64, out_wide=32, out_deep=64, out_rnn=64)
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
        label = (label+0.01).to(device)
        # label = label.to(device)
        optimizer_w.zero_grad()
        outs = net(sparse_feature, global_feature, trip_feature, lengthes)
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
                "epoch: {}, step: {}, loss:{}, rmse: {}, mae: {}, step:{}, lr:{}, time:{}".format( e, step,
                                                                                         running_loss / 1000,
                                                                                         l2.item(),
                                                                                         l3.item(),
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
            # sparse_feature = sparse_feature.to(device)
            label = (label+0.01).to(device)
            label = label.to(device)
            outs = net(sparse_feature, global_feature, trip_feature, lengthes)

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
                       r'/DATA/CGW/graduate/model/WideDeepNet/model_backup/{}'.format(string_param))
            np.save(r'/DATA/CGW/graduate/model/WideDeepNet/model_backup/best_loss_{}.npy'.format(string_param),
                    np.array([best_loss]))
        print('test loss: {}, best loss: {}, rmse:{}, mae:{},  epoch time: {}'.format(final_loss, best_loss, final_rmse,
                                                                                         final_mae,
                                                                                  epoch_end - epoch_start))

# trans2vec 12 all feature test 0.1358123653745915, best loss: 0.1346242565719026, rmse:147.84530031826955, mae:103.92732728068837,  epoch time: 120.31114053726196
# node2vec 12 all feature test loss: 0.1348523818459428, best loss: 0.1348523818459428, rmse:151.53499397250724, mae:105.70701078346394,  epoch time: 119.43715214729309
# deepwalk 12 all feature test loss: 0.13616310864498715, best loss: 0.13598878756077262, rmse:149.54396856362953, mae:104.55296473723365,  epoch time: 119.90740942955017


# xa trans2vec all_feature test loss: 0.04989019716563432, best loss: 0.0351312346149074, rmse:49.93454829506252, mae:38.07156150237374,  epoch time: 2.714294910430908
# xa node2vec all feature test loss: 0.05805244709810485, best loss: 0.041642228019950184, rmse:55.309480656748235, mae:46.134706455728285,  epoch time: 2.565608024597168
# xa deepwalk all feature test loss: 0.036736237614051155, best loss: 0.036736237614051155, rmse:35.801749167235, mae:26.654098604036413,  epoch time: 2.6116976737976074