import torch
import torch.optim as optim
# from dataset import DiDiData
from data.dataset import DiDiData
from wdn import WideDeepNet, mape, NoamOpt
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
print('wdn')
string_param = 'trans2vec'
data_length = 8443  # batch数量
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
# torch.cuda.empty_cache()
index = [i for i in range(data_length)]
dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=index,
                         )
dataloader = DataLoader(dataset, shuffle=False,batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=False)

net = WideDeepNet(in_wide=22, in_deep=32, in_rnn=13, v_wide=8, hidden_wide=16, hidden_deep=128, out_wide=32,
                  out_deep=128, out_rnn=128)
net.to(device)

net.load_state_dict(torch.load('/DATA/CGW/graduate/model/WideDeepNet/model_backup/{}12_allfeature'.format(string_param)))


result = []
criterion_regression = mape()

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        sparse_feature = sparse_feature.to(device)
        outs = net(sparse_feature, global_feature, trip_feature, lengthes)
        result.append([label.detach().numpy(),outs.cpu().detach().numpy()])
np.save('/DATA/CGW/graduate/model/WideDeepNet/result/label_out_{}.npy'.format(string_param), np.array(result))