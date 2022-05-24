import torch
import torch.optim as optim
# from dataset import DiDiData
from data.dataset import DiDiData
from GRUDNNnet import GRUDNNnet,mape
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
print('gru-dnn')
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

net = GRUDNNnet(input_size=13,g_input=32,g_hidden=64,hidden_size=64,g_out=64)
net.to(device)

net.load_state_dict(torch.load('/DATA/CGW/graduate/model/GRU-DNN/model_backup/{}_allfeature'.format(string_param)))


result = []
criterion_regression = mape()

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        sparse_feature = sparse_feature.to(device)
        outs = net(trip_feature, global_feature)
        result.append([label.detach().numpy(),outs.cpu().detach().numpy()])
np.save('/DATA/CGW/graduate/model/GRU-DNN/result/label_out_{}.npy'.format(string_param), np.array(result))