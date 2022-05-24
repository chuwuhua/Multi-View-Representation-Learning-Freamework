import torch
import torch.optim as optim
from data.dataset import DiDiData,index
from data_xa.dataset_xa import XAData
from GRUnet import GRUnet,mape
from utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import numpy as np
print('gru')
string_param = 'trans2vec'
data_length = 8443  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
index = [i for i in range(data_length)]
dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=index,
                         )
dataloader = DataLoader(dataset, shuffle=False,batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=False)

net = GRUnet(input_size=13,hidden_size=128)
net.to(device)

net.load_state_dict(torch.load('/DATA/CGW/graduate/model/GRU-base/model_backup/{}_allfeature'.format(string_param)))

result = []
criterion_regression = mape()

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        sparse_feature = sparse_feature.to(device)
        label = label.to(device)
        outs = net(trip_feature)
        t_loss = criterion_regression(outs, label)
        result.append([t_loss.item(),max(lengthes),min(lengthes)])
np.save('/DATA/CGW/graduate/model/GRU-base/result/link_num_{}.npy'.format(string_param), np.array(result))