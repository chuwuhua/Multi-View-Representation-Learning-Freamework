import torch
import torch.optim as optim
# from dataset import DiDiData
from data.dataset import DiDiData
from utilis import fold_cross, rmse, mae
from rnndeepnet_d import edn, mape, NoamOpt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
print('deeptte')
string_param = 'deepwalk'

data_length = 8443  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
index = [i for i in range(data_length)]
dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=index)

dataloader = DataLoader(dataset, shuffle=False, batch_size=None, num_workers=16, prefetch_factor=32,
                        persistent_workers=False)

net = edn(in_feature_d=32, in_feature_s=13, hidden_unit_d=64, hidden_unit_s=128, out_feature_d=32)

net.to(device)

net.load_state_dict(torch.load('/DATA/CGW/graduate/model/DeepTTE/model_backup/{}12_allfeature'.format(string_param)))

result = []

# cut_size = 300

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        global_feature = global_feature.to(device)
        trip_feature = trip_feature.to(device)
        lengthes = np.array(lengthes)

        # gf = global_feature.split(cut_size, dim=0)
        # tf = trip_feature.split(cut_size, dim=1)
        # sf = sparse_feature.split(cut_size, dim=0)


        outs = net(global_feature, trip_feature, lengthes)

        result.append([label.detach().numpy(),outs.cpu().detach().numpy()])
np.save('/DATA/CGW/graduate/model/DeepTTE/result/label_out_{}.npy'.format(string_param),np.array(result))
