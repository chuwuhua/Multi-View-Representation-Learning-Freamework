import torch
import torch.optim as optim
# from dataset import DiDiData
from data.dataset import DiDiData,index
from utilis import fold_cross, rmse, mae
from rnndeepnet_d import edn, mape, NoamOpt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np

print('deeptte')
string_param = 'trans2vec'

data_length = 8443  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
index = [i for i in range(data_length)]
dataset = DiDiData(source_data=r'/DATA/CGW/graduate/data/trainingData', index=index)
dataloader = DataLoader(dataset, shuffle=False,batch_size=None, num_workers=16, prefetch_factor=32,
                              persistent_workers=False)

net = edn(in_feature_d=32, in_feature_s=13, hidden_unit_d=64, hidden_unit_s=128, out_feature_d=32)

net.to(device)

net.load_state_dict(torch.load('/DATA/CGW/graduate/model/DeepTTE/model_backup/{}12_allfeature'.format(string_param)))

result = np.empty((0,2))
criterion_regression = mape()

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        cur_result = []
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        gf = global_feature.split(1, dim=0)
        tf = trip_feature.split(1, dim=1)
        sf = sparse_feature.split(1, dim=0)
        lb = label.split(1, dim=0)

        for j in range(len(lengthes)):
            g = gf[j].to(device)
            t = tf[j].to(device)
            s = sf[j].to(device)
            length = [lengthes[j]]
            l = lb[j].to(device)
            outs = net(g,t, length)
            # t_loss = criterion_regression(outs, l)
            cur_result.append([lb[j][0][0],outs.cpu().detach().numpy()[0][0]])
        result = np.append(result, np.array(cur_result), axis=0)
    np.save('/DATA/CGW/graduate/model/DeepTTE/result/label_out_{}.npy'.format(string_param),np.array(result))