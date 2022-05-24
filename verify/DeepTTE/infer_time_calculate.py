import torch
import torch.optim as optim
from dataset import DiDiData
from utilis import fold_cross, rmse, mae
from rnndeepnet_d import edn, mape, NoamOpt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np

data_length = 8443  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
index = [i for i in range(data_length)]
dataset = DiDiData(source_data=r'/home/dc/CGW/train_processed_1024/', index=index,
                   device=device)
dataloader = DataLoader(dataset, shuffle=False, batch_size=None, num_workers=16, prefetch_factor=32,
                        persistent_workers=False)

net = edn(in_feature_d=32, in_feature_s=33, hidden_unit_d=64, hidden_unit_s=128, out_feature_d=32)

net.to(device)

net.load_state_dict(torch.load('/mnt/DATA2/CGW/滴滴比赛/run_model/experments/DeepTTE_1022_backup/1'))

result = []

cut_size = 300

with torch.no_grad():
    net.eval()
    for i in tqdm(dataloader):
        global_feature, trip_feature, sparse_feature, lengthes, label = i
        lengthes = np.array(lengthes)

        gf = global_feature.split(cut_size, dim=0)
        tf = trip_feature.split(cut_size, dim=1)
        # sf = sparse_feature.split(cut_size, dim=0)

        for j in range(len(gf)):
            if gf[j].size()[0] != cut_size:
                break
            g = gf[j].to(device)
            t = tf[j].to(device)
            # s = sf[j].to(device)
            length = lengthes[0:cut_size]
            lengthes = lengthes[cut_size:]

            start = time.time()
            outs = net(g, t, length)
            end = time.time()

            result.append([max(length), min(length), end - start])
np.save('/mnt/DATA2/CGW/滴滴比赛/run_model/experments/DeepTTE_1022_backup/infer_time5_{}.npy'.format(cut_size), np.array(result))
