import torch
import torch.nn as nn
from torch.nn import GRU, Linear


class DNNnet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(DNNnet, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        residual = x
        # * before the dropout or not
        x = self.relu(self.fc2(x) + residual)
        residual = x
        x = self.relu(self.fc3(x) + residual)
        return x


class GRUDNNnet(nn.Module):
    def __init__(self, input_size, g_input, g_hidden, hidden_size, g_out):
        super(GRUDNNnet, self).__init__()

        self.gru = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=False,
                       dropout=0.3)
        self.dnn = DNNnet(input_size=g_input, hidden_size=g_hidden, out_size=g_out)
        self.linear = Linear(hidden_size + g_out, 1)

    def forward(self, trip_feature, global_feature):
        packed_out, h_n = self.gru(trip_feature)
        h_n = h_n[-1]
        g = self.dnn(global_feature)
        x = torch.cat([h_n, g], dim=-1)
        return self.linear(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.GRU):
                nn.init.orthogonal_(m.weight_ih_l0)
                nn.init.orthogonal_(m.weight_hh_l0)
                nn.init.orthogonal_(m.weight_ih_l1)
                nn.init.orthogonal_(m.weight_hh_l1)
                nn.init.zeros_(m.bias_ih_l0)
                nn.init.zeros_(m.bias_hh_l0)
                nn.init.zeros_(m.bias_ih_l1)
                nn.init.zeros_(m.bias_hh_l1)


class mape(nn.Module):
    def __init__(self):
        super(mape, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.mean(torch.abs((outs - labels) / labels))
        return loss
