import torch
import torch.nn as nn
from torch.nn import GRU, Linear


class GRUnet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUnet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=False,
                       dropout=0.3)
        self.linear = Linear(hidden_size, 1)

    def forward(self, x):
        packed_out, h_n = self.gru(x)
        h_n = h_n[-1]
        return self.linear(h_n)

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
