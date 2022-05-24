import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class CrossProduct(nn.Module):
    def __init__(self, x_dim, v_dim):
        """
            Cross production operation based on factorization machine
        :param x_dim:
        :param v_dim:
        """
        super(CrossProduct, self).__init__()
        self.x_dim = x_dim
        self.k = v_dim
        self.vparam = nn.Parameter(torch.FloatTensor(self.x_dim, self.k))
        self.reset_parameter()

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batchsize, x_dim)
        :return:
            Output tensor with shape (batch_size, 1)
        """
        # *? need to use sparse operation
        x = torch.diag_embed(x)
        x = torch.matmul(x, self.vparam)
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        x = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return 0.5 * x

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.vparam.size(1))
        self.vparam.data.uniform_(-stdv, stdv)


class CPLayer(nn.Module):
    def __init__(self, x_dim, v_dim, out_feature):
        super(CPLayer, self).__init__()
        self.in_feature = x_dim
        self.k = v_dim
        self.out_feature = out_feature
        self.cps = self.generate_cp()

    def generate_cp(self):
        cp_list = nn.ModuleList()
        for i in range(self.out_feature):
            cp_list.append(CrossProduct(x_dim=self.in_feature, v_dim=self.k))
        return cp_list

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batch_size, in_feature)
        :return:
            Output tensor with shape (batch_szie, out_feature)
        """
        x = torch.cat([per_cp(x) for per_cp in self.cps], dim=-1)
        return x


class WideNet(nn.Module):
    def __init__(self, x_dim, v_dim, out_feature_1, out_feature_2):
        super(WideNet, self).__init__()
        self.in_deature = x_dim
        self.k = v_dim
        self.out_feature_1 = out_feature_1
        self.out_feature_2 = out_feature_2
        self.cps = CPLayer(self.in_deature, self.k, self.out_feature_1)
        self.aff_transform = nn.Linear(self.out_feature_1, self.out_feature_2)

    def forward(self, x):
        """
        :param x:
            Input tensor with shape (batch_size, in_feature)
        :return:
            Output tensor with shape (batch_szie, out_feature)
        """
        x = self.cps(x)
        x = self.aff_transform(x)
        return x


class DeepNet(nn.Module):
    def __init__(self, in_feature, hidden_unit, out_feature):
        super(DeepNet, self).__init__()
        self.in_feature = in_feature
        self.hidden_unit = hidden_unit
        self.out_feature = out_feature
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(in_feature, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc3 = nn.Linear(hidden_unit, out_feature)

    def forward(self, x):
        """ To learn the dense global features including time, weather , ...
            using simple full connected layer
            the normalization layer should to be considered
        :param x:
            Input tensor with shape [batch_size, feature]
            represent the several global features
        :return:
        """
        x = self.relu(self.fc1(x))
        residual = x
        # * before the dropout or not
        x = self.relu(self.fc2(x) + residual)
        residual = x
        x = self.relu(self.fc3(x) + residual)
        return x


class RecurrentNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(RecurrentNet, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc = nn.Linear(in_feature, out_feature)
        self.relu = nn.LeakyReLU(inplace=True)
        self.gru = nn.GRU(input_size=self.out_feature, hidden_size=self.out_feature, num_layers=2, batch_first=True, dropout=0.3)

    def forward(self, x, length):
        """
        :param x:
            Input tensor with shape (seq, batch_size, feature)
        :return:
        """
        x = self.relu(self.fc(x))
        x = pack_padded_sequence(x, length, batch_first=False)
        _, x = self.gru(x)
        # x = x[-1].squeeze(0)
        x = x[-1]
        return x


class Regression(nn.Module):
    def __init__(self, dim_wide, dim_deep, dim_rnn):
        super(Regression, self).__init__()
        self.dim_wide = dim_wide
        self.dim_deep = dim_deep
        self.dim_rnn = dim_rnn
        self.fc = nn.Linear(self.dim_wide + self.dim_deep + self.dim_rnn, 1)

    def forward(self, x_wide, x_deep, x_rnn):
        x = torch.cat([x_wide, x_deep, x_rnn], dim=-1)
        x = self.fc(x)
        return x


class WideDeepNet(nn.Module):
    def __init__(self, in_wide, in_deep, in_rnn, v_wide, hidden_wide, hidden_deep, out_wide, out_deep, out_rnn):
        super(WideDeepNet, self).__init__()
        self.in_wide = in_wide
        self.in_deep = in_deep
        self.in_rnn = in_rnn
        self.k = v_wide
        self.hidden_wide = hidden_wide
        self.hidden_deep = hidden_deep
        self.out_wide = out_wide
        self.out_deep = out_deep
        self.out_rnn = out_rnn
        self.wide = WideNet(x_dim=self.in_wide, v_dim=self.k, out_feature_1=self.hidden_wide, out_feature_2=self.out_wide)
        self.deep = DeepNet(in_feature=self.in_deep, hidden_unit=self.hidden_deep, out_feature=self.out_deep)
        self.rnn = RecurrentNet(in_feature=self.in_rnn, out_feature=self.out_rnn)
        self.regression = Regression(dim_wide=self.out_wide, dim_deep=self.out_deep, dim_rnn=self.out_rnn)

    def forward(self, x_wide, x_deep, x_rnn, length):
        x_wide = self.wide(x_wide)
        x_deep = self.deep(x_deep)
        x_rnn = self.rnn(x_rnn, length)
        x = self.regression(x_wide, x_deep, x_rnn)
        return x

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


class NoamOpt:
    """
    Optim wrapper that control the learning rate (including warmup and decrease)
    """

    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self.__step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self.__lr = 0

    def step(self):
        """
        Replace optim.step()
        :return:
        """
        self.__step = self.__step + 1
        lr = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.__lr = lr
        self.optimizer.step()

    def zero_grad(self):
        """
        Replace optim.zero_grad()
        :return:
        """
        self.optimizer.zero_grad()

    def learning_rate(self, step=None):
        """
        Refresh the learning rate
        :param step:
            Auto generation
        :return:
        """
        if step is None:
            step = self.__step
        lr = self.factor * (self.d_model ** (-0.5)) * min((step ** (-0.5)), (step * (self.warmup ** (-1.5))))
        return lr

    def qurey(self):
        return self.__step, self.__lr