import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from utilis import org_rnn_input

"""
    To add the condition as the other task for rnn
    And to cat the global feature (the output of deepnet) to the input of RNN
    And cat the output of attention with global feature
    And add the skip connection
    
    vision 2: to get deep rnn inplace wide rnn
    vision 3: to utilize the 0621_ dataset
    
    version d:
    To cat the raw global feature with the input of RNN
    To add more features (e.g. length and ratio)
    To take the init of GRU
"""


class deepnet(nn.Module):
    def __init__(self, in_feature_d, hidden_unit, out_feature_d):
        super(deepnet, self).__init__()
        # self.in_feature_d = in_feature_d
        # self.hidden_unit = hidden_unit
        # self.out_feature_d = out_feature_d
        # self.relu = nn.LeakyReLU(inplace=True)
        # self.linear_1 = nn.Linear(in_feature_d, hidden_unit)
        # self.linear_2 = nn.Linear(hidden_unit, hidden_unit)
        # self.linear_3 = nn.Linear(hidden_unit, out_feature_d)
        # self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        """ To learn the global features including time, weather , ...
            using simple full connected layer
            the normalization layer should to be considered
        :param x:
            Input tensor with shape [batch_size, feature]
            represent the several global features
        :return:
        """
        # x = self.relu(self.linear_1(x))
        # residual = x
        # # x = self.dp(x)
        # # * before the dropout or not
        # x = self.relu(self.linear_2(x) + residual)
        # residual = x
        # # x = self.dp(x)
        # x = self.relu(self.linear_3(x) + residual)
        return x


class seqnet(nn.Module):
    def __init__(self, in_feature_s, hidden_unit_s, in_feature_d):
        super(seqnet, self).__init__()
        self.in_feature_s = in_feature_s
        self.hidden_unit_s = hidden_unit_s
        self.in_feature_d = in_feature_d
        self.GRU = nn.GRU(input_size=self.in_feature_s + self.in_feature_d, hidden_size=self.hidden_unit_s,
                          num_layers=2, batch_first=False, dropout=0.3)

    def forward(self, seq_x, deep_x, length):
        """ To learn the trip features with the road segment embeddings and the outputs of deepnet
        :param x:
            Input tensor with shape []
            represent the feature of trip (road segment, current status)
        :param deep_feature:
            Input tensor with shape [batch_size, deep_feature]
            represent the learned features of global feature by deepnet
        :param length:
            Input
            represent the seq length
        :return:
        """
        seq_x = org_rnn_input(seq_x, deep_x, length)
        # batch_first = False
        seq_x, h_n = self.GRU(seq_x)
        seq_x = pad_packed_sequence(seq_x)[0]
        # To split the data along the dimension of seq
        seq_x = torch.chunk(seq_x, chunks=seq_x.shape[0], dim=0)
        return seq_x


class attention(nn.Module):
    def __init__(self, in_dim, at_dim):
        super(attention, self).__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=at_dim)
        self.sofmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """ To calculate the attention between the trip features and global features
        :param query:
            with shape [batch_size, feature]
        :param context:
            with shape (seq, batch_size, at_feature)
            (batch_size, seq, at_feature)
        :return:
        """
        context = context.permute(1, 0, 2).contiguous()
        query = self.tanh(self.linear(query)).unsqueeze(-1)
        # query with shape [batch_size, at_feature, 1]
        attention_scores = torch.bmm(context, query)
        # attention_scores with shape [batch_size, seq, 1] the third dimension is scores
        attention_scores = attention_scores.squeeze(-1)
        # * The padded is 0 but the negative number is exist, that need to masked?
        attention_scores = self.sofmax(attention_scores).unsqueeze(1)
        # attention_scores with shape [batch_size, 1, seq]
        values = torch.bmm(attention_scores, context)
        values = values.squeeze(1)
        return values


class outnet(nn.Module):
    def __init__(self, out_feature_d, hidden_s):
        super(outnet, self).__init__()
        self.attention = attention(out_feature_d, hidden_s)
        self.linear = nn.Linear(out_feature_d + hidden_s, 1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, deep_out, seq_out_list):
        """

        :param deep_out:
            the output of deepnet with shape [batch_size, in_feature_1]
        :param seq_out_list:
            the output of each step of seqnet with shape list([batch_size, in_feature_2]), len=seq
        :return:
        """
        context = torch.cat(seq_out_list, dim=0)
        values = self.attention(deep_out, context)
        # To cat the feature of seq_model and deep_model
        x = torch.cat([values, deep_out], dim=1)
        x = self.linear(x)
        return x


class edn(nn.Module):
    def __init__(self, in_feature_d, hidden_unit_d, out_feature_d, in_feature_s, hidden_unit_s):
        super(edn, self).__init__()
        self.dn = deepnet(in_feature_d, hidden_unit_d, out_feature_d)
        self.sn = seqnet(in_feature_s, hidden_unit_s, in_feature_d)
        self.on = outnet(out_feature_d, hidden_unit_s)

    def forward(self, x_deep, x_seq, length):
        out_deep = self.dn(x_deep)
        # *** dose the deepout which as the input of seqnet need to be differentiable ?
        out_seq = self.sn(x_seq, x_deep, length)
        out = self.on(out_deep, out_seq)
        return out

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