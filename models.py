import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, SpGraphAttentionLayer
import torch

class GVCLN(nn.Module):
    def __init__(self, nfeat, nclass, nhid_1, dropout_1, nhid_2, dropout_2, alpha_2, nheads_2):
        super(GVCLN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid_1)
        self.gc2 = GraphConvolution(nhid_1*3, nclass)
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid_2, dropout=dropout_2, alpha=alpha_2, concat=True) for _ in range(nheads_2)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid_2*nheads_2, nclass, dropout=dropout_2, alpha=alpha_2, concat=False)
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.b1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        self.b2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True).data.fill_(0.05).to(torch.device("cuda"))
        
    def forward(self, x, adj, idx_train, labels):
        y = F.relu(self.gc1(x, adj))
        y = torch.cat([y, y, y], dim=1)
        y = F.dropout(y, self.dropout_1, training=self.training)
        y = self.gc2(y, adj)
        semi_loss_1 = torch.nn.CrossEntropyLoss()(y[idx_train], labels[idx_train])

        z = F.dropout(x, self.dropout_2, training=self.training)
        z = torch.cat([att(z, adj) for att in self.attentions], dim=1)
        z = F.dropout(z, self.dropout_2, training=self.training)
        z = F.elu(self.out_att(z, adj))
        semi_loss_2 = torch.nn.CrossEntropyLoss()(z[idx_train], labels[idx_train])

        log_probs = self.logsoftmax(y)
        CL_loss_12 = (- F.softmax(z, dim=1).detach() * log_probs).mean(0).sum()

        loss_11 = semi_loss_1
        loss_21 = semi_loss_2

        loss_12 = semi_loss_1 + self.b1*CL_loss_12
        loss_22 = semi_loss_2 + self.b2*CL_loss_12
        return y, z, loss_11, loss_21, loss_12, loss_22