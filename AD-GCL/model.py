import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))

        logits = torch.cat(tuple(scs))

        return logits

class Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.gcn = GCN(n_in, n_h, activation)

        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round)

    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn(seq1, adj, sparse)

        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])

        ret = self.disc(c, h_mv)

        return ret, h_mv, c


class Gene(nn.Module):
    def __init__(self, nb_nodes, hid_dim, out_dim):
        super(Gene, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.layer1 = GraphConv(hid_dim, hid_dim)
        self.layer2 = GraphConv(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm = nn.BatchNorm1d(out_dim)

        self.epsilon = torch.nn.Parameter(torch.Tensor(nb_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.epsilon, 0.5)
        
    def forward(self, g, x, adj):
        h1 = self.fc(x)
        h2 = F.relu(self.layer1(g, x))
        h2 = self.layer2(g, h2)

        h = (1 - self.epsilon.view(-1,1)) * h1 + self.epsilon.view(-1,1) * h2

        ret = (torch.mm(h, h.t()) + torch.mm(x, x.t())) / 2

        h = self.batchnorm(h)

        return ret, h

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, g, features):
        g = g.local_var()
        g.ndata['h'] = features
        g.update_all(message_func=dgl.function.copy_src(src='h', out='m'), reduce_func=dgl.function.sum(msg='m', out='h'))
        h = g.ndata['h']
        return self.linear(h)

class Disc(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(Disc, self).__init__()
        self.f_k = nn.Bilinear(hid_dim, hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        logits = self.f_k(x1, x2)
        logits = self.sigmoid(logits)

        return logits