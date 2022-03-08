import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from set2set import Set2Set


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
            self.dropout_layer1 = nn.Dropout(p=0.5)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def nomalize_adj(self, adj):
        D = torch.sum(adj[1], 1)
        D1 = torch.diag(torch.pow(D, D.new_full((adj.size(1),),-0.5))).cuda()
        return torch.matmul(torch.matmul(D1, adj), D1)

    def preprocess_support(self, adj):
        In =  torch.ones((adj.size(1),))
        adj_normalized = self.nomalize_adj(torch.add(adj, torch.diag(In).cuda()))
        return adj_normalized

    def forward_na(self, x, adj):
        adj = self.preprocess_support(adj)
        # x = torch.cat([torch.t(x[i]).unsqueeze(0) for i in range(batch_size)])
        if self.dropout > 0.001:
            x = self.dropout_layer1(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims,
                                                 label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)


        if self.bn:
            x = self.apply_bn(x)
        if self.concat:
            x_all = [x]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            if self.concat:
                x_all.append(x)
        x = conv_last(x, adj)
        if self.concat:
            x_all.append(x)
            # x_tensor: [batch_size x num_nodes x embedding]
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
        # print(x_tensor.shape)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def read_forward(self, x, adj, conv_first, conv_block, conv_last, batch_size):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first.forward_na(x, adj)
        x = self.act(x)
        x1 = x
        if self.bn:
            x = self.apply_bn(x)

        for i in range(len(conv_block)):
            x = conv_block[i].forward_na(x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)

        x = conv_last.forward_na(x, adj)

        x_tensor = x

        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred

    def loss(self, pred, label, type='softmax'):
        # softmax + CE
        if type == 'softmax':
            return F.cross_entropy(pred, label, size_average=True)
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        out = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        ypred = self.pred_model(out)
        return ypred

class Gcn(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], gcn_concat=True, pool_concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(Gcn, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=gcn_concat,
                                                    bn=bn, args=args)
        add_self = not gcn_concat
        self.num_pooling = num_pooling
        self.pool_concat = pool_concat
        self.linkpred = linkpred
        self.assign_ent = True
        self.adj = nn.Parameter(torch.FloatTensor(args.batch_size, max_num_nodes, max_num_nodes).cuda())

        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            # print(self.pred_input_dim)
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        # assign_dims = []
        # if assign_num_layers == -1:
        #     assign_num_layers = num_layers
        # if assign_input_dim == -1:
        #     assign_input_dim = input_dim
        #
        # self.assign_conv_first_modules = []
        # self.assign_conv_block_modules = []
        # self.assign_conv_last_modules = []
        # self.assign_pred_modules = []
        # assign_dim = int(max_num_nodes * assign_ratio)
        # for i in range(num_pooling):
        #     assign_dims.append(assign_dim)
        #     self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
        #         assign_input_dim if i==0 else self.pred_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
        #         normalize=True)
        #     assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if gcn_concat else assign_dim
        #     print(assign_pred_input_dim)
        #     self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1).cuda()
        #
        #     # next pooling layer
        #     assign_input_dim = embedding_dim
        #     assign_dim = int(assign_dim * assign_ratio)
        #
        #     self.assign_conv_first_modules.append(self.assign_conv_first)
        #     self.assign_conv_block_modules.append(self.assign_conv_block)
        #     self.assign_conv_last_modules.append(self.assign_conv_last)
        #     self.assign_pred_modules.append(self.assign_pred)

        # self.pred_model = self.build_pred_layers(
        #     hidden_dim * (num_pooling)+embedding_dim if pool_concat else embedding_dim, pred_hidden_dims,
        #     label_dim, num_aggs=self.num_aggs)
        self.pred_model = self.build_pred_layers(
            hidden_dim * (num_pooling + 1)+embedding_dim if pool_concat else embedding_dim, pred_hidden_dims,
            label_dim, num_aggs=self.num_aggs)  #三层

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)
    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        # self.adj.data = adj

        # mask
        max_num_nodes = self.adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []


        # if x is not None:
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        if self.concat:
            x_all = [x]
        #
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        #
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            if self.concat:
                x_all.append(x)
            #
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            #
        x = self.conv_last(x, adj)
        #
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        #



        # embedding_tensor = self.gcn_forward(x, self.adj,
        #                                     self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        # self.embedding_tensor = embedding_tensor
        #
        # out, _ = torch.max(embedding_tensor, dim=1)
        # # print(out.shape)
        # out_all.append(out)
        # if self.num_aggs == 2:
        #     out = torch.sum(embedding_tensor, dim=1)
        #     out_all.append(out)
        #
        # for i in range(self.num_pooling):
        #     if batch_num_nodes is not None and i == 0:
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #     else:
        #         embedding_mask = None
        #     # print(x_a.shape)
        #     self.assign_tensor = self.gcn_forward(x_a, adj,
        #                                           self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
        #                                           self.assign_conv_last_modules[i],
        #                                           embedding_mask)
        #     # print(self.assign_tensor.shape)
        #
        #     # [batch_size x num_nodes x next_lvl_num_nodes]
        #     self.assign_tensor = nn.Softmax(dim=-1)((self.assign_pred_modules[i])(self.assign_tensor))
        #     if embedding_mask is not None:
        #         self.assign_tensor = self.assign_tensor * embedding_mask
        #
        #     # update pooled features and adj matrix
        #     x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
        #     adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
        #     x_a = x
        #
        #     embedding_tensor = self.gcn_forward(x, adj,
        #                                         self.conv_first_after_pool[i], self.conv_block_after_pool[i],
        #                                         self.conv_last_after_pool[i])
        #
        #     out, _ = torch.max(embedding_tensor, dim=1)
        #     out_all.append(out)
        #     # print(out.shape)
        #     if self.num_aggs == 2:
        #         # out = torch.mean(embedding_tensor, dim=1)
        #         out = torch.sum(embedding_tensor, dim=1)
        #         out_all.append(out)

        if self.pool_concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        # print(output.shape)
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(Gcn, self).loss(pred, label)
        # if self.linkpred:
        #     max_num_nodes = adj.size()[1]
        #     pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
        #     tmp = pred_adj0
        #     pred_adj = pred_adj0
        #     for adj_pow in range(adj_hop - 1):
        #         tmp = tmp @ pred_adj0
        #         pred_adj = pred_adj + tmp
        #     pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
        #     # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
        #     # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
        #     # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
        #     self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
        #     if batch_num_nodes is None:
        #         num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
        #         print('Warning: calculating link pred loss without masking')
        #     else:
        #         num_entries = np.sum(batch_num_nodes * batch_num_nodes)
        #         embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        #         adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
        #         self.link_loss[1 - adj_mask.byte()] = 0.0
        #
        #     self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        #     # print('linkloss: ', self.link_loss)
        #     return loss + self.link_loss
        return loss

    def update_adj(self):
        return self.adj


class Gcn_EGC(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1, batch_size=10,
                 pred_hidden_dims=[50], gcn_concat=True, pool_concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(Gcn_EGC, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=gcn_concat,
                                                    bn=bn, args=args)
        add_self = not gcn_concat
        self.batch_size = batch_size
        self.hidden = hidden_dim
        self.num_pooling = num_pooling
        self.pool_concat = pool_concat
        self.linkpred = linkpred
        self.ratio = assign_ratio
        self.assign_ent = True
        self.adj = nn.Parameter(torch.FloatTensor(args.batch_size, max_num_nodes, max_num_nodes).cuda())
        assign_dim = int(max_num_nodes * assign_ratio)
        #read_GC
        self.read_first, self.read_block, self.read_last = self.build_conv_layers(
            max_num_nodes, self.hidden, 1, num_layers,
            add_self, normalize=False, dropout=dropout)
        self.read_first_after_pool, self.read_block_after_pool, self.read_last_after_pool = self.build_conv_layers(
            max_num_nodes, self.hidden, 1, num_layers,
            add_self, normalize=False, dropout=dropout)

        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            # print(self.pred_input_dim)
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        self.pred_model = self.build_pred_layers(
            hidden_dim * (num_pooling + 1)+embedding_dim if pool_concat else embedding_dim, pred_hidden_dims,
            label_dim, num_aggs=self.num_aggs)  #三层

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def get_fea_A(self, x, batch_size, ratio):

        x_sim = torch.matmul(torch.cat([torch.t(x[i]).unsqueeze(0) for i in range(batch_size)]), torch.cat([F.normalize(x[i]).unsqueeze(0) for i in range(batch_size)]))
        x_sim = torch.cat([((torch.t(x_sim[i])+x_sim[i])/2).unsqueeze(0) for i in range(batch_size)])


        node_num = int(x_sim.size(2) * x_sim.size(2) * ratio)
        x_sim_sort = [x_sim[i].view(-1).sort(dim=0, descending=True)[0] for i in range(batch_size)]
        num = [x_sim_sort[i][node_num] for i in range(batch_size)]
        fea_A = torch.cat([torch.sign(F.relu(x_sim[i] - num[i])).unsqueeze(0) for i in range(batch_size)])
        return fea_A

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        # self.adj.data = adj

        # mask
        max_num_nodes = self.adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.out_all = []


        # if x is not None:
        x = self.conv_first(x, adj)
        x = self.act(x)

        self.x1 = x
        self.x1_ = torch.cat([torch.t(self.x1[i]).unsqueeze(0) for i in range(self.batch_size)])

        self.fea_A1 = self.get_fea_A(self.x1, self.batch_size, self.ratio)
        # self.x1_111 = [self.x1_[i].cpu().detach().numpy() for i in range(self.batch_size)]
        # self.fea_A11 = [self.fea_A1[i].cpu().detach().numpy() for i in range(self.batch_size)]
        self.out1 = self.read_forward(self.x1_, self.fea_A1, self.read_first, self.read_block, self.read_last, embedding_mask).squeeze(2)
        self.out_all.append(self.out1)

        if self.bn:
            x = self.apply_bn(x)
        # if self.concat:
        #     x_all = [x]
        #
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        #
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, adj)
            x = self.act(x)

            self.x2 = x
            self.x2_ = torch.cat([torch.t(self.x2[i]).unsqueeze(0) for i in range(self.batch_size)])
            # self.a21 = [self.x2[i].cpu().detach().numpy() for i in range(10)]
            # self.a2 = [self.x2_[i].cpu().detach().numpy() for i in range(10)]

            self.fea_A2 = self.get_fea_A(self.x2, self.batch_size, self.ratio)

            # self.a22 = [self.fea_A2[i].cpu().detach().numpy() for i in range(10)]
            self.out2 = self.read_forward(self.x2_, self.fea_A2, self.read_first_after_pool, self.read_block_after_pool, self.read_last_after_pool,
                                          embedding_mask).squeeze(2)
            self.out_all.append(self.out2)

            if self.bn:
                x = self.apply_bn(x)
            # if self.concat:
            #     x_all.append(x)
            #
            out, _ = torch.max(x, dim=1)
            # out_all.append(out)
            #
        x = self.conv_last(x, adj)
        #
        out, _ = torch.max(x, dim=1)
        self.out_all.append(out)
        #

        if self.pool_concat:
            output = torch.cat(self.out_all, dim=1)
        else:
            output = out
        # print(output.shape)
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(Gcn_EGC, self).loss(pred, label)

        return loss

    def update_adj(self):
        return self.adj



class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 pred_hidden_dims=[50], gcn_concat=True, pool_concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                                                    num_layers, pred_hidden_dims=pred_hidden_dims, concat=gcn_concat,
                                                    bn=bn, args=args)
        add_self = not gcn_concat
        self.num_pooling = num_pooling
        self.pool_concat = pool_concat
        self.linkpred = linkpred
        self.assign_ent = True
        self.adj = nn.Parameter(torch.FloatTensor(args.batch_size, max_num_nodes, max_num_nodes).cuda())

        # GC
        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            # print(self.pred_input_dim)
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(
                assign_input_dim if i==0 else self.pred_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if gcn_concat else assign_dim
            print(assign_pred_input_dim)
            self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1).cuda()

            # next pooling layer
            assign_input_dim = embedding_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)
            self.assign_pred_modules.append(self.assign_pred)

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim * (num_pooling + 1) if pool_concat else self.pred_input_dim, pred_hidden_dims,
            label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x
        self.adj.data = adj

        # mask
        max_num_nodes = self.adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # self.assign_tensor = self.gcn_forward(x_a, adj,
        #        self.assign_conv_first_modules[0], self.assign_conv_block_modules[0], self.assign_conv_last_modules[0],
        #        embedding_mask)
        ## [batch_size x num_nodes x next_lvl_num_nodes]
        # self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
        # if embedding_mask is not None:
        #    self.assign_tensor = self.assign_tensor * embedding_mask
        # [batch_size x num_nodes x embedding_dim]

        embedding_tensor = self.gcn_forward(x, self.adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        self.embedding_tensor = embedding_tensor

        out, _ = torch.max(embedding_tensor, dim=1)
        # print(out.shape)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            # print(x_a.shape)
            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # print(self.assign_tensor.shape)

            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)((self.assign_pred_modules[i])(self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            # update pooled features and adj matrix
            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            # print(out.shape)
            if self.num_aggs == 2:
                # out = torch.mean(embedding_tensor, dim=1)
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.pool_concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        # print(output.shape)
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            # print('adj1', torch.sum(pred_adj0) / torch.numel(pred_adj0))
            # print('adj2', torch.sum(pred_adj) / torch.numel(pred_adj))
            # self.link_loss = F.nll_loss(torch.log(pred_adj), adj)
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('Warning: calculating link pred loss without masking')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1 - adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            # print('linkloss: ', self.link_loss)
            return loss + self.link_loss
        return loss

    def update_adj(self):
        return self.adj