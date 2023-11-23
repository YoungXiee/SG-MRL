from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import numpy as np
import math

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class layer_block(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(layer_block, self).__init__()
        self.conv_output = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1), padding=(0, int( (k_size-1)/2 ) ) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1) )
        self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2))
        self.relu = nn.ReLU()
        
        
    def forward(self, input):
        conv_output = self.conv_output(input) # shape (B, D, N, T)

        conv_output1 = self.conv_output1(input)
        
        output = self.output(conv_output1)

        return self.relu( output+conv_output[...,-output.shape[3]:] )

        # return self.relu( conv_output )


class multi_scale_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, seq_length, layer_num, kernel_set, layer_norm_affline=True):
        super(multi_scale_block, self).__init__()

        self.seq_length = seq_length
        self.layer_num = layer_num
        self.norm = nn.ModuleList()
        self.scale = nn.ModuleList()

        for i in range(self.layer_num):
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))
        #     # self.norm.append(LayerNorm((c_out, num_nodes, int(self.seq_length/2**i)),elementwise_affine=layer_norm_affline))
        #     self.norm.append(LayerNorm((c_out, num_nodes, length_set[i]),elementwise_affine=layer_norm_affline))
        
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))

        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))

        for i in range(1, self.layer_num):
            
            self.scale.append(layer_block(c_out, c_out, kernel_set[i]))

        
    def forward(self, input, idx): # input shape: B D N T

        self.idx = idx

        scale = []
        scale_temp = input
        
        scale_temp = self.start_conv(scale_temp)
        # scale.append(scale_temp)
        for i in range(self.layer_num):
            scale_temp = self.scale[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp, self.idx)

            # scale.append(scale_temp[...,-self.k:])
            scale.append(scale_temp)

        return scale


class gated_fusion(nn.Module):
    def __init__(self, skip_channels, layer_num, ratio=1):
        super(gated_fusion, self).__init__()
        # self.reduce = torch.mean(x,dim=2,keepdim=True)
        self.dense1 = nn.Linear(in_features=skip_channels*(layer_num+1), out_features=(layer_num+1)*ratio, bias=False)
         
        self.dense2 = nn.Linear(in_features=(layer_num+1)*ratio, out_features=(layer_num+1), bias=False)


    def forward(self, input1, input2):

        se = torch.mean(input1, dim=2, keepdim=False)
        se = torch.squeeze(se)

        se = F.relu(self.dense1(se))
        se = F.sigmoid(self.dense2(se))

        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)
        se = torch.unsqueeze(se, -1)

        x = torch.mul(input2, se)
        x = torch.mean(x, dim=1, keepdim=False)
        return x

class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)

        ho = torch.cat(out,dim=1)

        ho = self.mlp(ho)

        return ho


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device, alpha=3):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num
        
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(dim,dim))
            self.lin2.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        

    def forward(self, idx, scale_idx, scale_set):
        
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        adj_set = []

        for i in range(self.layers):
            nodevec1 = torch.tanh(self.alpha*self.lin1[i](nodevec1*scale_set[i]))
            nodevec2 = torch.tanh(self.alpha*self.lin2[i](nodevec2*scale_set[i]))
            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj0 = F.relu(torch.tanh(self.alpha*a))
            
        
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj0.topk(self.k,1)
            mask.scatter_(1,t1,s1.fill_(1))
            # print(mask)
            adj = adj0*mask
            adj_set.append(adj)


        return adj_set
    
class graph_constructor_dynamic(nn.Module):
    def __init__(self, nnodes, k, dim, layer_num, device, alpha=3):
        super(graph_constructor_dynamic, self).__init__()
        self.nnodes = nnodes
        self.layers = layer_num
        
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.emb3 = nn.Embedding(nnodes, dim)
        self.emb4 = nn.Embedding(nnodes, dim)

        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        self.lin3 = nn.ModuleList()
        self.lin4 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(dim,dim))
            self.lin2.append(nn.Linear(dim,dim))
            self.lin3.append(nn.Linear(dim,dim))
            self.lin4.append(nn.Linear(dim,dim))

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

        self.W1 = nn.Linear(168, dim)
        self.W2 = nn.Linear(nnodes, dim)
        

    def forward(self, idx, scale_set, x):
        original_data = x.squeeze()  # B x N x T
        variable =  self.W1(original_data).transpose(1,2)  # variable [B, F, N]
        DI = torch.relu(self.W2(torch.relu(variable)))    # DI [B, F, F]
        
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        nodevec3 = self.emb3(idx)
        nodevec4 = self.emb4(idx)  

        adj_set = []

        for i in range(self.layers):
            nodevec1 = torch.tanh(self.alpha*self.lin1[i](nodevec1*scale_set[i]))
            nodevec2 = torch.tanh(self.alpha*self.lin2[i](nodevec2*scale_set[i]))
            nodevec3 = torch.tanh(self.alpha*self.lin3[i](nodevec3*scale_set[i]))
            nodevec4 = torch.tanh(self.alpha*self.lin4[i](nodevec4*scale_set[i]))

            MD1 = torch.tanh(self.alpha*(nodevec3 @ DI)) # [B, N, F]
            MD2 = torch.tanh(self.alpha*(nodevec4 @ DI))
            MD1_trans = MD1.transpose(1,2)
            MD2_trans = MD2.transpose(1,2)
            a1 = torch.matmul(MD1, MD2_trans)
            a2 = torch.matmul(MD2, MD1_trans)

            a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
            adj_static = F.relu(torch.tanh(self.alpha*a))
            adj_dynamic = F.relu(torch.tanh(self.alpha*(a1 - a2)))
            adj = F.relu(torch.tanh(adj_static + adj_dynamic))
            adj = torch.mean(adj, dim=0)
        
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1,t1 = adj.topk(self.k,1)
            mask.scatter_(1,t1,s1.fill_(1))
            # print(mask)
            adj = adj*mask
            adj_set.append(adj)


        return adj_set

class Hypergraph_construction(nn.Module):
    def __init__(self, node_dim):
        super(Hypergraph_construction,self).__init__()
        self.hidden = node_dim
        self.alpha = 3
        self.emb = nn.Linear(1, self.hidden)
        self.lin1 = nn.Linear(1, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)


    def forward(self, x, idx):  # [B, T, N, F]
        x = x.reshape(-1, x.size(2), x.size(3)) # [B x T, N, F]

        cluster_emb = self.emb(idx.float())  # [B, K, F]
        cluster_emb = torch.unsqueeze(cluster_emb,dim=0) 

        MS1 = torch.tanh(self.alpha*(self.lin1(x)))   # [B x T, N, F]
        MS2 = torch.tanh(self.alpha*(self.lin2(cluster_emb)))  # [1, K, F]

        HEs = F.relu(torch.tanh(MS1 @ (MS2.transpose(1,2))))

        return HEs  #[N, K]

class HypergraphLearing(nn.Module):
    def __init__(self, num_nodes, node_dim, device, K):
        super(HypergraphLearing, self).__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.K = K
        self.device = device
        self.edge_clf = torch.randn(self.node_dim, self.K) / math.sqrt(self.K)
        self.edge_clf = nn.Parameter(self.edge_clf, requires_grad=True)
        self.edge_map = torch.randn(self.K, self.K) / math.sqrt(self.K)
        self.edge_map = nn.Parameter(self.edge_map, requires_grad=True)

        # self.K_idx = torch.arange(self.K).unsqueeze(1).to(self.device)
        self.hypergraph_construction = Hypergraph_construction(self.node_dim).to(self.device)

    def forward(self, x, HEs):  # B x D x N x T
        # HEs = self.hypergraph_construction(x, self.K_idx).to(self.device) # [N, K]
        HEs = torch.unsqueeze(HEs, dim=0)  # 1 x N x K
        #使用自建的超图
        feat = x.transpose(1,3)  # B x T x N x D
        feat = feat.reshape(-1, feat.size(2), feat.size(3)) # (B x T) x N x D
        hyper_feat = HEs.transpose(1,2) @ feat # (B x T) x K x D
        hyper_feat_mapped = F.relu(self.edge_map @ hyper_feat) # (B x T) x K x D
        hyper_out = hyper_feat_mapped + hyper_feat 
        y = F.relu(HEs @ hyper_out) # (B x T) x N x D
        y = y.reshape(x.size(0), x.size(3), x.size(2), x.size(1)) # B x T x N x D
        x_trans = x.transpose(1,3)  # B x T x N x D
        layernorm = nn.LayerNorm(x_trans.size(3)).to(self.device)
        y_final = layernorm(y + x_trans)
        y_final = y_final.transpose(1,3)  # B x D x N x T
        return y_final
    

class multi_Hypergraph_construction(nn.Module):
    def __init__(self, node_dim, layer_num):
        super(multi_Hypergraph_construction,self).__init__()
        self.hidden = node_dim
        self.alpha = 3
        self.layer_num = layer_num
        self.emb = nn.Linear(1, self.hidden)
        self.lin1 = nn.ModuleList()
        self.lin2 = nn.ModuleList()
        for i in range(layer_num):
            self.lin1.append(nn.Linear(16, self.hidden))
            self.lin2.append(nn.Linear(self.hidden, self.hidden))


    def forward(self, idx, scale):  # scale [B, F, N, T]
        for i in range(self.layer_num):
            scale[i] = scale[i].transpose(1,3)  # [B, T, N, F]
            scale[i] = scale[i].reshape(-1, scale[i].size(2), scale[i].size(3)) # [B x T, N, F]

        cluster_emb = self.emb(idx.float())  # [B, K, F]
        cluster_emb = torch.unsqueeze(cluster_emb,dim=0) 

        HEs_set = []

        for j in range(self.layer_num):
            MS1 = torch.tanh(self.alpha*(self.lin1[j](scale[j])))   # [B x T, N, F = hidden]
            MS2 = torch.tanh(self.alpha*(self.lin2[j](cluster_emb)))  # [1, K, F]
            HEs = F.relu(torch.tanh(MS1 @ (MS2.transpose(1,2))))
            HEs = torch.mean(HEs, dim=0)
            HEs_set.append(HEs)

        return HEs_set  #[N, K]