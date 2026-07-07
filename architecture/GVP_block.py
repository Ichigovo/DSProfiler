import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  
from torch_geometric.nn import MessagePassing
import torch_scatter
import numpy as np
from collections import defaultdict
import json
import itertools
import functools
from torch_scatter import scatter_add
import matplotlib.pyplot as plt
import random
import pandas as pd

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
           torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP_embedding(nn.Module):
    def __init__(self, in_dims, hidden_dims, edge_in_dims, edge_hidden_dims,
                 num_layers=3, drop_rate=0.1, vector_gate=False):
        super(GVP_embedding, self).__init__()

        self.num_layers = num_layers
        self.drop_rate = drop_rate

        self.W_v = nn.Sequential(
            LayerNorm(in_dims),
            GVP(in_dims, hidden_dims, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dims),
            GVP(edge_in_dims, edge_hidden_dims, activations=(None, None))
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(hidden_dims, edge_hidden_dims, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = hidden_dims
        self.W_out = nn.Sequential(
            LayerNorm(hidden_dims),
            GVP(hidden_dims, (ns, 0))
        )

    def forward(self, h_V, edge_index, h_E):

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out
    
class GVP(nn.Module):

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):

        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s
    
class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPConv(MessagePassing):

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

class GVPConvLayer(nn.Module):

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):

        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP,
                                 activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

class ResidueGlobalPooling(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(ResidueGlobalPooling, self).__init__()
        
        self.si, _ = in_dims  
        self.so, _ = out_dims 
         
        self.ws = nn.Linear(self.si, self.so)    
        self.w = nn.Linear(self.so, 1, bias=False)  
        self.scalar_act = nn.SiLU()

    def forward(self, x, batch):
        s, _ = x 
        s_out = self.ws(s)
        s_out = self.scalar_act(s_out)         
        
        alpha = self.w(s_out).squeeze(-1)
        from torch_scatter import scatter_softmax
        alpha = scatter_softmax(alpha, batch, dim=0)                
        global_feature = scatter_add(alpha.unsqueeze(-1) * s_out, batch, dim=0)
        
        return global_feature

class GVPClassifier(nn.Module):
    def __init__(self, in_dims, hidden_dims, edge_in_dims, edge_hidden_dims,
                 num_layers=3, drop_rate=0.1, vector_gate=False, 
                 pool_out_dim=128, output_dim=1):
        super(GVPClassifier, self).__init__()
        
        self.gvp_encoder = GVP_embedding(
            in_dims=in_dims,
            hidden_dims=hidden_dims,
            edge_in_dims=edge_in_dims, 
            edge_hidden_dims=edge_hidden_dims,
            num_layers=num_layers,
            drop_rate=drop_rate,
            vector_gate=vector_gate
        )
        
        self.global_pooling = ResidueGlobalPooling(
            in_dims=(hidden_dims[0], 0),
            out_dims=(pool_out_dim, 0)
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_out_dim, pool_out_dim // 2),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(pool_out_dim // 2, output_dim)
        )
    
    def forward(self, data):
        if isinstance(data, Batch):  
            node_s_37dim = data.node_s[:, :37]
            h_V = (node_s_37dim, data.node_v)
            edge_index = data.edge_index
            h_E = (data.edge_s, data.edge_v)
            batch = data.batch
        else:  
            node_s_37dim = data['node_s'][:, :37]
            h_V = (node_s_37dim, data['node_v'])
            edge_index = data['edge_index']
            h_E = (data['edge_s'], data['edge_v'])
            batch = data['batch']
        
        node_embeddings = self.gvp_encoder(h_V, edge_index, h_E)
        
        if isinstance(node_embeddings, tuple):
            graph_embedding = self.global_pooling(node_embeddings, batch)
        else:
            graph_embedding = self.global_pooling((node_embeddings, None), batch)
        
        logits = self.output_mlp(graph_embedding)
        
        return logits if logits.dim() > 0 else logits.unsqueeze(0)


class ProteinDataset(Dataset):
    def __init__(self, data_dir, shuffle=True, seed=42):
        self.data_dir = data_dir
        file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        if shuffle:
            random.seed(seed)
            random.shuffle(file_list)
        
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        try:
            data = torch.load(file_path)
            node_s_sliced = data['node_s'][:, :37] 
            return Data(
                node_s=node_s_sliced.cpu().to(torch.float32).clone(),
                node_v=data['node_v'].cpu().to(torch.float32),
                edge_s=data['edge_s'].cpu().to(torch.float32),
                edge_v=data['edge_v'].cpu().to(torch.float32),
                edge_index=data['edge_index'].cpu(),
                y=data['y'].cpu().to(torch.float32).clone(),
                num_nodes=data['node_s'].shape[0]  
            )
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
