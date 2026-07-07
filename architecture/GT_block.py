import torch 
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean, scatter_add
from torch import nn
import os
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
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

def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)

class MultiHeadAttentionLayer(nn.Module):
    """Compute attention scores with a DGLGraph's node and edge (geometric) features."""
    def __init__(self, num_input_feats, num_output_feats,
                num_heads, using_bias=False, update_edge_feats=True,
                use_formula_attention=True, dropout_rate=0.1):  
        super(MultiHeadAttentionLayer, self).__init__()
        
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats
        self.use_formula_attention = use_formula_attention 
        self.dropout_rate = dropout_rate 
        
        # Define node features' query, key, and value tensors, and define edge features' projection tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        if self.use_formula_attention:
            self.edge_feats_mlp = nn.Sequential(
                nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(),
                nn.Linear(self.num_output_feats * self.num_heads, self.num_output_feats * self.num_heads, bias=using_bias)
            )
        else:
            self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)
            
            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)
            
            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            if self.use_formula_attention:
                for layer in self.edge_feats_mlp:
                    if hasattr(layer, 'weight'):
                        glorot_orthogonal(layer.weight, scale=scale)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            layer.bias.data.fill_(0)
            else:
                glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
                self.edge_feats_projection.bias.data.fill_(0)
            
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            if self.use_formula_attention:
                for layer in self.edge_feats_mlp:
                    if hasattr(layer, 'weight'):
                        glorot_orthogonal(layer.weight, scale=scale)
            else:
                glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
    
    def propagate_attention(self, edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection):
        row, col = edge_index
        e_out = None

        alpha = node_feats_k[row] * node_feats_q[col]
        alpha = (alpha / np.sqrt(self.num_output_feats)).clamp(-5.0,5.0)
        alpha = alpha * edge_feats_projection
        if self.update_edge_feats:
            e_out = alpha
		
        attention_scores = alpha.sum(-1, keepdim=True)
        attention_weights = torch_scatter.scatter_softmax(attention_scores, col, dim=0)

        wV = scatter_add(node_feats_v[row]*attention_weights, col, dim=0, dim_size=node_feats_q.size(0))
        z = scatter_add(attention_weights, col, dim=0, dim_size=node_feats_q.size(0))
        return wV, z, e_out
    
    def forward(self, x, edge_attr, edge_index):
        node_feats_q = self.Q(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_k = self.K(x).view(-1, self.num_heads, self.num_output_feats)
        node_feats_v = self.V(x).view(-1, self.num_heads, self.num_output_feats)
        if self.use_formula_attention:
            edge_feats_projection = self.edge_feats_mlp(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        else:
            edge_feats_projection = self.edge_feats_projection(edge_attr).view(-1, self.num_heads, self.num_output_feats)
        wV, z, e_out = self.propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection)
        
        h_out = wV / (z + torch.full_like(z, 1e-6))
        return h_out, e_out

class GraphTransformerModule(nn.Module):
    """A Graph Transformer module (equivalent to one layer of graph convolutions)."""
    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            use_formula_attention=True,
            use_gradient_checkpointing=True  
            ):
        super(GraphTransformerModule, self).__init__()
        
        # Record parameters given
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_formula_attention = use_formula_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
        
        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
        
        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,
            update_edge_feats=True,
            use_formula_attention=self.use_formula_attention
        )
        
        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        
        # MLP for node features 
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])
        
        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:  # Otherwise, default to using batch normalization
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)
        
        # MLP for edge features 
        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)
        
        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)
        
        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, edge_index, node_feats, edge_feats):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._run_gt_layer_impl, 
                edge_index, node_feats, edge_feats,
                use_reentrant=False
            )
        else:
            return self._run_gt_layer_impl(edge_index, node_feats, edge_feats)
    
    def _run_gt_layer_impl(self, edge_index, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats
        edge_feats_in1 = edge_feats
            
        # Apply first round of normalization 
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)
        
        # Get multi-head attention output
        node_attn_out, edge_attn_out = self.mha_module(node_feats, edge_feats, edge_index)
        
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)
        
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)
        
        # Make first residual connection 
        if self.residual:
            node_feats = node_feats_in1 + node_feats
            edge_feats = edge_feats_in1 + edge_feats
        
        node_feats_in2 = node_feats
        edge_feats_in2 = edge_feats
        
        # Apply second round of normalization 
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)
        
        # Apply MLPs for node and edge features 
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)
        
        # Make second residual connection 
        if self.residual:
            node_feats = node_feats_in2 + node_feats
            edge_feats = edge_feats_in2 + edge_feats
        
        return node_feats, edge_feats
    
    def forward(self, edge_index, node_feats, edge_feats):
        """Perform a forward pass of a Geometric Transformer to get intermediate node and edge representations."""
        node_feats, edge_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
        return node_feats, edge_feats

class FinalGraphTransformerModule(nn.Module):
    """A (final layer) Graph Transformer module that combines node and edge representations using self-attention."""	
    def __init__(self,
                num_hidden_channels,
                activ_fn=nn.SiLU(),
                residual=True,
                num_attention_heads=4,
                norm_to_apply='batch',
                dropout_rate=0.1,
                num_layers=4,
                use_formula_attention=True,
                use_gradient_checkpointing=True):  
        super(FinalGraphTransformerModule, self).__init__()
        
        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_formula_attention = use_formula_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()
        
        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)
        
        self.mha_module = MultiHeadAttentionLayer(
                    self.num_hidden_channels,
                    self.num_output_feats // self.num_attention_heads,
                    self.num_attention_heads,
                    self.num_hidden_channels != self.num_output_feats,
                    update_edge_feats=False,
                    use_formula_attention=self.use_formula_attention)  
        
        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        
        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
                    nn.Linear(self.num_output_feats, self.num_output_feats * 2, bias=False),
                    self.activ_fn,
                    dropout,
                    nn.Linear(self.num_output_feats * 2, self.num_output_feats, bias=False)
                    ])
        
        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        
        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, edge_index, node_feats, edge_feats):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._run_gt_layer_impl, 
                edge_index, node_feats, edge_feats,
                use_reentrant=False
            )
        else:
            return self._run_gt_layer_impl(edge_index, node_feats, edge_feats)
    
    def _run_gt_layer_impl(self, edge_index, node_feats, edge_feats):
        """Perform a forward pass of geometric attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats
        
        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)
        
        node_attn_out, _ = self.mha_module(node_feats, edge_feats, edge_index)
        node_feats = node_attn_out.view(-1, self.num_output_feats)        
        node_feats = F.dropout(node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)
        
        if self.residual:
            node_feats = node_feats_in1 + node_feats
        
        node_feats_in2 = node_feats
        
        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)
        
        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        
        if self.residual:
            node_feats = node_feats_in2 + node_feats
        
        return node_feats
    
    def forward(self, edge_index, node_feats, edge_feats):
        node_feats = self.run_gt_layer(edge_index, node_feats, edge_feats)
        return node_feats

class GraphTransformer(nn.Module):
    def __init__(
            self,
            in_channels, 
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            use_formula_attention=True, 
            use_gradient_checkpointing=True,
            **kwargs
            ):
        super(GraphTransformer, self).__init__()
        
        # Initialize model parameters
        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.use_formula_attention = use_formula_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.actual_edge_dim = edge_features
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(self.actual_edge_dim, num_hidden_channels)
        
        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
                                        num_hidden_channels=num_hidden_channels,
                                        activ_fn=activ_fn,
                                        residual=transformer_residual,
                                        num_attention_heads=num_attention_heads,
                                        norm_to_apply=norm_to_apply,
                                        dropout_rate=dropout_rate,
                                        num_layers=num_layers,
                                        use_formula_attention=use_formula_attention,
                                        use_gradient_checkpointing=use_gradient_checkpointing) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([
                            FinalGraphTransformerModule(
                                        num_hidden_channels=num_hidden_channels,
                                        activ_fn=activ_fn,
                                        residual=transformer_residual,
                                        num_attention_heads=num_attention_heads,
                                        norm_to_apply=norm_to_apply,
                                        dropout_rate=dropout_rate,
                                        num_layers=num_layers,
                                        use_formula_attention=use_formula_attention,
                                        use_gradient_checkpointing=use_gradient_checkpointing)])
        self.gt_block = nn.ModuleList(gt_block_modules)
    
    def forward(self, node_s, edge_s, edge_index): 
        node_feats = self.node_encoder(node_s)
        edge_feats = self.edge_encoder(edge_s)
            
        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(edge_index, node_feats, edge_feats)
        
        node_feats = self.gt_block[-1](edge_index, node_feats, edge_feats)
        return node_feats
    
class AtomGlobalPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AtomGlobalPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.w_A = nn.Linear(hidden_dim, 1, bias=False) 

    def forward(self, h_i, batch):
        alpha_i = self.w_A(h_i).squeeze(-1)  
        from torch_scatter import scatter_softmax
        alpha_i = scatter_softmax(alpha_i, batch, dim=0)
        global_feature = scatter_add(alpha_i.unsqueeze(-1) * h_i, batch, dim=0)
        
        return global_feature

class GTClassifier(nn.Module):
    def __init__(self, 
                 in_channels, 
                 edge_features=10,
                 num_hidden_channels=128,
                 num_attention_heads=4,
                 num_layers=4,
                 use_formula_attention=True,
                 dropout_rate=0.1,
                 use_gradient_checkpointing=True):
        super(GTClassifier, self).__init__()

        self.gt = GraphTransformer(
            in_channels=in_channels,
            edge_features=edge_features,
            num_hidden_channels=num_hidden_channels,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            use_formula_attention=use_formula_attention,
            dropout_rate=dropout_rate,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # 全局池化
        self.global_pool = AtomGlobalPooling(num_hidden_channels)

        self.output_mlp = nn.Sequential(
            nn.Linear(num_hidden_channels, num_hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden_channels // 2, 1)
        )
       
    def forward(self, node_s, edge_s, edge_index, batch):
        node_features = self.gt(node_s, edge_s, edge_index)
        
        global_features = self.global_pool(node_features, batch)
        
        logits = self.output_mlp(global_features)
        
        return logits

def load_center_dict(csv_path):
    df = pd.read_csv(csv_path)
    center_dict = {}
    for _, row in df.iterrows():
        c_str = row['name']
        coords = row['center'].strip('()').split(',')
        center_dict[c_str] = np.array([float(x.strip()) for x in coords])
    return center_dict

class ProteinDataset(Dataset):
    def __init__(self, data_dir, center_dict, cutoff=9.0, shuffle=True, seed=42):
        self.data_dir = data_dir
        self.center_dict = center_dict
        self.cutoff = cutoff
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.file_list = [f for f in all_files if f.replace('.pt', '') in center_dict]

        if shuffle:
            random.seed(seed)
            random.shuffle(self.file_list)
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        key = file_name.replace('.pt', '')
        file_path = os.path.join(self.data_dir, file_name)
        try:
            data_dict = torch.load(file_path)
            center = self.center_dict[key]

            coords = data_dict['coords'].cpu()
            dists = torch.norm(coords - center, dim=1)
            subset_indices = torch.where(dists <= self.cutoff)[0]
            
            if len(subset_indices) == 0: 
                subset_indices = torch.tensor([torch.argmin(dists)])

            edge_index, edge_attr = subgraph(
                subset_indices, 
                data_dict['edge_index'].cpu(), 
                edge_attr=data_dict['edge_s'].cpu(), 
                relabel_nodes=True,
                num_nodes=coords.size(0)
            )
            
            return Data(
                node_s=data_dict['node_s'][subset_indices].float(),
                edge_s=edge_attr.float(),
                edge_index=edge_index,
                y=data_dict['y'].float(),
                num_nodes=len(subset_indices)
            )
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
