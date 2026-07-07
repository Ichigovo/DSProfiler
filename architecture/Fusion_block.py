import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
import os
import json
import matplotlib.pyplot as plt
import random
import numpy as np
from torch import optim
import itertools
import sys
import pandas as pd
from collections import defaultdict
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean, scatter_add

from GT_block import GTClassifier
from GVP_block import GVPClassifier

class DualScaleFusionModel(nn.Module):
    def __init__(self, gt_model, gvp_model, fusion_dim=64, dropout_rate=0.1):
        super(DualScaleFusionModel, self).__init__()
        
        self.gt_backbone = gt_model.gt
        self.gt_pool = gt_model.global_pool
        
        self.gvp_backbone = gvp_model.gvp_encoder
        self.gvp_pool = gvp_model.global_pooling

        self.conditional_gate = nn.Sequential(
            nn.Linear(256, 64),  
            nn.SiLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(64, 2),  
            nn.Softmax(dim=1)  
        )
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128, fusion_dim), 
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.SiLU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, 1)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.conditional_gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        for m in self.fusion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, gt_data, gvp_data):
        with torch.no_grad():
            gt_node_features = self.gt_backbone(
                gt_data.node_s, gt_data.edge_s, gt_data.edge_index
            )
            f_A = self.gt_pool(gt_node_features, gt_data.batch)
        
            node_s_37dim = gvp_data.node_s[:, :37]
            h_V = (node_s_37dim, gvp_data.node_v)
            edge_index = gvp_data.edge_index
            h_E = (gvp_data.edge_s, gvp_data.edge_v)
        
            node_embeddings = self.gvp_backbone(h_V, edge_index, h_E)

            if not isinstance(node_embeddings, tuple):
                node_embeddings = (node_embeddings, None)
            f_R = self.gvp_pool(node_embeddings, gvp_data.batch)
        
        context_features = torch.cat([f_A, f_R], dim=1)
        gates = self.conditional_gate(context_features)  
        gate_A = gates[:, 0].unsqueeze(1)  
        gate_R = gates[:, 1].unsqueeze(1) 
        
        f_A_gated = gate_A * f_A
        f_R_gated = gate_R * f_R
        
        f_fused = f_A_gated + f_R_gated
        
        logits = self.fusion_mlp(f_fused)
        
        return logits
    