import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from tqdm import tqdm 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader  
from torch_geometric.data import Data, Batch 
from torch_geometric.utils import subgraph
from Fusion_block import DualScaleFusionModel
from GT_block import GTClassifier
from GVP_block import GVPClassifier

class PocketDataset(Dataset):
    def __init__(self, csv_path, base_dir, cutoff=9.0):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.cutoff = cutoff

    def _parse_center(self, center_str):
        coords = center_str.strip('()').split(',')
        return np.array([float(x.strip()) for x in coords])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = str(row['pdb_id'])
        name = row['name']
        label = row['label']

        center = torch.from_numpy(self._parse_center(row['center'])).float()
        pocket_dir = os.path.join(self.base_dir, pdb_id, name)
        atm_path = os.path.join(pocket_dir, f"{name}_a.pt")
        res_path = os.path.join(pocket_dir, f"{name}_r.pt")
        
        if not os.path.exists(atm_path) or not os.path.exists(res_path):
            return None, name, label
            
        atm_d = torch.load(atm_path, map_location='cpu')
        res_d = torch.load(res_path, map_location='cpu')
        
        coords = atm_d['coords']
        dists = torch.norm(coords - center, dim=1)
        subset_indices = torch.where(dists <= self.cutoff)[0]
        if len(subset_indices) == 0:
            subset_indices = torch.tensor([torch.argmin(dists)])

        edge_index_atom, edge_attr_atom = subgraph(
            subset_indices, 
            atm_d['edge_index'], 
            edge_attr=atm_d['edge_s'], 
            relabel_nodes=True,
            num_nodes=coords.size(0)
        )

        gt_g = Data(
            node_s=atm_d['node_s'][subset_indices].float(),
            edge_s=edge_attr_atom.float(),
            edge_index=edge_index_atom
        )

        node_s_sliced = res_d['node_s'][:, :37]
        gvp_g = Data(node_s=node_s_sliced.float(), node_v=res_d['node_v'].float(), 
                     edge_s=res_d['edge_s'].float(), edge_v=res_d['edge_v'].float(), edge_index=res_d['edge_index'])
        
        return (gt_g, gvp_g), name, label

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: return None
    
    graphs, name, label = zip(*batch)
    gt_list, gvp_list = zip(*graphs)
    
    return Batch.from_data_list(gt_list), Batch.from_data_list(gvp_list), name, label

def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_obj = GTClassifier(in_channels=39, edge_features=3, num_hidden_channels=128, 
                                num_attention_heads=4, num_layers=3, use_formula_attention=False, 
                                use_gradient_checkpointing=True, dropout_rate=0.2)
    
    gvp_obj = GVPClassifier(in_dims=(37, 3), hidden_dims=(128, 32), edge_in_dims=(18, 1), 
                                 edge_hidden_dims=(64, 8), num_layers=3, drop_rate=0.2, 
                                 vector_gate=False, pool_out_dim=128)
    
    model = DualScaleFusionModel(gt_obj, gvp_obj)

    ckpt = torch.load(args.model_weight_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    dataset = PocketDataset(args.input_csv, args.base_dir, cutoff=9.0)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    results = []

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            
            gt_batch, gvp_batch, name, label = batch
            gt_batch = gt_batch.to(device)
            gvp_batch = gvp_batch.to(device)
            
            logits = model(gt_batch, gvp_batch)
            logits = logits.view(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            
            for name, label, score in zip(name, label, scores):
                results.append({
                    'name': name,
                    'label': label,
                    'score': score
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument("--input_csv", type=str, default="example_test.csv",
                        help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, default="example_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(),
                        help="The directory containing PDB_ID subfolders")
    parser.add_argument("--model_weight_path", type=str, required=True,
                        help="Path to the best model weights")
    
    args = parser.parse_args()
    run_pipeline(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    