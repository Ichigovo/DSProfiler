import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
from collections import defaultdict
from Bio.PDB import MMCIFParser
from tqdm import tqdm
import os
import re
from torch_cluster import knn_graph 
import argparse

STANDARD_AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                        'THR', 'TRP', 'TYR', 'VAL']
MODIFIED_AMINO_ACIDS = ['MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP', 
                        'MLY', 'PCA', 'LLP']
ALL_AMINO_ACIDS = STANDARD_AMINO_ACIDS + MODIFIED_AMINO_ACIDS + ['OTHER']

def get_atom_features(atom, pdb_info):
    features = []
        
    if pdb_info:
        res_name = pdb_info.GetResidueName().strip()
        aa_onehot = [0] * len(ALL_AMINO_ACIDS)
        if res_name in ALL_AMINO_ACIDS:
            aa_idx = ALL_AMINO_ACIDS.index(res_name)
        else:
            aa_idx = ALL_AMINO_ACIDS.index('OTHER')
        aa_onehot[aa_idx] = 1
    else:
        aa_onehot = [0] * len(ALL_AMINO_ACIDS)
        aa_onehot[ALL_AMINO_ACIDS.index('OTHER')] = 1
    
    features.extend(aa_onehot)

    atom_type_mapping = {
        'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 
        'P': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'B': 9, 'SE': 10, 'Other': 11
    }
    element = atom.GetSymbol()
    atom_type = atom_type_mapping.get(element, 11)
    features.append(atom_type)
    
    if pdb_info:
        atom_name = pdb_info.GetName().strip()
        backbone_atoms = ['N', 'CA', 'C', 'O']
        is_backbone = 1 if atom_name in backbone_atoms else 0
    else:
        is_backbone = 0
    features.append(is_backbone)

    degree = atom.GetDegree()
    degree = min(degree, 5)
    features.append(degree)
    
    hybridization_mapping = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    hybridization = hybridization_mapping.get(atom.GetHybridization(), 5)
    features.append(hybridization)

    is_aromatic = 1 if atom.GetIsAromatic() else 0
    features.append(is_aromatic)
    
    return features

def get_bond_type(mol, atom1_idx, atom2_idx):
    bond = mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
    if bond is not None:
        bond_type_mapping = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }
        return bond_type_mapping.get(bond.GetBondType(), 4)
    else:
        return 4  

def are_same_residue(mol, atom1_idx, atom2_idx):
    atom1 = mol.GetAtomWithIdx(atom1_idx)
    atom2 = mol.GetAtomWithIdx(atom2_idx)
    
    pdb_info1 = atom1.GetPDBResidueInfo()
    pdb_info2 = atom2.GetPDBResidueInfo()
    
    if pdb_info1 and pdb_info2:
        return (pdb_info1.GetResidueNumber() == pdb_info2.GetResidueNumber() and 
                pdb_info1.GetChainId() == pdb_info2.GetChainId())
    return False

def should_include_residue(residue):
    resname = residue.resname.strip()
    if residue.id[0] != ' ' and resname not in MODIFIED_AMINO_ACIDS:
        return False

    c_alpha, n, c = None, None, None
    for atom in residue:
        if atom.name == 'CA':
            c_alpha = list(atom.get_vector())
        if atom.name == 'N':
            n = list(atom.get_vector())
        if atom.name == 'C':
            c = list(atom.get_vector())
    if not (c_alpha and n and c):
        return False
    return True

def parse_pocket_cif(pocket_cif_file, name):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("pocket", pocket_cif_file)[0]
    pocket_atoms = []

    for chain in structure:
        for residue in chain:
            if should_include_residue(residue): 
                chain_id = chain.id
                res_num = residue.get_id()[1]
                ins_code = residue.get_id()[2].strip()

                for atom in residue:
                    atom_name = atom.get_name().strip()
                    if ins_code:
                        atom_key = f"{chain_id}_{res_num}_{ins_code}_{atom_name}"
                    else:
                        atom_key = f"{chain_id}_{res_num}_{atom_name}"
                    
                    pocket_atoms.append({
                        'atom_key': atom_key,
                        'atom_name': atom_name
                    })

    return pocket_atoms

def extract_pocket_features_from_full_protein(full_protein_pkl_file, pocket_cif_file, k=20):
    with open(full_protein_pkl_file, 'rb') as f:
        full_protein_data = pickle.load(f)
    
    full_protein_mol = full_protein_data['mol']
    pybel_features = full_protein_data['pybel_features']
    atom_mapping = full_protein_data['atom_mapping']
    
    pocket_name = os.path.splitext(os.path.basename(pocket_cif_file))[0]
    pocket_atoms = parse_pocket_cif(pocket_cif_file, pocket_name)

    pocket_atom_indices = []
    for atom_info in pocket_atoms:
        atom_key = atom_info['atom_key']
        if atom_key in atom_mapping:
            atom_idx = atom_mapping[atom_key]
            atom = full_protein_mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() > 1:
                pocket_atom_indices.append(atom_idx)

    if len(pocket_atom_indices) == 0:
        print(f"Error: No matching pocket atoms found for {pocket_name}")
        return None

    node_features = []
    coords = []
    atom_global_to_local = {}  
    
    conformer = full_protein_mol.GetConformer()
    for local_idx, global_idx in enumerate(pocket_atom_indices):
        atom = full_protein_mol.GetAtomWithIdx(global_idx)
        pdb_info = atom.GetPDBResidueInfo()
        
        features = get_atom_features(atom, pdb_info)
        if pybel_features is not None and global_idx < pybel_features.shape[0]:
            pybel_feat = pybel_features[global_idx].tolist()
            combined_features = features + pybel_feat
        else:
            print("Could not find corresponding Pybel features")
            combined_features = features + [0.0, 0.0, 0.0]  

        node_features.append(combined_features)
        
        pos = conformer.GetAtomPosition(global_idx)
        coords.append([pos.x, pos.y, pos.z])
        
        atom_global_to_local[global_idx] = local_idx
    
    node_features = torch.tensor(node_features, dtype=torch.float)
    coords = torch.tensor(coords, dtype=torch.float)

    edge_index = knn_graph(coords, k=k)
    
    edge_features = []
    num_edges = edge_index.size(1)

    for i in range(num_edges):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        
        global_src = pocket_atom_indices[src_idx]
        global_dst = pocket_atom_indices[dst_idx]
        
        bond_type = get_bond_type(full_protein_mol, global_src, global_dst)
        
        coord_src = coords[src_idx]
        coord_dst = coords[dst_idx]
        spatial_distance = torch.norm(coord_src - coord_dst).item()
        
        same_residue = 1 if are_same_residue(full_protein_mol, global_src, global_dst) else 0
        
        edge_features.append([same_residue, bond_type, spatial_distance])
    
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    return {
        'node_s': node_features,
        'coords': coords,
        'edge_index': edge_index,
        'edge_s': edge_features,
        'pocket_id': pocket_name,
        'y': torch.tensor([1], dtype=torch.long)  
    }

def process_single_pocket_atom(pdb_id, name, base_working_dir, k=20):
    try:
        pdb_id = str(pdb_id).strip()
        name = str(name).strip()

        protein_pkl_path = os.path.join(base_working_dir, pdb_id, f"{pdb_id}_a.pkl")
        
        pocket_name = f"{name}"
        pocket_cif_path = os.path.join(base_working_dir, pdb_id, pocket_name, f"{pocket_name}.cif")
        output_dir = os.path.join(base_working_dir, pdb_id, pocket_name)
        
        if not os.path.exists(protein_pkl_path):
            print(f"Error: Protein PKL file not found at {protein_pkl_path}")
            return False
            
        if not os.path.exists(pocket_cif_path):
            print(f"Error: Pocket file not found at {pocket_cif_path}")
            return False
        
        features_dict = extract_pocket_features_from_full_protein(
            protein_pkl_path, 
            pocket_cif_path, 
            k=k
        )
        
        if features_dict is None:
            print(f"Error: Failed to extract features for {pocket_name}")
            return False
        
        output_path = os.path.join(output_dir, f"{pocket_name}_a.pt")
        torch.save(features_dict, output_path)
        
        return True
        
    except Exception as e:
        print(f"Error: Failed to process {name} : {e}")
        return False

def run_pipeline(input_csv, base_dir, k=20):
    df = pd.read_csv(input_csv, dtype={'pdb_id': str})

    required_columns = ['pdb_id', 'name']
    missing_columns = [col for col in required_columns if col not in df.columns]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row['pdb_id']
        name = row['name']
        
        success = process_single_pocket_atom(
            pdb_id=pdb_id,
            name=name,
            base_working_dir=base_dir,
            k=k
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline for protein atom feature.')
    parser.add_argument("--input_csv", type=str, default="example_test.csv", help="Input CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(), 
                        help="The directory containing PDB_ID subfolders")
    args = parser.parse_args()
    
    run_pipeline(input_csv=args.input_csv, base_dir=args.base_dir)