import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from scipy.spatial import distance_matrix
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import os
import warnings
import pandas as pd
import tqdm
from Bio.PDB import MMCIFParser, PDBIO, Select, is_aa
import tempfile
import re
import argparse

STANDARD_AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                       'THR', 'TRP', 'TYR', 'VAL']
MODIFIED_AMINO_ACIDS = ['MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP', 
                       'MLY', 'PCA', 'LLP']
ALL_AMINO_ACIDS = STANDARD_AMINO_ACIDS + MODIFIED_AMINO_ACIDS + ['OTHER']
AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(ALL_AMINO_ACIDS)}
NUM_AMINO_ACID_TYPES = len(ALL_AMINO_ACIDS)

KYTE_DOOLITTLE_SCALE = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}
for mod_aa in MODIFIED_AMINO_ACIDS:
    KYTE_DOOLITTLE_SCALE[mod_aa] = 0.0

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_residue_name(res):
    resname = res.resname.strip()
    return resname if resname in ALL_AMINO_ACIDS else 'OTHER'

def get_kyte_doolittle_value(resname):
    return KYTE_DOOLITTLE_SCALE.get(resname, 0.0)

def get_dihedral_angles(res):
    angles = [0.0, 0.0, 0.0, 0.0]
    angle_selectors = [res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]
    for i, selector in enumerate(angle_selectors):
        try:
            selection = selector()
            if selection is not None:
                angles[i] = selection.dihedral.value()
        except:
            continue
    return angles

def get_ca_position(res):
    try:
        return res.atoms.select_atoms("name CA").positions[0]
    except:
        return res.atoms.positions.mean(axis=0)
    
    
def check_covalent_connection(res_list, i, j):
    return 1 if abs(i - j) == 1 and res_list[i].segid == res_list[j].segid else 0

def get_orientation_vectors(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def get_sidechain_vectors(n_pos, ca_pos, c_pos):
    c_vec = _normalize(c_pos - ca_pos)
    n_vec = _normalize(n_pos - ca_pos)
    bisector = _normalize(c_vec + n_vec)
    perp = _normalize(torch.cross(c_vec, n_vec))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec

class ProteinSelect(Select):
    def accept_residue(self, residue):
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

def filter_pocket_structure(cif_file):
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_pdb:
        temp_pdb_path = temp_pdb.name
    
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)

        chain_mapping = {}
        available_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

        existing_single_char_chains = set()
        for model in structure:
            for chain in model:
                if len(chain.id) == 1 and chain.id in available_chars:
                    existing_single_char_chains.add(chain.id)
        
        for model in structure:
            for chain in model:
                orig_chain = chain.id
                
                if len(orig_chain) == 1 and orig_chain in available_chars:
                    chain_mapping[orig_chain] = orig_chain
                else:
                    for char in available_chars:
                        if char not in existing_single_char_chains:
                            chain_mapping[orig_chain] = char
                            existing_single_char_chains.add(char)  # 标记为已使用
                            break
                    else:
                        chain_mapping[orig_chain] = orig_chain[0] if orig_chain else 'A'
        
        for model in structure:
            for chain in model:
                if chain.id in chain_mapping:
                    chain.id = chain_mapping[chain.id]
        
        io = PDBIO()
        io.set_structure(structure)

        selector = ProteinSelect()
        io.save(temp_pdb_path, selector)

    except Exception as e:
        print(f"Error: Failed to process {cif_file} : {e}")
        return False
    
    universe = mda.Universe(temp_pdb_path)
    os.remove(temp_pdb_path)
    
    return universe

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

def extract_pocket_sasa(pdb_id, pocket_file, full_protein_file, sasa_dict):
    try:
        if not os.path.exists(pocket_file):
            print(f"Error: pocket file {pocket_file} not found.")
            return None
            
        if not os.path.exists(full_protein_file):
            print(f"Error: full protein file {full_protein_file} not found.")
            return None
        
        mmcif_parser = MMCIFParser(QUIET=True)
        pocket_structure = mmcif_parser.get_structure("pocket", pocket_file)[0]  
        full_structure = mmcif_parser.get_structure(pdb_id, full_protein_file)[0]

        pocket_sasa_values = []

        for chain_id, chain in pocket_structure.child_dict.items():
            if chain_id not in full_structure.child_dict:
                print(f"Warning: Chain {chain_id} not found in the full protein, skipping.")
                continue

            sasa_chain_key = f"{pdb_id.lower()}_{chain_id}"
            if sasa_chain_key not in sasa_dict:
                print(f"Warning: Chain {sasa_chain_key} not found in the SASA dictionary.")
                continue

            chain_sasa_tensor = sasa_dict[sasa_chain_key]
        
            full_chain = full_structure[chain_id]
            full_chain_residue_map = {}
            valid_residues = []
            
            for residue in full_chain.get_residues():
                if should_include_residue(residue):
                    valid_residues.append(residue)
                else:
                    continue
            
            if len(valid_residues) != len(chain_sasa_tensor):
                print(f"{sasa_chain_key}: Embedding length must match the number of residues!")
                print(f"Number of residues: {len(valid_residues)}, SASA tensor length: {len(chain_sasa_tensor)}")
                continue

            for res_idx, residue in enumerate(valid_residues):
                if residue.resname != 'HOH':
                    full_chain_residue_map[residue.id] = (res_idx, residue.resname)

            pocket_indices = []
            for residue in chain.get_residues():                       
                if not should_include_residue(residue):
                    continue
          
                if residue.id in full_chain_residue_map:
                    full_res_idx, full_res_name = full_chain_residue_map[residue.id]
                    if residue.resname == full_res_name:
                        pocket_indices.append(full_res_idx)
                    else:
                        print(f"Residue name mismatch: {residue.resname} in pocket, {full_res_name} in full protein.")
                else:
                    print(f"Residue {residue.id} not found in the full protein.")

            if pocket_indices:
                chain_pocket_sasa = chain_sasa_tensor[pocket_indices]
                pocket_sasa_values.append(chain_pocket_sasa)
    
        if pocket_sasa_values:
            final_pocket_sasa = torch.cat(pocket_sasa_values, dim=0)
            return final_pocket_sasa.unsqueeze(1)  
        else:
            return None
        
    except Exception as e:
        print(f"Error extracting pocket SASA for {pdb_id}: {e}")
        return None

def generate_protein_graph(mda_universe, sasa_embedding=None, key=None):
    with torch.no_grad():
        pure_res_lis, node_scalar_features, X_ca, X_n, X_c = [], [], [], [], []
        
        for res in mda_universe.residues:
            try:
                res_name = res.resname.strip()
                res_name = res_name if res_name in ALL_AMINO_ACIDS else 'OTHER'

                ca = res.atoms.select_atoms("name CA")
                c = res.atoms.select_atoms("name C")
                n = res.atoms.select_atoms("name N")
                                
                aa_idx = AMINO_ACID_TO_IDX[res_name]
                aa_onehot = F.one_hot(torch.tensor(aa_idx), num_classes=NUM_AMINO_ACID_TYPES).float()
                dihedral = torch.tensor(get_dihedral_angles(res), dtype=torch.float32)
                hydrophobicity = torch.tensor([get_kyte_doolittle_value(res_name)], dtype=torch.float32)
                
                node_feat = torch.cat([aa_onehot, dihedral, hydrophobicity])
                node_scalar_features.append(node_feat)
                pure_res_lis.append(res)
                X_ca.append(ca.positions[0]) 
                X_n.append(n.positions[0])
                X_c.append(c.positions[0])
                
            except Exception as e:
                continue
        
        if len(pure_res_lis) == 0:
            raise ValueError("No valid residues found in the pocket")
        
        node_scalar_features = torch.stack(node_scalar_features)

        X_ca = torch.from_numpy(np.asarray(X_ca))
        X_n = torch.from_numpy(np.asarray(X_n))
        X_c = torch.from_numpy(np.asarray(X_c))
        
        if sasa_embedding is not None:
            if node_scalar_features.size(0) == sasa_embedding.size(0):
                node_scalar_features = torch.cat([node_scalar_features, sasa_embedding], dim=-1)
            else:
                print(f"Warning:{key} Number of nodes do not match!")

        edge_index = torch_cluster.knn_graph(X_ca, k=30)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        ca_distances = torch.norm(E_vectors, dim=1, keepdim=True)
        
        covalent_connections = torch.tensor([
            check_covalent_connection(pure_res_lis, i, j) 
            for i, j in edge_index.T.numpy()
        ]).unsqueeze(1)
        
        rbf_features = _rbf(E_vectors.norm(dim=-1), D_count=16, device='cpu')
        
        edge_scalar_features = torch.cat([
            covalent_connections,
            ca_distances, 
            rbf_features
        ], dim=-1)
        
        orientation_vectors = get_orientation_vectors(X_ca)
        sidechain_vectors = get_sidechain_vectors(X_n, X_ca, X_c)
        node_vector_features = torch.cat([orientation_vectors, sidechain_vectors.unsqueeze(-2)], dim=-2)
        edge_vector_features = _normalize(E_vectors).unsqueeze(-2)
        
        node_scalar_features, node_vector_features, edge_scalar_features, edge_vector_features = map(
            torch.nan_to_num, (node_scalar_features, node_vector_features, edge_scalar_features, edge_vector_features)
        )
        
        return {
            'node_s': node_scalar_features,
            'node_v': node_vector_features,
            'edge_index': edge_index,
            'edge_s': edge_scalar_features,
            'edge_v': edge_vector_features,
            'ca_coords': X_ca,
            'y': torch.tensor([1], dtype=torch.long)
        }
    
def process_single_pocket(pdb_id, name, base_working_dir):

    pdb_id = str(pdb_id).strip()
    name = str(name).strip()
        
    protein_cif_path = os.path.join(base_working_dir, pdb_id, f"{pdb_id}.cif")
    pocket_name = name
    pocket_cif_path = os.path.join(base_working_dir, pdb_id, pocket_name, f"{pocket_name}.cif")
    sasa_result_path = os.path.join(base_working_dir, pdb_id, f"{pdb_id}_r.pt")
    sasa_dict = torch.load(sasa_result_path)
    output_dir = os.path.join(base_working_dir, pdb_id, pocket_name)
    
    sasa_embedding = extract_pocket_sasa(pdb_id, pocket_cif_path, protein_cif_path, sasa_dict)
    if sasa_embedding is None:
        print(f"Warning: Failed to extract SASA features for {pocket_name}.")
        return False
    
    filtered_universe = filter_pocket_structure(pocket_cif_path)
            
    graph_data = generate_protein_graph(
        filtered_universe,
        sasa_embedding=sasa_embedding,
        key=pocket_name
        )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{pocket_name}_r.pt")
    torch.save(graph_data, output_path)
        
    return True

def run_pipeline(input_csv, base_dir):

    df = pd.read_csv(input_csv, dtype={'pdb_id': str})

    required_columns = ['pdb_id', 'name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: CSV file is missing required columns: {missing_columns}")

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        pdb_id = row['pdb_id']
        name = row['name']
        try:
            success = process_single_pocket(
                pdb_id=pdb_id,
                name=name,
                base_working_dir=base_dir
            )
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline for protein residue feature.')
    parser.add_argument("--input_csv", type=str, default="example_test.csv", help="Input CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(), 
                        help="The directory containing PDB_ID subfolders")
    args = parser.parse_args()
    
    run_pipeline(input_csv=args.input_csv, base_dir=args.base_dir)