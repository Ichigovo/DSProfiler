import os
import re
import warnings
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from Bio.PDB import MMCIFParser
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

def parse_center(center_str):
    if pd.isna(center_str):
        return None
    nums = re.findall(r'-?\d+\.\d+|-?\d+', str(center_str))
    if len(nums) >= 3:
        return np.array([float(x) for x in nums[:3]])
    return None

def get_chains_within_radius(center, atom_coords, atom_chains, radius=6.0):
    if center is None or len(atom_coords) == 0:
        return set()
    distances = cdist([center], atom_coords)[0]
    idx = np.where(distances <= radius)[0]
    return set(atom_chains[idx])

def process_structure(pdb_id, cif_dir):
    cif_path = os.path.join(cif_dir, f"{pdb_id}.cif")
    if not os.path.exists(cif_path):
        return None, None
    try:
        parser = MMCIFParser()
        model = parser.get_structure(pdb_id, cif_path)[0]
        atom_coords, atom_chains = [], []
        
        for chain in model:
            cid = chain.get_id()
            for residue in chain:
                if residue.id[0] == ' ': 
                    for atom in residue:
                        atom_coords.append(atom.get_coord())
                        atom_chains.append(cid)
                        
        if not atom_coords:
            return None, None
            
        return np.array(atom_coords), np.array(atom_chains)
    except Exception as e:
        print(f"Error parsing {pdb_id}: {e}")
        return None, None

def main():
    results_csv = "./example/rank/example_results.csv"
    reference_csv = "./evaluation/reference.csv"
    structure_dir = "./example/rank/structure"
    filtered_csv = "./evaluation/example_results_filter.csv"

    df_res = pd.read_csv(results_csv)
    df_ref = pd.read_csv(reference_csv)

    df_ref['center_ref'] = df_ref['center'].apply(parse_center)
    df_ref = df_ref.dropna(subset=['center_ref'])
    ref_dict = df_ref.groupby('pdb_id')['center_ref'].apply(list).to_dict()

    df_res['center_pred'] = df_res['center'].apply(parse_center)
    df_res = df_res.dropna(subset=['center_pred'])

    filtered_rows = []
    for pdb_id, group in df_res.groupby('pdb_id'):
        if pdb_id not in ref_dict:
            continue 
            
        ref_centers = ref_dict[pdb_id]
        atom_coords, atom_chains = process_structure(pdb_id, structure_dir)

        if atom_coords is None:
            for _, row in group.iterrows():
                dccs = [np.linalg.norm(row['center_pred'] - c_ref) for c_ref in ref_centers]
                row_dict = row.to_dict()
                row_dict['DCC'] = min(dccs)
                filtered_rows.append(row_dict)
            continue

        global_T = set()
        for c_ref in ref_centers:
            global_T.update(get_chains_within_radius(c_ref, atom_coords, atom_chains, radius=6.0))

        for _, row in group.iterrows():
            c_pred = row['center_pred']
            P = get_chains_within_radius(c_pred, atom_coords, atom_chains, radius=6.0)
            
            is_touching = len(P.intersection(global_T)) > 0
            if len(P) > 0 and not is_touching:
                continue

            dccs = [np.linalg.norm(c_pred - c_ref) for c_ref in ref_centers]
            
            row_dict = row.to_dict()
            row_dict['DCC'] = min(dccs)
            filtered_rows.append(row_dict)

    if not filtered_rows:
        print("No data left after filtering!")
        return

    filtered_df = pd.DataFrame(filtered_rows)

    filtered_df = filtered_df.sort_values(['pdb_id', 'DSProfiler_score'], ascending=[True, False])
    filtered_df['DSProfiler_rank'] = filtered_df.groupby('pdb_id').cumcount() + 1

    out_df = filtered_df[['pdb_id', 'DSProfiler_rank', 'DSProfiler_score', 'center', 'DCC']]
    out_df.to_csv(filtered_csv, index=False)
    print(f"Filtered results saved to {filtered_csv}")

    total_n = df_ref['pdb_id'].nunique() 
    df_tp = out_df[out_df['DCC'] <= 4.0].copy() 
    first_tp = df_tp.groupby('pdb_id')['DSProfiler_rank'].min()

    top1 = (first_tp <= 1).sum() / total_n
    top3 = (first_tp <= 3).sum() / total_n
    top5 = (first_tp <= 5).sum() / total_n
    top10 = (first_tp <= 10).sum() / total_n
    mrr = (1 / first_tp).sum() / total_n

    print("-" * 30)
    print(f"Total valid PDBs evaluated: {total_n}")
    print(f"Top-1  Accuracy: {top1:.4f}")
    print(f"Top-3  Accuracy: {top3:.4f}")
    print(f"Top-5  Accuracy: {top5:.4f}")
    print(f"Top-10 Accuracy: {top10:.4f}")
    print(f"MRR:             {mrr:.4f}")

if __name__ == "__main__":
    main()