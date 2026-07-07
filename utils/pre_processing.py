import os
import pandas as pd
import shutil
import argparse
import numpy as np
from Bio.PDB import MMCIFParser, MMCIFIO, Select

class WholeResidueSelect(Select):
    def __init__(self, valid_residue_ids):
        self.valid_residue_ids = valid_residue_ids

    def accept_residue(self, residue):
        res_id = (residue.get_parent().id, residue.id)
        return 1 if res_id in self.valid_residue_ids else 0

def parse_center(center_str):
    try:
        coords = center_str.replace('(', '').replace(')', '').split(',')
        return np.array([float(c) for c in coords])
    except Exception as e:
        print(f"Error parsing coordinates {center_str}: {e}")
        return None

def get_pocket_residue_ids(structure, center, radius=12.0):
    """
    Finds all residues that have at least one atom within the specified radius.
    """
    MODIFIED_AMINO_ACIDS = ['MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']
    valid_res_ids = set()
    for atom in structure.get_atoms():
        distance = np.linalg.norm(atom.coord - center)
        if distance <= radius:
            residue = atom.get_parent()
            chain = residue.get_parent()
            hetero_flag = residue.id[0]
            res_name = residue.get_resname().strip()
            if hetero_flag == ' ' or res_name in MODIFIED_AMINO_ACIDS: 
                chain = residue.get_parent()
                valid_res_ids.add((chain.id, residue.id))
            else:
                continue
    return valid_res_ids

def run_pipeline(csv_path, source_dir, base_dir):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype={'pdb_id': str})
    parser = MMCIFParser(QUIET=True)
    io = MMCIFIO()

    for _, row in df.iterrows():
        pdb_id = str(row['pdb_id']).strip()
        name = str(row['name']).strip()
        center_str = str(row['center']).strip()
        
        center_coords = parse_center(center_str)
        if center_coords is None:
            continue

        output_dir = os.path.join(base_dir, pdb_id)
        os.makedirs(output_dir, exist_ok=True)

        source_cif = os.path.join(source_dir, f"{pdb_id}.cif")
        target_full_cif = os.path.join(output_dir, f"{pdb_id}.cif")

        if not os.path.exists(source_cif):
            print(f"Warning: Source CIF missing: {source_cif}")
            continue

        if not os.path.exists(target_full_cif):
            shutil.copy(source_cif, target_full_cif)

        try:
            pocket_name = name
            pocket_dir = os.path.join(output_dir, pocket_name)
            os.makedirs(pocket_dir, exist_ok=True)

            structure = parser.get_structure(pdb_id, target_full_cif)
            valid_res_ids = get_pocket_residue_ids(structure, center_coords, 12.0)

            pocket_filename = f"{name}.cif"
            pocket_path = os.path.join(pocket_dir, pocket_filename)

            io.set_structure(structure)
            io.save(pocket_path, WholeResidueSelect(valid_res_ids))
        except Exception as e:
            print(f"Error: Failed to process {pdb_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for protein pocket extraction.")
    parser.add_argument("--input_csv", type=str, default="example_test.csv", help="Input CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(), 
                        help="The directory containing PDB_ID subfolders")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory for all structures")
    
    args = parser.parse_args()
    run_pipeline(args.input_csv, args.source_dir, args.base_dir)