import pandas as pd
import numpy as np
import os
import subprocess
import tempfile
import shutil
import warnings
import argparse
from pathlib import Path
from Bio.PDB import MMCIFParser, MMCIFIO, Select
from tqdm import tqdm

def remove_hetatm_from_cif(input_cif, output_cif):
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('protein', input_cif)

        class HetatmRemover(Select):
            def accept_residue(self, residue):
                return residue.id[0] == ' ' 
        
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(output_cif, select=HetatmRemover())
        return True
    except Exception as e:
        print(f"Error: Preprocessing failed {os.path.basename(input_cif)}: {e}")
        return False

def parse_center_string(center_str):
    try:
        coords = center_str.strip('()').split(',')
        return np.array([float(x.strip()) for x in coords])
    except:
        return None

def run_p2rank(cif_file, tmp_dir):
    try:
        output_dir = os.path.join(tmp_dir, "p2rank_output")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "./run_p2rank/prank", "predict",
            "-c", "alphafold",
            "-f", cif_file,
            "-visualizations", "0",
            "-o", output_dir
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=900
        )
        
        if result.returncode != 0:
            return None

        for file in Path(output_dir).glob("*predictions.csv"):
            return str(file)
        return None
        
    except Exception as e:
        warnings.warn(f"Error: P2Rank failed: {e}")
        return None

def process_structure(cif_path, pdb_id, tmp_base_dir):    
    results = []
    with tempfile.TemporaryDirectory(dir=tmp_base_dir) as tmp_dir:
        clean_cif = os.path.join(tmp_dir, f"{pdb_id}_clean.cif")
        if not remove_hetatm_from_cif(cif_path, clean_cif):
            clean_cif = cif_path   

        pred_file = run_p2rank(clean_cif, tmp_dir)
        if pred_file and os.path.exists(pred_file):
            try:
                pred_df = pd.read_csv(pred_file)
                pred_df.columns = [c.strip() for c in pred_df.columns]
                required_cols = ['rank', 'probability', 'center_x', 'center_y', 'center_z']
                if not all(col in pred_df.columns for col in required_cols):
                    return []

                for _, p_row in pred_df.iterrows():
                    p_center = np.array([p_row['center_x'], p_row['center_y'], p_row['center_z']])
                    results.append({
                        'pdb_id': pdb_id,
                        'P2Rank_rank': int(p_row['rank']),
                        'P2Rank_score': p_row['probability'],
                        'center': f"({p_center[0]:.3f},{p_center[1]:.3f},{p_center[2]:.3f})"
                    })
            except Exception as e:
                warnings.warn(f"Error: Failed to parse results for {pdb_id}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract pockets by P2Rank")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(),
                        help="The directory containing PDB_ID subfolders")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory for all structures")
    parser.add_argument("--input_csv", type=str, default="p2rank_pockets.csv",
                        help="Input_csv for next step")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_csv = args.input_csv
    tmp_base_dir = os.path.join(args.base_dir, "tmp_p2rank")

    os.makedirs(tmp_base_dir, exist_ok=True)

    if not os.path.isdir(source_dir):
        print(f"Error: Structure directory does not exist: {source_dir}")
        return

    cif_files = list(Path(source_dir).glob("*.cif"))
    if not cif_files:
        print(f"Error: No cif files found in {source_dir}")
        return

    all_pockets = []
    for cif_path in tqdm(cif_files, unit="file"):
        pdb_id = cif_path.stem
        pockets = process_structure(str(cif_path), pdb_id, tmp_base_dir)
        all_pockets.extend(pockets)

    if all_pockets:
        df = pd.DataFrame(all_pockets)
        df['name'] = df.apply(lambda row: f"{row['pdb_id']}_{row['P2Rank_rank']}", axis=1)
        df['label'] = ""
        df = df.sort_values(by=['pdb_id', 'P2Rank_rank'])
        df = df[['name', 'pdb_id', 'P2Rank_rank', 'P2Rank_score', 'center', 'label']]
        df.to_csv(output_csv, index=False)
    else:
        print("\nNo pockets extracted.")

    if os.path.exists(tmp_base_dir):
        shutil.rmtree(tmp_base_dir)

if __name__ == "__main__":
    main()