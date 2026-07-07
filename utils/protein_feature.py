import freesasa
from Bio.PDB import MMCIFParser, PDBIO, Select, is_aa
import tempfile
import os
import glob
import pandas as pd
import subprocess
from rdkit import Chem
import pickle
from collections import defaultdict
import openbabel
from openbabel import pybel
import torch
import sys
import argparse

MODIFIED_AMINO_ACIDS = ['MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']

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

def cif_to_pdb_with_mapping(cif_file, pdb_file):
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        
        original_residue_order = defaultdict(list)
        residue_counter = 0
        
        for model in structure:
            for chain in model:
                chain_id = chain.id
                for residue in chain:
                    res_id = residue.get_id()
                    res_num = res_id[1]
                    ins_code = res_id[2].strip()
                    res_name = residue.get_resname().strip()
                    
                    residue_key = (res_num, ins_code, res_name)
                    if residue_key not in original_residue_order[chain_id]:
                        original_residue_order[chain_id].append((residue_key, residue_counter))
                        residue_counter += 1
        
        chain_mapping = {}
        available_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        used_chars = set()
        
        original_chains = []
        for model in structure:
            for chain in model:
                if chain.id not in original_chains:
                    original_chains.append(chain.id)
        for chain_id in original_chains:
            if len(chain_id) == 1 and chain_id in available_chars:
                used_chars.add(chain_id)

        char_index = 0
        for orig_chain in original_chains:
            if len(orig_chain) == 1 and orig_chain in available_chars:
                continue
            else:
                while char_index < len(available_chars) and available_chars[char_index] in used_chars:
                    char_index += 1
                if char_index < len(available_chars):
                    new_chain = available_chars[char_index]
                    chain_mapping[orig_chain] = new_chain
                    used_chars.add(new_chain)
                    char_index += 1
                else:
                    chain_mapping[orig_chain] = orig_chain[0] if orig_chain else 'A'
        
        for model in structure:
            for chain in model:
                if chain.id in chain_mapping:
                    chain.id = chain_mapping[chain.id]
        
        io = PDBIO()
        io.set_structure(structure)
        selector = ProteinSelect()
        io.save(pdb_file, selector)
        
        return True, chain_mapping, original_residue_order
        
    except Exception as e:
        print(f"{cif_file}:{e}")
        return False, {}, {}
    
def calculate_sasa_from_cif_to_pt(cif_file):
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_pdb:
        temp_pdb_path = temp_pdb.name
    
    try:
        success, chain_mapping, original_residue_order = cif_to_pdb_with_mapping(
            cif_file, temp_pdb_path
        )
        
        if not success:
            return {}

        options = {
            'hetatm': True,   
            'hydrogen': False, 
            'join-models': False,
            'skip-unknown': False,
            'halt-at-unknown': False
        }
        
        pdb_structure = freesasa.Structure(temp_pdb_path, options=options)
        result = freesasa.calc(pdb_structure)

        residue_dict = {}
        reverse_mapping = {v: k for k, v in chain_mapping.items()}
        
        for i in range(pdb_structure.nAtoms()):
            chain = pdb_structure.chainLabel(i)
            res_num = pdb_structure.residueNumber(i)
            res_num = res_num.strip()
            orig_chain = reverse_mapping.get(chain, chain)
            
            residue_key = f"{orig_chain}_{res_num}"
            residue_dict[residue_key] = residue_dict.get(residue_key, 0.0) + result.atomArea(i)
        
        pdb_id = os.path.splitext(os.path.basename(cif_file))[0].lower()
        final_sasa_dict = {}
        for chain_id, residue_list in original_residue_order.items():
            sasa_values = []

            for residue_key, order_index in residue_list:
                res_num = residue_key[0]  
                ins_code = residue_key[1]                  

                if ins_code and ins_code != ' ' and ins_code != '':
                    lookup_key = f"{chain_id}_{res_num}{ins_code}"
                else:
                    lookup_key = f"{chain_id}_{res_num}"
                
                if lookup_key in residue_dict:
                    sasa_value = residue_dict.get(lookup_key)
                    sasa_values.append(sasa_value)
            
            if sasa_values:
                outer_key = f"{pdb_id}_{chain_id}"
                final_sasa_dict[outer_key] = torch.tensor(sasa_values, dtype=torch.float32)
        
        return final_sasa_dict
        
    except Exception as e:
        print(f"Error: Failed to process {cif_file} : {e}")
        import traceback
        traceback.print_exc()
        return {}
        
    finally:
        if os.path.exists(temp_pdb_path):
            os.remove(temp_pdb_path)
        
def process_with_reduce(input_pdb, output_pdb, cif_file):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        return_code = subprocess.run(
            f"reduce -Trim {input_pdb} > {tmp_path}", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        return_code = subprocess.run(
            f"reduce -HIS {tmp_path} > {output_pdb}", 
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        return True
        
    except Exception as e:
        print(f"Error: RDKit failed - {cif_file}: {e}")
        return False

def are_residues_adjacent(original_residue_order, chain_id, res1, ins1, res2, ins2):
    if chain_id not in original_residue_order:
        return False
    
    residues = original_residue_order[chain_id]
    
    idx1 = None
    idx2 = None
    
    for i, ((res_num, ins_code, res_name), order) in enumerate(residues):
        if res_num == res1 and ins_code == ins1:
            idx1 = order
        if res_num == res2 and ins_code == ins2:
            idx2 = order
            
        if idx1 is not None and idx2 is not None:
            break
    
    if idx1 is None or idx2 is None:
        return False
    
    return abs(idx1 - idx2) == 1

def is_disulfide_bond(atom1, atom2, pdb_info1, pdb_info2, mol):
    if (pdb_info1.GetName().strip() == 'SG' and 
        pdb_info2.GetName().strip() == 'SG' and
        pdb_info1.GetResidueName().strip() == 'CYS' and 
        pdb_info2.GetResidueName().strip() == 'CYS'):
        return True    
    return False

def clean_protein_bonds(mol, original_residue_order, chain_mapping):
    rw_mol = Chem.RWMol(mol)
    
    reverse_chain_mapping = {v: k for k, v in chain_mapping.items()}
    
    atom_infos = []
    for atom in rw_mol.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        atom_infos.append(pdb_info)
    
    bonds_to_remove = []
    for bond in rw_mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        
        info1 = atom_infos[atom1_idx]
        info2 = atom_infos[atom2_idx]
        
        if info1 is None or info2 is None:
            continue
        chain1, resi1, ins1 = info1.GetChainId(), info1.GetResidueNumber(), info1.GetInsertionCode().strip()
        chain2, resi2, ins2 = info2.GetChainId(), info2.GetResidueNumber(), info2.GetInsertionCode().strip()
        
        orig_chain1 = reverse_chain_mapping.get(chain1, chain1)
        orig_chain2 = reverse_chain_mapping.get(chain2, chain2)
        
        if orig_chain1 == orig_chain2 and resi1 == resi2 and ins1 == ins2:
            continue
            
        if (orig_chain1 == orig_chain2 and 
            are_residues_adjacent(original_residue_order, orig_chain1, resi1, ins1, resi2, ins2)):
            continue
            
        if is_disulfide_bond(
            rw_mol.GetAtomWithIdx(atom1_idx),
            rw_mol.GetAtomWithIdx(atom2_idx),
            info1, info2, mol
        ):
            continue
            
        bonds_to_remove.append((atom1_idx, atom2_idx))
    
    for atom1_idx, atom2_idx in sorted(bonds_to_remove, reverse=True):
        rw_mol.RemoveBond(atom1_idx, atom2_idx)

    return rw_mol.GetMol()

def calculate_pybel_features(pdb_file, full_protein_mol, cif_file):
    try:
        rdkit_atom_mapping = {}
        for atom_idx, atom in enumerate(full_protein_mol.GetAtoms()):
            pdb_info = atom.GetPDBResidueInfo()
            if not pdb_info:
                continue               
            current_chain = pdb_info.GetChainId()
            current_res_num = pdb_info.GetResidueNumber()
            current_ins_code = pdb_info.GetInsertionCode().strip()
            current_atom_name = pdb_info.GetName().strip()

            if current_ins_code:
                atom_key = f"{current_chain}_{current_res_num}_{current_ins_code}_{current_atom_name}"
            else:
                atom_key = f"{current_chain}_{current_res_num}_{current_atom_name}"
                
            rdkit_atom_mapping[atom_key] = atom_idx

        mol = next(pybel.readfile("pdb", pdb_file))     

        num_atoms = full_protein_mol.GetNumAtoms()
        pybel_features_tensor = torch.zeros(num_atoms, 3, dtype=torch.float)
        
        for atom in mol:
            if atom.atomicnum > 1:
                obatom = atom.OBAtom
                residue = obatom.GetResidue()
                
                current_chain = residue.GetChain()
                
                res_num = residue.GetNum()
                insertion_code = residue.GetInsertionCode()
                atom_name = residue.GetAtomID(obatom).strip()

                charge = atom.partialcharge
                hbond_donor = 1 if obatom.IsHbondDonor() else 0
                hbond_acceptor = 1 if obatom.IsHbondAcceptor() else 0
                
                res_num_str = str(res_num).replace('\x00', '').strip()
                insertion_code_clean = insertion_code.replace('\x00', '').strip()
                atom_name_clean = atom_name.replace('\x00', '').strip()
                
                if insertion_code_clean:  
                    atom_key = f"{current_chain}_{res_num}_{insertion_code_clean}_{atom_name_clean}"
                else:
                    atom_key = f"{current_chain}_{res_num}_{atom_name_clean}"

                if atom_key in rdkit_atom_mapping:
                    atom_idx = rdkit_atom_mapping[atom_key]
                    features_tensor = torch.tensor([charge, hbond_donor, hbond_acceptor], dtype=torch.float)
                    pybel_features_tensor[atom_idx] = features_tensor
        
        return pybel_features_tensor, rdkit_atom_mapping
    
    except Exception as e:
        print(f"Error: Pybel failed - {cif_file}: {e}")
        return None, {}

def create_protein_pkl(cif_file, output_pkl):
    pdb_id = os.path.basename(cif_file).split('.')[0]
    
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_pdb1:
        tmp_pdb1_path = tmp_pdb1.name
    
    success, chain_mapping, original_residue_order = cif_to_pdb_with_mapping(cif_file, tmp_pdb1_path)
    if not success:
        print(f"Error: CIF to PDB failed-{cif_file}")
        os.unlink(tmp_pdb1_path)
        return False
    
    with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_pdb2:
        tmp_pdb2_path = tmp_pdb2.name
    
    success = process_with_reduce(tmp_pdb1_path, tmp_pdb2_path, cif_file)
    os.unlink(tmp_pdb1_path)
    if not success:
        print("Error: Reduce failed")
        os.unlink(tmp_pdb2_path)
        return False

    mol = Chem.MolFromPDBFile(tmp_pdb2_path, removeHs=False, sanitize=False, proximityBonding=True)
    
    if mol is None:
        print(f"Error: RDKit failed - {cif_file}")
        return False
    
    pybel_features, rdkit_atom_mapping = calculate_pybel_features(tmp_pdb2_path, mol, cif_file)
    os.unlink(tmp_pdb2_path)
    
    if pybel_features is None:
        print(f"Error: Pybel failed - {cif_file}")
        return False
    
    reverse_chain_mapping = {v: k for k, v in chain_mapping.items()}
    original_atom_mapping = {}
    for atom_key, atom_idx in rdkit_atom_mapping.items():
        parts = atom_key.split('_')
        mapped_chain = parts[0]
        original_chain = reverse_chain_mapping.get(mapped_chain, mapped_chain)
        new_key = f"{original_chain}_{'_'.join(parts[1:])}"
        original_atom_mapping[new_key] = atom_idx

    cleaned_mol = clean_protein_bonds(mol, original_residue_order, chain_mapping)

    try:
        Chem.SanitizeMol(cleaned_mol)
    except Exception as e:
        print(f"Warning: Valence check failed but continuing: {e} - {cif_file}")

    result_dict = {
        'mol': cleaned_mol,
        'pybel_features': pybel_features,  
        'atom_mapping': original_atom_mapping, 
        'pdb_id': pdb_id
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(result_dict, f)
    return True

def run_pipeline(csv_path, base_working_dir):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    df['pdb_id'] = df['name'].str.split('_').str[0]
    unique_pdb_ids = df['pdb_id'].unique()
    failed_pdbs = []

    for pdb_id in unique_pdb_ids:
        pdb_id = str(pdb_id).strip()
        output_file_r = os.path.join(base_working_dir, pdb_id, f"{pdb_id}_r.pt")
        output_file_a = os.path.join(base_working_dir, pdb_id, f"{pdb_id}_a.pkl")
        protein_cif_path = os.path.join(base_working_dir, pdb_id, f"{pdb_id}.cif")

        if os.path.exists(protein_cif_path):
            try:
                sasa_result = calculate_sasa_from_cif_to_pt(protein_cif_path)
                torch.save(sasa_result, output_file_r)
                create_protein_pkl(protein_cif_path, output_file_a)
            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                failed_pdbs.append(pdb_id)
        else:
            print(f"Warning: Full protein file for {pdb_id} not found at {protein_cif_path}")
    return failed_pdbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline for protein feature extraction.')
    parser.add_argument("--input_csv", type=str, default="example_test.csv", help="Input CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(), 
                        help="The directory containing PDB_ID subfolders")
    args = parser.parse_args()
    failed = run_pipeline(args.input_csv, args.base_dir)
    if failed:
        print(f"Warning: {len(failed)} PDB(s) failed: {failed}", file=sys.stderr)
