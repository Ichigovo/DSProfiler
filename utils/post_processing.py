import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Post-process DSProfiler results")
    parser.add_argument("--output_csv", type=str, default="tmp_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(),
                        help="The directory containing PDB_ID subfolders")
    parser.add_argument("--final_csv", type=str, default="example_results.csv",
                        help="Final CSV file path")
    args = parser.parse_args()

    if not os.path.exists(args.output_csv):
        print(f"Error: {args.output_csv} not found.")
        return

    df_ds = pd.read_csv(args.output_csv)
    required_cols = ['name', 'label', 'score']
    if not all(col in df_ds.columns for col in required_cols):
        print(f"Error: {args.output_csv} must contain columns: {required_cols}")
        return

    df_ds[['pdb_id', 'p2rank_rank']] = df_ds['name'].str.rsplit('_', n=1, expand=True)

    df_ds = df_ds.drop(columns=['label'])
    df_ds = df_ds.rename(columns={'score': 'DSProfiler_score'})

    df_ds['DSProfiler_rank'] = df_ds.groupby('pdb_id')['DSProfiler_score']\
                                     .rank(method='first', ascending=False)\
                                     .astype(int)

    p2rank_path = os.path.join(args.base_dir, "p2rank_pockets.csv")
    if not os.path.exists(p2rank_path):
        print(f"Error: p2rank_pockets.csv not found at {p2rank_path}")
        return

    df_p2 = pd.read_csv(p2rank_path)
    if 'name' not in df_p2.columns or 'center' not in df_p2.columns:
        print("Error: p2rank_pockets.csv must contain 'name' and 'center' columns")
        return

    df_merged = df_ds.merge(df_p2[['name', 'center']], on='name', how='inner')

    final_df = df_merged[['pdb_id', 'DSProfiler_rank', 'DSProfiler_score', 'center']]
    final_df = final_df.sort_values(by=['pdb_id', 'DSProfiler_rank'])

    final_df.to_csv(args.final_csv, index=False)

    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

if __name__ == "__main__":
    main()