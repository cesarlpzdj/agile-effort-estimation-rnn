import os
import glob
import json
import shutil
import pandas as pd
import numpy as np

def main():
    base_dir = "replication_package"
    os.makedirs(f"{base_dir}/data/splits", exist_ok=True)
    os.makedirs(f"{base_dir}/data/predictions", exist_ok=True)
    os.makedirs(f"{base_dir}/scripts", exist_ok=True)
    
    # 1. Generate master splits file (train, val, test)
    project_files = glob.glob("effort-estimator/dataset/*.csv")
    dfs = []
    for f in project_files:
        pname = os.path.splitext(os.path.basename(f))[0].lower()
        pdf = pd.read_csv(f)
        pdf['project'] = pname
        dfs.append(pdf)
    df_all = pd.concat(dfs, ignore_index=True)
    
    def extract_key_num(k):
        try:
            return int(k.split('-')[-1])
        except:
            return 0
    df_all['key_num'] = df_all['issuekey'].apply(extract_key_num)
    df_all = df_all.sort_values(by=['project', 'key_num']).reset_index(drop=True)
    
    split_info = []
    for pname, group in df_all.groupby('project', sort=False):
        n = len(group)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        group = group.copy()
        group['split'] = ['train']*n_train + ['val']*n_val + ['test']*n_test
        split_info.append(group[['issuekey', 'project', 'storypoint', 'split']])
    df_splits = pd.concat(split_info, ignore_index=True)
    splits_csv = f"{base_dir}/data/splits/deep_se_chronological_splits.csv"
    df_splits.to_csv(splits_csv, index=False)
    print(f"Generated {splits_csv} with {len(df_splits)} issues.")
    
    # 2. Copy predictions
    src_pred = "effort-estimator/artefactos_exportados/v2/temporal/predicciones.csv"
    dst_pred = f"{base_dir}/data/predictions/predictions_chronological_test.csv"
    if os.path.exists(src_pred):
        shutil.copyfile(src_pred, dst_pred)
        print(f"Copied predictions to {dst_pred}")
        
    # Copy factorial summary
    src_fact = "effort-estimator/artefactos_exportados/etapa2/resumen_factoriales.csv"
    dst_fact = f"{base_dir}/data/predictions/factorial_ablation_summary.csv"
    if os.path.exists(src_fact):
        shutil.copyfile(src_fact, dst_fact)
        print(f"Copied factorial summary to {dst_fact}")
        
    # Copy project detailed table
    src_proj = "effort-estimator/artefactos_exportados/etapa3/tabla_proyectos_detallada.csv"
    dst_proj = f"{base_dir}/data/predictions/per_project_evaluation.csv"
    if os.path.exists(src_proj):
        shutil.copyfile(src_proj, dst_proj)
        print(f"Copied per-project evaluation to {dst_proj}")

if __name__ == '__main__':
    main()
