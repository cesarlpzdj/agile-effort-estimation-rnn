import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import stats

def vargha_delaney_a12(m, n):
    """
    Computes Vargha-Delaney A12 effect size.
    m: errors from model 1
    n: errors from model 2 (baseline)
    A12 > 0.5 means m has larger values (worse if metric is error).
    To align with conventional convention: A12 of model 1 being better than model 2:
    P(Error_m < Error_n) + 0.5 * P(Error_m == Error_n)
    """
    m = np.asarray(m)
    n = np.asarray(n)
    r = stats.rankdata(np.concatenate([m, n]))
    r1 = r[:len(m)]
    # A12 = (sum(r1) / len(m) - (len(m) + 1) / 2) / len(n)
    # This is P(m > n). Let's compute P(baseline_err > model_err) so > 0.5 means model is better:
    u, _ = stats.mannwhitneyu(n, m, alternative='two-sided')
    a12 = u / (len(m) * len(n))
    return a12

def main():
    os.makedirs("effort-estimator/artefactos_exportados/etapa3", exist_ok=True)
    
    # 1. Load predictions
    pred_path = "effort-estimator/artefactos_exportados/v2/temporal/predicciones.csv"
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found.")
        return
    df_pred = pd.read_csv(pred_path)
    
    # 2. Load raw datasets to get text lengths and full dataset characteristics
    project_files = glob.glob("effort-estimator/dataset/*.csv")
    project_dfs = {}
    for f in project_files:
        pname = os.path.splitext(os.path.basename(f))[0].lower()
        pdf = pd.read_csv(f)
        pdf['project_name'] = pname
        project_dfs[pname] = pdf
        
    df_all = pd.concat(project_dfs.values(), ignore_index=True)
    df_all['title'] = df_all['title'].fillna('')
    df_all['description'] = df_all['description'].fillna('')
    df_all['full_text'] = df_all['title'] + " " + df_all['description']
    df_all['word_count'] = df_all['full_text'].apply(lambda x: len(x.split()))
    
    # Sort chronologically by issue key number per project (matching the splitting rule in fase2_lstm_v2.py)
    # In Deep-SE: issue keys are like 'PROJ-123'
    def extract_key_num(k):
        try:
            return int(k.split('-')[-1])
        except:
            return 0
    df_all['key_num'] = df_all['issuekey'].apply(extract_key_num)
    df_all = df_all.sort_values(by=['project_name', 'key_num']).reset_index(drop=True)
    
    # Identify splits
    split_info = []
    for pname, group in df_all.groupby('project_name', sort=False):
        n = len(group)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)
        n_test = n - n_train - n_val
        group = group.copy()
        group['split'] = ['train']*n_train + ['val']*n_val + ['test']*n_test
        split_info.append(group)
    df_split = pd.concat(split_info, ignore_index=True)
    
    # Project characteristics
    proj_rows = []
    wilcox_rows = []
    
    for pname in sorted(df_pred['Project'].unique()):
        sub_pred = df_pred[df_pred['Project'] == pname].copy()
        sub_all = df_split[df_split['project_name'] == pname]
        sub_train = sub_all[sub_all['split'] == 'train']
        sub_history = sub_all[sub_all['split'].isin(['train', 'val'])]
        sub_test = sub_all[sub_all['split'] == 'test']
        
        n_test = len(sub_pred)
        n_train = len(sub_train)
        n_history = len(sub_history)
        
        # Ground truth distributions
        sp_test = sub_pred['SP_real'].values
        sp_hist = sub_history['storypoint'].values
        
        med_hist = np.median(sp_hist)
        med_test = np.median(sp_test)
        q75, q25 = np.percentile(sp_hist, [75, 25])
        iqr_hist = q75 - q25
        pct_ge13_hist = np.mean(sp_hist >= 13) * 100
        pct_ge13_test = np.mean(sp_test >= 13) * 100
        
        mean_words = sub_all['word_count'].mean()
        
        # Absolute errors
        err_lstm = np.abs(sub_pred['SP_real'] - sub_pred['EMB_LSTM_v2'])
        err_med_proj = np.abs(sub_pred['SP_real'] - sub_pred['Baseline_mediana_proyecto'])
        err_mean_base = np.abs(sub_pred['SP_real'] - sub_pred['Baseline_media'])
        
        mae_lstm = np.mean(err_lstm)
        mae_med = np.mean(err_med_proj)
        mae_mean = np.mean(err_mean_base)
        
        delta_mae = mae_med - mae_lstm  # > 0 means LSTM is better
        ratio_mae = mae_lstm / mae_med   # < 1 means LSTM is better
        
        # Wilcoxon test: paired err_lstm vs err_med_proj
        # Note: if all differences are zero, handle edge case
        diff = err_med_proj - err_lstm
        if np.all(diff == 0):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(err_lstm, err_med_proj, alternative='two-sided')
                w_stat, p_val = w_res.statistic, w_res.pvalue
            except Exception as e:
                w_stat, p_val = 0.0, 1.0
                
        # A12: P(err_med > err_lstm) + 0.5 * P(err_med == err_lstm)
        # > 0.5 means LSTM has smaller errors (is better)
        a12 = vargha_delaney_a12(err_lstm, err_med_proj)
        
        proj_rows.append({
            'project': pname,
            'n_train': n_train,
            'n_test': n_test,
            'med_hist': med_hist,
            'iqr_hist': iqr_hist,
            'pct_ge13_hist': pct_ge13_hist,
            'pct_ge13_test': pct_ge13_test,
            'mean_words': mean_words,
            'mae_mean': mae_mean,
            'mae_project_median': mae_med,
            'mae_lstm_v2': mae_lstm,
            'delta_mae': delta_mae,
            'ratio_mae': ratio_mae,
            'wilcoxon_stat': w_stat,
            'p_value_raw': p_val,
            'a12_lstm_vs_med': a12
        })

    df_res = pd.DataFrame(proj_rows)
    
    # Holm-Bonferroni correction on p-values
    df_res = df_res.sort_values(by='p_value_raw').reset_index(drop=True)
    m = len(df_res)
    holm_p = []
    for i, row in df_res.iterrows():
        adj_p = min(1.0, row['p_value_raw'] * (m - i))
        holm_p.append(adj_p)
    # Enforce monotonicity
    for i in range(1, m):
        holm_p[i] = max(holm_p[i], holm_p[i-1])
        holm_p[i] = min(1.0, holm_p[i])
    df_res['p_value_holm'] = holm_p
    
    # Sort back by project name
    df_res = df_res.sort_values(by='project').reset_index(drop=True)
    df_res.to_csv("effort-estimator/artefactos_exportados/etapa3/tabla_proyectos_detallada.csv", index=False)
    
    # 3. Spearman Rank Correlations
    spearman_features = [
        ('iqr_hist', 'Historical Effort Dispersion (IQR)'),
        ('pct_ge13_hist', 'Historical % Large Stories (>= 13 SP)'),
        ('med_hist', 'Historical Median Story Points'),
        ('mean_words', 'Average Requirement Length (words)'),
        ('n_train', 'Training Set Size (N_train)')
    ]
    
    spearman_rows = []
    for feat, label in spearman_features:
        rho_delta, p_delta = stats.spearmanr(df_res[feat], df_res['delta_mae'])
        rho_ratio, p_ratio = stats.spearmanr(df_res[feat], df_res['ratio_mae'])
        spearman_rows.append({
            'feature': feat,
            'label': label,
            'spearman_rho_delta': rho_delta,
            'p_val_delta': p_delta,
            'spearman_rho_ratio': rho_ratio,
            'p_val_ratio': p_ratio
        })
    df_spearman = pd.DataFrame(spearman_rows)
    df_spearman.to_csv("effort-estimator/artefactos_exportados/etapa3/correlaciones_spearman.csv", index=False)
    
    print("=== TABLA POR PROYECTO DETALLADA ===")
    print(df_res[['project', 'n_test', 'iqr_hist', 'pct_ge13_hist', 'mae_project_median', 'mae_lstm_v2', 'delta_mae', 'p_value_raw', 'p_value_holm', 'a12_lstm_vs_med']].to_string())
    
    print("\n=== CORRELACIONES DE SPEARMAN CON DELTA MAE (MAE_med - MAE_lstm) ===")
    print(df_spearman[['label', 'spearman_rho_delta', 'p_val_delta']].to_string())

if __name__ == '__main__':
    main()
