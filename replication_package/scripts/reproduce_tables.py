#!/usr/bin/env python3
"""
Replication Script: Reproduces Tables 1, 2, 3, 4, and 5 from the manuscript.
"Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation"
Target: Empirical Software Engineering (Springer Nature).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

def vargha_delaney_a12(m, n):
    """
    Computes Vargha-Delaney A12 effect size.
    P(m < n) + 0.5 * P(m == n).
    > 0.5 indicates model m has lower error (is better than baseline n).
    """
    m = np.asarray(m)
    n = np.asarray(n)
    u, _ = stats.mannwhitneyu(n, m, alternative='two-sided')
    return u / (len(m) * len(n))

def round_fibonacci(arr):
    fib = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
    arr = np.asarray(arr)
    diffs = np.abs(arr[:, None] - fib[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    return fib[nearest_idx]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    splits_path = os.path.join(pkg_root, "data", "splits", "deep_se_chronological_splits.csv")
    pred_path = os.path.join(pkg_root, "data", "predictions", "predictions_chronological_test.csv")
    fact_path = os.path.join(pkg_root, "data", "predictions", "factorial_ablation_summary.csv")
    proj_path = os.path.join(pkg_root, "data", "predictions", "per_project_evaluation.csv")
    
    if not os.path.exists(pred_path):
        pred_path = os.path.join(pkg_root, "..", "effort-estimator", "artefactos_exportados", "v2", "temporal", "predicciones.csv")
    if not os.path.exists(fact_path):
        fact_path = os.path.join(pkg_root, "..", "effort-estimator", "artefactos_exportados", "etapa2", "resumen_factoriales.csv")
    if not os.path.exists(proj_path):
        proj_path = os.path.join(pkg_root, "..", "effort-estimator", "artefactos_exportados", "etapa3", "tabla_proyectos_detallada.csv")

    print("=" * 85)
    print("REPLICATION PACKAGE: EMPIRICAL BENCHMARK REPRODUCTION")
    print("Paper: Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation")
    print("=" * 85)
    
    # -------------------------------------------------------------
    # TABLE 1: DATASET DESCRIPTIVE STATISTICS
    # -------------------------------------------------------------
    if os.path.exists(splits_path):
        df_sp = pd.read_csv(splits_path)
        print("\n" + "=" * 85)
        print("TABLE 1: DEEP-SE BENCHMARK CORPUS SUMMARY (N = 23,313 ISSUES)")
        print("=" * 85)
        t1_rows = []
        for p, grp in df_sp.groupby('project'):
            sp = grp['storypoint'].values
            n_tot = len(grp)
            n_tr = len(grp[grp['split'] == 'train'])
            n_vl = len(grp[grp['split'] == 'val'])
            n_te = len(grp[grp['split'] == 'test'])
            t1_rows.append({
                'Project': p,
                'Total': n_tot,
                'Train (60%)': n_tr,
                'Val (20%)': n_vl,
                'Test (20%)': n_te,
                'Mean SP': np.mean(sp),
                'Median SP': np.median(sp),
                'IQR': np.percentile(sp, 75) - np.percentile(sp, 25),
                '% >= 13': np.mean(sp >= 13) * 100
            })
        df_t1 = pd.DataFrame(t1_rows)
        print(df_t1.to_string(index=False, float_format="%.2f"))
        print(f"Total Issues: {len(df_sp)} across {len(t1_rows)} repositories.")

    # -------------------------------------------------------------
    # TABLE 2: GLOBAL BENCHMARK RESULTS
    # -------------------------------------------------------------
    df = pd.read_csv(pred_path)
    N = len(df)
    y = df['SP_real'].values
    extreme_mask = (y >= 13)
    
    models = {
        'Baseline: Mean': df['Baseline_media'].values,
        'Baseline: Global Median': df['Baseline_mediana'].values,
        'Baseline: Project Median': df['Baseline_mediana_proyecto'].values,
        'TFIDF_LSTM_replica': df['TFIDF_LSTM_v1_replica'].values,
        'EMB_LSTM_mse_raw': df['EMB_LSTM_mse_raw'].values,
        'EMB_LSTM_v2': df['EMB_LSTM_v2'].values
    }
    
    print("\n" + "=" * 85)
    print("TABLE 2: GLOBAL BENCHMARK PERFORMANCE (CHRONOLOGICAL TEST SPLIT, N = 4,671)")
    print("=" * 85)
    
    np.random.seed(42)
    p0_pred = np.random.choice(y, size=len(y), replace=True)
    mae_p0 = np.mean(np.abs(y - p0_pred))
    
    t2_rows = []
    for name, pred in models.items():
        err = np.abs(y - pred)
        mae = np.mean(err)
        mdae = np.median(err)
        rmse = np.sqrt(np.mean((y - pred)**2))
        sa = (1 - mae / mae_p0) * 100
        bias = np.mean(pred - y)
        max_pred = np.max(pred)
        mae_extreme = np.mean(np.abs(y[extreme_mask] - pred[extreme_mask]))
        
        t2_rows.append({
            'Model': name,
            'MAE': mae,
            'MdAE': mdae,
            'RMSE': rmse,
            'SA (%)': sa,
            'Bias': bias,
            'Max Pred': max_pred,
            'MAE (>=13 SP)': mae_extreme
        })
    df_t2 = pd.DataFrame(t2_rows)
    print(df_t2.to_string(index=False, float_format="%.3f"))

    # -------------------------------------------------------------
    # TABLE 3: FACTORIAL ABLATION MATRIX ACROSS 5 SEEDS (RQ2 & RQ3)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TABLE 3: FACTORIAL ABLATION MATRIX & PROJECT-CONDITIONED ARCHITECTURES (5 SEEDS)")
    print("=" * 85)
    if os.path.exists(fact_path):
        df_fact = pd.read_csv(fact_path)
        print(df_fact.to_string(index=False, float_format="%.3f"))
    else:
        print("Factorial ablation summary file not found.")

    # -------------------------------------------------------------
    # TABLE 4: STATISTICAL HYPOTHESIS TESTING (WILCOXON + HOLM + A12)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TABLE 4: STATISTICAL HYPOTHESIS TESTING (EMB_LSTM_v2 vs BASELINES & RIVALS)")
    print("=" * 85)
    
    err_v2 = np.abs(y - models['EMB_LSTM_v2'])
    t4_rows = []
    comp_models = [
        'Baseline: Mean',
        'Baseline: Global Median',
        'Baseline: Project Median',
        'TFIDF_LSTM_replica',
        'EMB_LSTM_mse_raw'
    ]
    
    for name in comp_models:
        err_other = np.abs(y - models[name])
        w_res = stats.wilcoxon(err_v2, err_other, alternative='two-sided')
        a12 = vargha_delaney_a12(err_v2, err_other)
        mae_other = np.mean(err_other)
        
        t4_rows.append({
            'Baseline / Rival': name,
            'MAE': mae_other,
            'p_raw': w_res.pvalue,
            'A12 (v2 > rival)': a12
        })
    df_t4 = pd.DataFrame(t4_rows)
    
    # Holm-Bonferroni correction
    df_t4 = df_t4.sort_values(by='p_raw').reset_index(drop=True)
    m = len(df_t4)
    holm_p = [min(1.0, row['p_raw'] * (m - i)) for i, row in df_t4.iterrows()]
    for i in range(1, m):
        holm_p[i] = max(holm_p[i], holm_p[i-1])
        holm_p[i] = min(1.0, holm_p[i])
    df_t4['p_holm'] = holm_p
    
    def effect_label(a12):
        if a12 > 0.5:
            d = a12 - 0.5
            fav = "v2 better"
        else:
            d = 0.5 - a12
            fav = "rival better"
        if d < 0.06:
            return f"Negligible ({fav})"
        elif d < 0.14:
            return f"Small ({fav})"
        elif d < 0.21:
            return f"Medium ({fav})"
        else:
            return f"Large ({fav})"
            
    df_t4['Effect Size'] = df_t4['A12 (v2 > rival)'].apply(effect_label)
    print(df_t4[['Baseline / Rival', 'MAE', 'p_raw', 'p_holm', 'A12 (v2 > rival)', 'Effect Size']].to_string(index=False, float_format="%.4e"))

    # -------------------------------------------------------------
    # TABLE 5 & SPEARMAN CORRELATIONS: CROSS-PROJECT EVALUATION (RQ3)
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print("TABLE 5: CROSS-PROJECT PERFORMANCE BREAKDOWN & HYPOTHESIS TESTING (RQ3)")
    print("=" * 85)
    if os.path.exists(proj_path):
        df_proj = pd.read_csv(proj_path)
        print(df_proj[['project', 'n_test', 'iqr_hist', 'pct_ge13_hist', 'mae_project_median', 'mae_lstm_v2', 'delta_mae', 'p_value_holm', 'a12_lstm_vs_med']].to_string(index=False, float_format="%.3f"))
        
        # Compute Spearman rank correlations with Delta MAE
        spearman_features = [
            ('iqr_hist', 'Historical Effort Dispersion (IQR)'),
            ('pct_ge13_hist', 'Historical % Large Stories (>= 13 SP)'),
            ('med_hist', 'Historical Median Story Points'),
            ('mean_words', 'Average Requirement Length (words)'),
            ('n_train', 'Training Set Size (N_train)')
        ]
        sp_rows = []
        for feat, label in spearman_features:
            rho, p_val = stats.spearmanr(df_proj[feat], df_proj['delta_mae'])
            sp_rows.append({
                'Characteristic': label,
                'Spearman Rho': rho,
                'p-value': p_val
            })
        df_sp = pd.DataFrame(sp_rows)
        print("\n" + "=" * 85)
        print("SPEARMAN RANK CORRELATIONS WITH DELTA MAE (MAE_med - MAE_lstm)")
        print("=" * 85)
        print(df_sp.to_string(index=False, float_format="%.4f"))

if __name__ == '__main__':
    main()
