# Replication Package: Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23788447.svg)](https://doi.org/10.5281/zenodo.23788447)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete replication package for the study:
> **Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation**  
> *Target Journal:* Empirical Software Engineering (Springer Nature)  
> *Authors:* César López de Jesús, Karla Miroslava Olmos Sánchez  
> *Affiliation:* Maestría en Inteligencia Artificial y Analítica de Datos (UACJ)  
> *Archived DOI:* [10.5281/zenodo.23788447](https://doi.org/10.5281/zenodo.23788447)

---

## 1. Overview and Package Structure

This package enables the full, independent reproduction of all empirical benchmarks, hypothesis tests, factorial ablations, and per-project analyses reported in the manuscript, as well as running the dual-indicator decision support prototype.

```
replication_package/
├── data/
│   ├── splits/
│   │   └── deep_se_chronological_splits.csv   # Master split mapping for 23,313 issues (train/val/test)
│   └── predictions/
│       ├── predictions_chronological_test.csv # Model and baseline predictions on test set (N=4,671)
│       ├── factorial_ablation_summary.csv     # Results of 2x2x2 factorial matrix across 5 seeds
│       └── per_project_evaluation.csv         # Per-project metrics and non-parametric tests
├── scripts/
│   ├── reproduce_tables.py                    # Standalone runner for Tables 2, 3, 5, and 6
│   └── run_factorial_ablation.py              # Script to rerun 50 factorial model trainings
├── requirements.txt                           # Pinned Python package dependencies
├── LICENSE                                    # MIT Open Source License
└── README.md                                  # This replication guide
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested with Python 3.11 / 3.12)
- Virtual environment tool (`venv` or `uv`)

```bash
# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 3. Fast Reproduction of Manuscript Results

To verify the tables and statistical tests in under 10 seconds using the precomputed test set predictions:

```bash
python scripts/reproduce_tables.py
```

This script reproduces:
1. **Table 1 (Corpus Descriptive Statistics):** Summary of all 23,313 issues across 16 repositories (sample sizes, splits, median, IQR, % large stories).
2. **Table 2 (Global Benchmark Results):** MAE, MdAE, RMSE, SA (%), Prediction Bias, Maximum Predicted Value, and Extreme-tier MAE ($\ge 13$ SP) for all baseline heuristics and neural models.
3. **Table 3 (Factorial Ablation Matrix across 5 Seeds):** Disentangling preprocessing, target scale, loss function, and project-conditioned bimodal models.
4. **Table 4 (Statistical Hypothesis Testing):** Two-tailed paired Wilcoxon signed-rank test, Holm-Bonferroni adjusted $p$-values, and Vargha-Delaney $\hat{A}_{12}$ effect sizes.
5. **Table 5 (Cross-Project Heterogeneity & RQ3):** Per-project sample size, historical dispersion (IQR), % large stories ($\ge 13$ SP), Project Median MAE, $\text{EMB\_LSTM}_{\text{v2}}$ MAE, $\Delta\text{MAE}$, Holm-adjusted $p$-values, and $\hat{A}_{12}$.
6. **Spearman Rank Correlations:** Correlation between project distribution characteristics and neural model advantage ($\Delta\text{MAE}$).

---

## 4. Rerunning the Factorial Ablation Matrix (50 Models across 5 Seeds)

To train and evaluate all 10 configurations across 5 independent random seeds (`[42, 101, 202, 303, 404]`):

```bash
python scripts/run_factorial_ablation.py
```
*Note:* Full factorial training takes approximately 45–60 minutes on modern multi-core CPUs. The outputs are saved to `data/predictions/runs_factoriales.csv` and `data/predictions/factorial_ablation_summary.csv`.

---

## 5. Running the Decision Support Prototype

The repository also includes the reference implementation of the decision support prototype:

### Backend API (FastAPI)
```bash
cd ../effort-estimator
python api.py
# The service starts at http://localhost:8000
# OpenAPI documentation: http://localhost:8000/docs
```

### Frontend Client (React + Vite + Tailwind CSS)
```bash
cd ../frontend-estimator
npm install
npm run dev
# The SPA is available at http://localhost:5173
```

---

## 6. Dataset Citation and Attribution

The benchmark utilizes the clean, consolidated **Deep-SE** dataset collected by Choetkiertikul et al. and curated by Tawosi et al.:
- Choetkiertikul, M., et al. (2019). *A deep learning model for estimating story points.* IEEE Transactions on Software Engineering, 45(7), 637-656.
- Tawosi, V., et al. (2023). *On the use of pre-trained language models for effort estimation.* IEEE Transactions on Software Engineering, 49(5), 3241-3259.
- Repository: `https://github.com/morakotch/datasets`

---

## 7. License

This replication package is released under the **MIT License**. See `LICENSE` for details.
