# Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23788447.svg)](https://doi.org/10.5281/zenodo.23788447)
[![Journal](https://img.shields.io/badge/Target-Empirical%20Software%20Engineering%20(Springer%20Nature)-blue.svg)](https://www.springer.com/journal/10664)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-blue.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete empirical replication package, deep learning pipelines, and decision support prototype for the research paper:

> **Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation**  
> **Authors:** César López de Jesús, Karla Miroslava Olmos Sánchez  
> **Affiliation:** Maestría en Inteligencia Artificial y Analítica de Datos (MIAAD), Universidad Autónoma de Ciudad Juárez (UACJ)  
> **Target Journal:** *Empirical Software Engineering* (Springer Nature)

---

## 1. Abstract and Key Empirical Findings

Accurate effort estimation is critical for sprint planning in agile software engineering. While deep neural networks have been widely proposed to predict Story Points from issue text, empirical replication studies question whether complex natural language models reliably outperform simple project-level heuristics under realistic chronological constraints. 

In this study, we present a rigorous empirical evaluation of recurrent neural networks using the complete Deep-SE benchmark (**23,313 issues across 16 open-source projects**) under both standard random (80/20) and project-wise chronological (60/20/20) partitions, with all data-dependent transformations fitted strictly on training data to eliminate leakage. 

### Key Findings:
1. **Practical Equivalence with Global Heuristics (RQ1):** Under chronological evaluation, the sequential text LSTM (`EMB_LSTM_v2`) achieves an MAE of $3.22 \pm 0.04$ (Fibonacci MAE $3.17 \pm 0.03$), statistically outperforming naive mean guessing (MAE 4.26) and leak-free TF-IDF recurrent replicas (MAE 3.78) ($p < 10^{-26}$). However, it is practically equivalent to the zero-parameter Global Median heuristic (MAE 3.26, $\hat{A}_{12} = 0.520$) and is outperformed by the historical Project Median heuristic (MAE 2.98).
2. **Factorial Disentanglement of Loss vs. Scale vs. Preprocessing (RQ2):** A full $2 \times 2 \times 2$ factorial experiment across 5 random seeds (50 trained models) reveals that **Huber loss ($\delta = 1.0$) is the primary driver of regression stability**, reducing MAE on raw Story Points from $4.23$ to $3.17$ SP. Furthermore, models optimizing quadratic MSE on raw targets achieve the lowest extreme-tier error ($\text{MAE}_{\ge 13} = 16.59 \pm 0.71$ SP), whereas Huber loss and logarithmic compression trade extreme-tier accuracy to minimize global loss.
3. **Closing the Gap via Project Conditioning (RQ3):** Incorporating categorical project embeddings into a bimodal neural architecture ($\text{EMB\_LSTM}_{\text{proj}}$) achieves a test MAE of **$3.06 \pm 0.05$** (and Fibonacci MAE **$3.01 \pm 0.02$**), closing the performance gap with the historical project median to within $0.08$ SP.
4. **Cross-Project Dichotomy & Heavy Tails:** Non-parametric Spearman rank correlations reveal that the neural model's advantage ($\Delta\text{MAE}$) correlates strongly and significantly with the proportion of large stories $\ge 13$ SP ($\rho = 0.544, p = 0.029$). In high-variance repositories with complex tasks (Moodle: $\text{IQR} = 15.0$, 45.4% $\ge 13$ SP), the LSTM significantly outperforms the Project Median heuristic (MAE $5.43$ vs. $6.48$, $p_{\text{Holm}} = 2.1 \times 10^{-4}$, $\hat{A}_{12} = 0.647$, medium effect).
5. **Temporal Distribution Shift:** Kolmogorov-Smirnov tests across project timelines reveal a significant distribution shift ($p < 10^{-8}$) where tasks mature into smaller granularities over time (Train: mean 6.71 SP; Test: mean 5.06 SP), explaining early validation plateaus during model training.

---

## 2. Repository Structure

```text
.
├── replication_package/        # Self-contained open-science replication package (< 10s execution)
│   ├── data/
│   │   ├── dataset/            # 16 raw Deep-SE project CSVs
│   │   ├── splits/             # Deterministic split mappings for all 23,313 issues
│   │   └── predictions/        # Test-set predictions and factorial ablation matrices
│   ├── scripts/
│   │   ├── reproduce_tables.py # Reproduces Tables 1, 2, 3, 4, and 5 in ~2 seconds
│   │   └── run_factorial_ablation.py # Trains and evaluates 50 models across 5 seeds
│   ├── requirements.txt        # Pinned Python dependencies
│   ├── LICENSE                 # MIT License
│   └── README.md               # Replication instructions
│
├── effort-estimator/           # Backend inference microservice & experimental ML pipelines
│   ├── api.py                  # FastAPI inference gateway (POST /api/v1/predict)
│   ├── fase2_lstm_v2.py        # Core sequential LSTM training pipeline
│   ├── diagnostico_epoca2.py   # Temporal distribution shift diagnosis (KS tests)
│   ├── experimentos_etapa2.py  # 2x2x2 factorial ablation runner across 5 seeds
│   ├── analisis_etapa3_proyectos.py # Per-project Wilcoxon tests & Spearman correlations
│   ├── dataset/                # Deep-SE benchmark datasets
│   ├── artefactos_exportados/  # Trained Keras models, JSON vocabularies, and metrics
│   ├── requirements.txt        # Python backend dependencies
│   └── README.md               # API documentation & architecture details
│
├── frontend-estimator/         # Interactive decision support dashboard (SPA)
│   ├── src/                    # React 19 components (EffortEstimator.jsx)
│   ├── package.json            # Node.js dependencies
│   ├── vite.config.js          # Vite 8 & Tailwind CSS v4 configuration
│   └── README.md               # Frontend setup & planning poker workflow guide
│
├── LICENSE                     # MIT License
└── README.md                   # This master documentation
```

---

## 3. Quick Start: Three Ways to Use This Repository

### Option A: Fast Replication of Paper Results (< 10 seconds)
To verify all numerical tables and non-parametric hypothesis tests reported in the manuscript without re-training:
```bash
cd replication_package
pip install -r requirements.txt
python scripts/reproduce_tables.py
```
This script reproduces:
* **Table 1:** Corpus descriptive statistics across all 16 repositories ($N = 23,313$).
* **Table 2:** Global benchmark results (Random vs. Chronological splits, MAE, MdAE, RMSE, SA, $\text{MAE}_{\ge 13}$).
* **Table 3:** Factorial ablation matrix across 5 seeds (Panels A, B, and C).
* **Table 4:** Statistical hypothesis testing (Wilcoxon paired signed-rank, Holm-adjusted $p$-values, $\hat{A}_{12}$ effect sizes).
* **Table 5:** Cross-project performance breakdown, Holm-adjusted Wilcoxon tests, and Spearman rank correlations with $\Delta\text{MAE}$.

---

### Option B: Rerunning the Factorial Ablation Matrix (50 Models across 5 Seeds)
To re-train all 8 factorial treatments and 2 project-conditioned models across 5 independent random seeds (`[42, 101, 202, 303, 404]`):
```bash
cd replication_package
python scripts/run_factorial_ablation.py
```
*Note:* Full training takes approximately 45–60 minutes on modern multi-core CPUs. Outputs are written directly to `data/predictions/runs_factoriales.csv`.

---

### Option C: Launching the Dual-Indicator Decision Support Prototype
The repository includes an operational client--server decision support system designed to assist Scrum teams during Planning Poker:

#### 1. Start the FastAPI Backend:
```bash
cd effort-estimator
pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```
* The API will be active at `http://localhost:8000`.
* Interactive OpenAPI/Swagger documentation: `http://localhost:8000/docs`.

#### 2. Start the React Frontend:
```bash
cd ../frontend-estimator
npm install
npm run dev
```
* The dashboard will open at `http://localhost:5173`.
* It provides interactive user story estimation with side-by-side comparison between the neural network prediction and the historical project median baseline, with automatic divergence alerts ($> 2$ SP) for Planning Poker review.

---

## 4. Citation and Benchmark Attribution

If you utilize this code, replication package, or benchmark partitions in your research, please cite:

```bibtex
@article{lopez2026empirical,
  author    = {L{\'o}pez de Jes{\'u}s, C{\'e}sar and Olmos S{\'a}nchez, Karla Miroslava},
  title     = {Empirical Evaluation of Recurrent Neural Networks for Agile Effort Estimation},
  journal   = {Empirical Software Engineering},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.5281/zenodo.23788447}
}
```

The Deep-SE dataset utilized in this benchmark was collected by Choetkiertikul et al. and curated by Tawosi et al.:
* Choetkiertikul, M., et al. (2019). *A deep learning model for estimating story points.* IEEE TSE, 45(7), 637-656.
* Tawosi, V., et al. (2023). *On the use of pre-trained language models for effort estimation.* IEEE TSE, 49(5), 3241-3259.

---

## 5. License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

