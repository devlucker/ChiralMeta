# ⚗️ ChiralMeta: Meta-Learning & Prototypical Network Workbench for Enantioselectivity Prediction in Asymmetric Catalysis (MAT-07)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9+-059669.svg)](https://www.rdkit.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ChiralMeta** is an end-to-end, publication-grade cheminformatics AI platform designed for medicinal and computational chemists. It solves the foundational challenge of asymmetric hydrogenation catalysis: **predicting stereochemical enantioselectivity ($\ge 80\%$ ee) from limited reaction data** using few-shot episodic Meta-Learning and deep metric learning (Prototypical Networks with Mahalanobis distance covariance estimation).

---

## 📑 Table of Contents
1. [Executive Summary & The Thalidomide Lesson](#-executive-summary--the-thalidomide-lesson)
2. [Hackathon Evaluation Rubric Mapping (100 Marks)](#-hackathon-evaluation-rubric-mapping)
3. [Repository Structure](#-repository-structure)
4. [Chemical Data Cleaning Pipeline (40 Marks)](#-chemical-data-cleaning-pipeline-40-marks)
5. [Molecular Featurization (1,544 Dimensions)](#-molecular-featurization-1544-dimensions)
6. [Machine Learning & Meta-Learning Architectures (40 Marks)](#-machine-learning--meta-learning-architectures-40-marks)
7. [Post-Cleaning Visual Analytics & EDA (10 Marks)](#-post-cleaning-visual-analytics--eda-10-marks)
8. [Quick Start & Single-Command Launcher](#-quick-start--single-command-launcher)
9. [REST API Documentation](#-rest-api-documentation)

---

## 💊 Executive Summary & The Thalidomide Lesson

In pharmacology, stereochemistry is a matter of life and death. The classic historical tragedy of **Thalidomide (1957–1961)** demonstrated that mirror-image enantiomers can possess radically divergent biological activities:
- **$(R)$-Thalidomide**: Safe sedative and anti-nausea therapeutic.
- **$(S)$-Thalidomide**: Potent teratogen causing phocomelia (severe limb deformities) in over 10,000 infants worldwide.

Asymmetric hydrogenation catalysis employs chiral transition-metal complexes (Co, Ir, Rh) to selectively hydrogenate prochiral olefins into enantiopure stereoisomers. However, reaction outcomes depend on subtle non-covalent interactions between substrate, chiral ligand scaffold, solvent, temperature, and pressure. Conventional screening requires months of empirical lab iteration.

**ChiralMeta** leverages **few-shot episodic meta-learning** to predict enantiomeric excess ($\ge 80\%$ ee vs $< 80\%$ ee) with calibrated uncertainty, enabling discovery even in low-data regimes.

---

## 🏆 Hackathon Evaluation Rubric Mapping

| Criterion | Marks | Implementation & File Path |
| :--- | :---: | :--- |
| **1. Repository Structure & README.md** | **10** | Modular tree, dynamic relative paths, no hardcoded `/homes/ss2971/`, CPU-safe model loaders |
| **2. Rigorous Data Cleaning & Auditability** | **40** | [`src/data_cleaning.py`](file:///src/data_cleaning.py), 8 documented stages, [`data/cleaned/cleaning_audit.json`](file:///data/cleaned/cleaning_audit.json) |
| **3. Post-Cleaning Graphical Representations** | **10** | Interactive Plotly.js charts in UI, high-contrast typography, non-overlapping labels |
| **4. Model Evaluation & Benchmarking Arena** | **40** | [`src/evaluator.py`](file:///src/evaluator.py), ProtoNet vs RF vs GP vs Graph proxy, data scarcity curves |

---

## 📂 Repository Structure

```text
chiral-meta-hackathon/
├── README.md                      # Comprehensive publication-grade documentation
├── requirements.txt               # Frozen dependencies (torch, rdkit, fastapi, uvicorn, etc.)
├── run.py                         # Single-command pipeline & web server launcher
├── data/
│   ├── raw/                       # Co_AHOreaction_data.xlsx (532 rows) & out-of-sample-test-set.xlsx (246 rows)
│   └── cleaned/                   # cleaned_reactions_combined.csv, feature_matrix.npz, cleaning_audit.json
├── saved_models/
│   └── fully_trained_proto.pt     # Pre-trained Prototypical Network checkpoint (1544 -> 1024 -> 512)
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py           # Complete 8-stage chemical data cleaning & audit trail
│   ├── featurizer.py              # RDKit 1,544-dim feature vector generator & 2D SVG renderer
│   ├── evaluator.py               # Episodic few-shot evaluation engine & benchmark metrics
│   └── models/
│       ├── __init__.py
│       ├── proto_net.py           # Refactored CPU-safe Prototypical Network
│       ├── gp_model.py            # ExactGPLayer (RBF / Matern / Linear) & fallback
│       └── graph_mpnn.py          # Message Passing Neural Network with Set2Set readout
├── backend/
│   └── main.py                    # FastAPI server exposing REST API & serving web UI
├── frontend/
│   ├── index.html                 # Bio-Chem Lab UI (Tailwind CSS, Plotly.js, Lucide Icons)
│   ├── styles.css                 # Custom high-contrast theme (dark/light modes)
│   └── app.js                     # Interactive client logic, live SVG drawing, Plotly controllers
└── legacy_repo/                   # Original reference scripts preserved for auditability
```

---

## 🧪 Chemical Data Cleaning Pipeline (40 Marks)

The pipeline processes **778 total reactions** across two datasets:
1. `Co_AHOreaction_data.xlsx`: 532 Cobalt reactions
2. `out-of-sample-test-set.xlsx`: 246 Iridium and Rhodium out-of-sample reactions

### The 8 Cleaning Stages:
1. **Missing Value Audit & Domain-Specific Imputation**:
   - `Additive SMILES`: Missing in 139/532 (26.1%) of Co and 246/246 (100%) of OOS. Imputed with `"None"` and encoded into binary indicators `add_1` and `add_2`.
   - `Product SMILES`: Excluded from model features to prevent target leakage; reconstructed dynamically for visualization.
   - `Conv.` & `Yield`: Parallel experimental outcomes, excluded from input features to prevent data leakage. Median-imputed grouped by `(Metal, Scaffold)` for EDA analysis only.
   - `time(h)`: 14 missing rows in Co dataset imputed via group median (24.0 h).
   - `ee`: Verified 0 missing values across all 778 reactions.
2. **Categorical Normalization (Typo Correction)**:
   - Fixed typographical inconsistency: `'P,N oxazoline'` (105 rows) and `'P,N-oxazoline'` (30 rows) merged into standardized `'P,N-oxazoline'` (135 rows), reducing fragmented categories.
3. **Unit Conversion Artifact Cleaning**:
   - Raw pressure artifacts from bar-to-atm conversions (e.g., `0.986923`, `49.3462`, `74.0192`, `98.6923`) rounded to 2 decimal places with `Pressure_bar_equivalent` tracked.
4. **SMILES Sanitization, RDKit Validation & Stereochemistry Extraction**:
   - Replaced coordinate bond tildes (`~` $\to$ `-`).
   - Parsed and canonicalized all SMILES via RDKit `MolFromSmiles`.
   - Extracted tetrahedral chirality (`Tet_CW`, `Tet_CCW`) and double-bond stereochemistry (`Bond_Cis`, `Bond_Trans`).
5. **DOI Normalization**:
   - Standardized literature DOIs to canonical `https://doi.org/...` URLs.
6. **Target Binarization & Task Clustering**:
   - Binarized target at $\ge 80.0\%$ ee: Class 1 (High Selectivity) vs Class 0 (Low Selectivity).
   - Constructed meta-learning task clusters (`C1`–`C8`) stratified by `(Metal, Scaffold type, Olefin Type)`.
7. **Metal Center One-Hot Encoding**:
   - `metal_1` (Co), `metal_2` (Ir), `metal_3` (Rh).
8. **1,544-Dimensional Feature Matrix Assembly**:
   - Assembled full matrix and exported `cleaning_audit.json` and `feature_matrix.npz`.

---

## 🧬 Molecular Featurization (1,544 Dimensions)

Each reaction is encoded into a fixed-length 1,544-dimensional representation:

$$\mathbf{x} = \Big[ \underbrace{\mathbf{f}_{\text{react}}}_{512} \,\Vert\, \underbrace{\mathbf{f}_{\text{lig}}}_{512} \,\Vert\, \underbrace{\mathbf{f}_{\text{solv}}}_{512} \,\Vert\, \underbrace{\mathbf{c}_{\text{cond}}}_{8} \Big] \in \mathbb{R}^{1544}$$

- **Reactant Olefin (0..511)**: 512-bit Morgan Circular Fingerprint ($\text{radius}=2$, chirality enabled).
- **Chiral Ligand (512..1023)**: 512-bit Morgan Circular Fingerprint ($\text{radius}=2$, chirality enabled).
- **Solvent (1024..1535)**: 512-bit Morgan Circular Fingerprint ($\text{radius}=2$, chirality enabled).
- **Reaction Conditions (1536..1543)**: 8 continuous & one-hot descriptors:
  $$\mathbf{c}_{\text{cond}} = \left[ P\,(\text{atm}),\, T\,(^\circ\text{C}),\, S/C,\, \text{metal}_1,\, \text{metal}_2,\, \text{metal}_3,\, \text{add}_1,\, \text{add}_2 \right]$$

---

## 🧠 Machine Learning & Meta-Learning Architectures (40 Marks)

### 1. Prototypical Network with Mahalanobis Distance Metric
A non-linear projection $f_\theta: \mathbb{R}^{1544} \to \mathbb{R}^{512}$ maps reactions into a metric space:
$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(\mathbf{x}_i, y_i) \in S_k} f_\theta(\mathbf{x}_i)$$
Mahalanobis distance to class prototypes with regularized covariance estimation:
$$d_M(\mathbf{z}, \mathbf{c}_k) = \sqrt{(\mathbf{z} - \mathbf{c}_k)^T \mathbf{\Sigma}_k^{-1} (\mathbf{z} - \mathbf{c}_k)}$$
$$p(y = k \mid \mathbf{x}) = \frac{\exp\left(-d_M(f_\theta(\mathbf{x}), \mathbf{c}_k)\right)}{\sum_{k'} \exp\left(-d_M(f_\theta(\mathbf{x}), \mathbf{c}_{k'})\right)}$$

### 2. Random Forest Baseline
- 200 estimators trained on standardized 1,544-dim features.
- Serves as the primary tabular non-deep baseline for feature importance ranking.

### 3. Gaussian Process Classifier (ExactGP)
- Kernel: Matérn-5/2 / RBF with automatic relevance determination (ARD).
- Provides exact Bayesian epistemic uncertainty quantification.

### 4. Graph MPNN + Set2Set Readout
- Node features (52 dimensions: elemental symbol, degree, hybridization, CIP stereochemistry).
- Edge features (8 dimensions: bond order, conjugation, cis/trans).
- 3 message-passing steps with GRU state updates and permutation-invariant Set2Set pooling.

---

## 📊 Post-Cleaning Visual Analytics (10 Marks)

The UI includes 6 interactive Plotly.js charts formatted according to scientific publication standards:
1. **Missing Data Imputation Audit**: Before vs after cleaning across all columns.
2. **ee% Distribution & Threshold**: Histogram with 80% ee boundary.
3. **Scaffold Performance**: Horizontal bar chart comparing ligand scaffolds.
4. **Metal Center Performance**: Violin plots comparing Co, Ir, and Rh enantioselectivity.
5. **Phase Space Scatter**: Temperature vs Pressure with bubble size scaled by S/C ratio and color mapped to ee%.
6. **Physicochemical Correlation Heatmap**: Pearson correlation matrix of reaction descriptors.

---

## 🚀 Quick Start & Single-Command Launcher

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/lakshit/chiral-meta-catalysis.git
cd chiral-meta-catalysis
pip install -r requirements.txt
```

### 2. Run the Single-Command Launcher
```bash
python run.py
```
This automatically:
- Executes the 8-stage chemical data cleaning pipeline.
- Builds `feature_matrix.npz` and `cleaning_audit.json`.
- Launches the FastAPI backend and web workbench at `http://localhost:8000`.

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/health` | `GET` | Health check and server status |
| `/api/cleaning-report` | `GET` | Retrieve the 8-stage cleaning audit log |
| `/api/render-molecules` | `POST` | Generate crisp 2D SVG structures and molecular descriptors |
| `/api/predict` | `POST` | Run Prototypical Network inference on a reaction |
| `/api/evaluate/benchmark` | `GET` | Run multi-model episodic few-shot benchmarking |
| `/api/evaluate/scarcity-curve` | `GET` | Compute ROC-AUC vs support size ($N \in \{8, \dots, 256\}$) |
| `/api/evaluate/embedding` | `GET` | Compute 2D PCA / t-SNE Prototypical embedding space |
| `/api/batch-predict` | `POST` | Upload CSV/Excel file for high-throughput reaction screening |
| `/api/download-cleaned-csv` | `GET` | Download the full cleaned 778-reaction dataset |
