"""
ChiralMeta FastAPI Backend — REST API + Static Frontend Server
"""

import sys
import json
import time
import logging
import pathlib
import io
import os
import warnings

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ChiralMeta.API")

ROOT     = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR  = ROOT / "src"
FE_DIR   = ROOT / "frontend"
DATA_DIR = ROOT / "data" / "cleaned"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_DIR))

# ── Lazy imports ─────────────────────────────────────────────────────────────
_cleaning_done = False
_df  = None
_X   = None
_y   = None
_ee  = None

def _ensure_data():
    global _cleaning_done, _df, _X, _y, _ee
    if not _cleaning_done:
        from src.data_cleaning import load_cleaned_data
        _df, _X, _y, _ee = load_cleaned_data()
        _cleaning_done = True
    return _df, _X, _y, _ee


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="ChiralMeta API",
    description="Meta-Learning Workbench for Enantioselectivity Prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Mount static frontend
if FE_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FE_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════════════
# Root — serve frontend
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    idx = FE_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text())
    return HTMLResponse("<h1>ChiralMeta — Frontend not found</h1>")


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": time.time()}


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLEANING REPORT  (40-Mark Tab)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/cleaning-report")
async def cleaning_report():
    from src.data_cleaning import load_audit
    try:
        audit = load_audit()
        return JSONResponse(content=audit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cleaning-report/trigger")
async def trigger_cleaning():
    """Re-run the cleaning pipeline from scratch."""
    from src.data_cleaning import run_cleaning_pipeline
    global _cleaning_done
    _cleaning_done = False
    try:
        audit = run_cleaning_pipeline()
        return JSONResponse(content={"status": "success", "summary": audit.get("row_counts", {})})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download-cleaned-csv")
async def download_cleaned_csv():
    csv_path = DATA_DIR / "cleaned_reactions_combined.csv"
    if not csv_path.exists():
        from src.data_cleaning import run_cleaning_pipeline
        run_cleaning_pipeline()
    return FileResponse(
        str(csv_path),
        media_type="text/csv",
        filename="chiralmeta_cleaned_reactions.csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
# DATASET BROWSER
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/reactions")
async def get_reactions(
    source: Optional[str] = None,
    metal: Optional[str] = None,
    scaffold: Optional[str] = None,
    ee_min: float = 0.0,
    ee_max: float = 100.0,
    page: int = 1,
    page_size: int = 50,
):
    df, X, y, ee = _ensure_data()
    q = df.copy()
    if source:
        q = q[q["source"].str.contains(source, case=False, na=False)]
    if metal:
        q = q[q["Metal"].str.upper() == metal.upper()] if "Metal" in q.columns else q
    if scaffold:
        q = q[q["Scaffold type"].str.contains(scaffold, case=False, na=False)] if "Scaffold type" in q.columns else q
    q = q[(q["ee"] >= ee_min) & (q["ee"] <= ee_max)] if "ee" in q.columns else q

    total = len(q)
    start = (page - 1) * page_size
    q = q.iloc[start : start + page_size]

    # Return safe serializable subset
    cols = [c for c in [
        "source", "Metal", "Scaffold type", "Olefin Type",
        "Reactant SMILES", "Ligand SMILES", "Solvent SMILES",
        "Pressure/atm", "Temperature/C", "S/C", "time(h)", "ee", "ee_class",
        "Additive SMILES", "cluster",
    ] if c in q.columns]

    records = q[cols].fillna("").to_dict(orient="records")
    return {"total": total, "page": page, "page_size": page_size, "reactions": records}


@app.get("/api/reactions/stats")
async def reactions_stats():
    df, X, y, ee = _ensure_data()
    stats = {
        "total_reactions": len(df),
        "co_reactions": int((df["source"] == "Co_train").sum()) if "source" in df.columns else 0,
        "oos_reactions": int((df["source"] == "OOS_test").sum()) if "source" in df.columns else 0,
        "high_selectivity": int((y == 1).sum()),
        "low_selectivity": int((y == 0).sum()),
        "ee_mean": round(float(ee.mean()), 2),
        "ee_median": round(float(np.median(ee)), 2),
        "ee_std": round(float(ee.std()), 2),
    }
    if "Scaffold type" in df.columns:
        stats["scaffold_counts"] = df["Scaffold type"].value_counts().to_dict()
    if "Metal" in df.columns:
        stats["metal_counts"] = df["Metal"].value_counts().to_dict()
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# EDA DATA  (10-Mark Tab Charts)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/eda/ee-distribution")
async def ee_distribution():
    df, X, y, ee = _ensure_data()
    return {
        "ee_values": ee.tolist(),
        "labels":    y.tolist(),
        "threshold": 80.0,
        "bins": 40,
    }


@app.get("/api/eda/missing-data")
async def missing_data():
    from src.data_cleaning import load_audit
    audit = load_audit()
    before = audit.get("missing_before", {})
    after  = audit.get("missing_after",  {})
    cols   = sorted(set(list(before.keys()) + list(after.keys())))
    return {
        "columns": cols,
        "before":  [before.get(c, 0) for c in cols],
        "after":   [after.get(c, 0)  for c in cols],
    }


@app.get("/api/eda/scaffold-performance")
async def scaffold_performance():
    df, X, y, ee = _ensure_data()
    if "Scaffold type" not in df.columns:
        return {"scaffolds": [], "mean_ee": [], "high_rate": []}
    grp = df.groupby("Scaffold type")["ee"].agg(["mean", "count"]).reset_index()
    high_rate = df.groupby("Scaffold type")["ee_class"].mean().reset_index()
    merged = grp.merge(high_rate, on="Scaffold type")
    return {
        "scaffolds": merged["Scaffold type"].tolist(),
        "mean_ee":   merged["mean"].round(2).tolist(),
        "count":     merged["count"].tolist(),
        "high_rate": (merged["ee_class"] * 100).round(1).tolist(),
    }


@app.get("/api/eda/metal-violin")
async def metal_violin():
    df, X, y, ee = _ensure_data()
    result = {}
    if "Metal" in df.columns:
        for metal in df["Metal"].unique():
            if pd.isna(metal):
                continue
            result[str(metal)] = df[df["Metal"] == metal]["ee"].dropna().tolist()
    return result


@app.get("/api/eda/condition-phase-space")
async def condition_phase_space():
    df, X, y, ee = _ensure_data()
    press_col = next((c for c in df.columns if "pressure" in c.lower()), None)
    temp_col  = next((c for c in df.columns if "temperature" in c.lower()), None)
    sc_col    = next((c for c in df.columns if "s/c" in c.lower()), None)

    def _col(c):
        return pd.to_numeric(df[c], errors="coerce").tolist() if c and c in df.columns else [0]*len(df)

    return {
        "temperature": _col(temp_col),
        "pressure":    _col(press_col),
        "sc_ratio":    _col(sc_col),
        "ee":          ee.tolist(),
        "ee_class":    y.tolist(),
        "metal":       df["Metal"].tolist() if "Metal" in df.columns else [],
    }


@app.get("/api/eda/correlation-heatmap")
async def correlation_heatmap():
    df, X, y, ee = _ensure_data()
    cols_wanted = [
        "Pressure/atm", "Temperature/C", "S/C", "time(h)",
        "MolWt_Reactant", "MolWt_Ligand", "ChiralCenters_Ligand", "ee"
    ]
    avail = [c for c in cols_wanted if c in df.columns]
    if len(avail) < 2:
        return {"columns": [], "matrix": []}
    sub = df[avail].apply(pd.to_numeric, errors="coerce").dropna()
    corr = sub.corr().round(3)
    return {
        "columns": avail,
        "matrix": corr.values.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MOLECULE RENDERING
# ═══════════════════════════════════════════════════════════════════════════

class MoleculeRequest(BaseModel):
    reactant_smiles: str = ""
    ligand_smiles: str   = ""
    solvent_smiles: str  = ""
    additive_smiles: str = "None"
    width: int  = 420
    height: int = 280


@app.post("/api/render-molecules")
async def render_molecules(req: MoleculeRequest):
    from src.featurizer import render_mol_svg, compute_mol_descriptors
    return {
        "reactant": {
            "svg": render_mol_svg(req.reactant_smiles, req.width, req.height),
            "descriptors": compute_mol_descriptors(req.reactant_smiles),
        },
        "ligand": {
            "svg": render_mol_svg(req.ligand_smiles, req.width, req.height),
            "descriptors": compute_mol_descriptors(req.ligand_smiles),
        },
        "solvent": {
            "svg": render_mol_svg(req.solvent_smiles, 240, 200),
            "descriptors": compute_mol_descriptors(req.solvent_smiles),
        },
        "additive": {
            "svg": render_mol_svg(req.additive_smiles, 240, 200),
            "descriptors": compute_mol_descriptors(req.additive_smiles),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    reactant_smiles: str
    ligand_smiles: str
    solvent_smiles: str
    pressure: float     = 1.0
    temperature: float  = 25.0
    sc_ratio: float     = 100.0
    metal: str          = "Co"
    additive_smiles: str = "None"
    n_support: int      = 64
    distance_metric: str = "mahalanobis"


@app.post("/api/predict")
async def predict(req: PredictionRequest):
    from src.featurizer import compute_feature_vector
    from src.evaluator  import predict_single

    fv = compute_feature_vector(
        reactant_smiles=req.reactant_smiles,
        ligand_smiles=req.ligand_smiles,
        solvent_smiles=req.solvent_smiles,
        pressure=req.pressure,
        temperature=req.temperature,
        sc_ratio=req.sc_ratio,
        metal=req.metal,
        additive_smiles=req.additive_smiles,
    )

    df, X, y, ee = _ensure_data()
    # Use n_support reactions from dataset as context
    n_sup = min(req.n_support, len(X))
    idx   = np.random.choice(len(X), n_sup, replace=False)
    sup_X = X[idx]
    sup_y = y[idx]

    result = predict_single(fv, support_X=sup_X, support_y=sup_y)

    # Find top-3 similar reactions from dataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    fv_sc = scaler.transform(fv.reshape(1, -1))
    sims  = cosine_similarity(fv_sc, X_sc)[0]
    top3_idx = np.argsort(sims)[::-1][:3]

    similar = []
    for idx_r in top3_idx:
        row = df.iloc[idx_r]
        similar.append({
            "rank":          len(similar) + 1,
            "similarity_pct": round(float(sims[idx_r]) * 100, 1),
            "ee_experimental": float(row["ee"]) if "ee" in row else 0.0,
            "ee_class":       "HIGH" if int(row.get("ee_class", 0)) == 1 else "LOW",
            "metal":          str(row.get("Metal", "?")),
            "scaffold":       str(row.get("Scaffold type", "?")),
            "source":         str(row.get("source", "?")),
        })

    result["similar_reactions"] = similar
    result["feature_dim"] = int(fv.shape[0])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MODEL EVALUATION  (40-Mark Tab)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/evaluate/benchmark")
async def evaluate_benchmark(
    n_support: int   = 64,
    n_trials: int    = 5,
    distance: str    = "mahalanobis",
):
    df, X, y, ee = _ensure_data()
    clusters = df["cluster"].values if "cluster" in df.columns else None

    from src.evaluator import evaluate_protonet, evaluate_random_forest, evaluate_gp_baseline, evaluate_graph_proxy

    proto = evaluate_protonet(X, y, n_support=n_support, n_trials=n_trials, clusters=clusters, distance_metric=distance)
    rf    = evaluate_random_forest(X, y, n_support=n_support, n_trials=n_trials, clusters=clusters)
    gp    = evaluate_gp_baseline(X, y, n_support=n_support, n_trials=min(n_trials, 3), clusters=clusters)
    graph = evaluate_graph_proxy(X, y, n_support=n_support, n_trials=min(n_trials, 3), clusters=clusters)

    return {
        "models": [proto, rf, gp, graph],
        "n_support": n_support,
        "n_trials":  n_trials,
    }


@app.get("/api/evaluate/scarcity-curve")
async def scarcity_curve_endpoint(n_trials: int = 3):
    df, X, y, ee = _ensure_data()
    clusters = df["cluster"].values if "cluster" in df.columns else None
    from src.evaluator import scarcity_curve
    result = scarcity_curve(X, y, n_trials=n_trials, clusters=clusters)
    return result


@app.get("/api/evaluate/embedding")
async def embedding_endpoint(n_support: int = 64, method: str = "pca"):
    df, X, y, ee = _ensure_data()
    from src.evaluator import compute_embedding
    result = compute_embedding(X, y, n_support=n_support, method=method)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD & SCREENING
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Accept CSV or XLSX; return predictions for all rows."""
    content = await file.read()
    try:
        if file.filename.endswith(".xlsx"):
            df_up = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            df_up = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    from src.featurizer import compute_feature_vector
    from src.evaluator  import predict_single
    from src.data_cleaning import load_cleaned_data

    df_clean, X_sup, y_sup, _ = load_cleaned_data()
    # Use a fixed support of 128
    n_sup = min(128, len(X_sup))
    idx   = np.arange(n_sup)
    sup_X = X_sup[idx]
    sup_y = y_sup[idx]

    results = []
    react_col = next((c for c in df_up.columns if "reactant" in c.lower()), None)
    lig_col   = next((c for c in df_up.columns if "ligand"   in c.lower()), None)
    solv_col  = next((c for c in df_up.columns if "solvent"  in c.lower()), None)
    press_col = next((c for c in df_up.columns if "pressure" in c.lower()), None)
    temp_col  = next((c for c in df_up.columns if "temperature" in c.lower()), None)
    sc_col    = next((c for c in df_up.columns if "s/c"  in c.lower() or "sc" == c.lower()), None)
    metal_col = next((c for c in df_up.columns if c.lower() == "metal"), None)
    add_col   = next((c for c in df_up.columns if "additive" in c.lower()), None)

    for i, row in df_up.iterrows():
        try:
            fv = compute_feature_vector(
                reactant_smiles  = str(row[react_col])  if react_col else "",
                ligand_smiles    = str(row[lig_col])    if lig_col   else "",
                solvent_smiles   = str(row[solv_col])   if solv_col  else "",
                pressure         = float(row[press_col]) if press_col else 1.0,
                temperature      = float(row[temp_col])  if temp_col  else 25.0,
                sc_ratio         = float(row[sc_col])    if sc_col    else 100.0,
                metal            = str(row[metal_col])   if metal_col else "Co",
                additive_smiles  = str(row[add_col])     if add_col   else "None",
            )
            pred = predict_single(fv, support_X=sup_X, support_y=sup_y)
            row_result = {"row": i + 1}
            row_result.update(pred)
            if "ee" in df_up.columns:
                row_result["ee_actual"] = float(row["ee"])
            results.append(row_result)
        except Exception as ex:
            results.append({"row": i + 1, "error": str(ex)})

    # Summary stats
    high_count = sum(1 for r in results if r.get("ee_class") == 1)
    ee_preds   = [r["ee_predicted"] for r in results if "ee_predicted" in r]
    summary = {
        "total_rows": len(results),
        "high_selectivity_count":  high_count,
        "high_selectivity_rate":   round(high_count / max(len(results), 1) * 100, 1),
        "mean_predicted_ee":       round(float(np.mean(ee_preds)), 2) if ee_preds else 0.0,
    }
    if "ee" in df_up.columns:
        actual_ee = pd.to_numeric(df_up["ee"], errors="coerce").dropna().values
        pred_class = [r.get("ee_class", 0) for r in results if "ee_class" in r]
        actual_class = (actual_ee >= 80.0).astype(int)
        if len(pred_class) == len(actual_class):
            from sklearn.metrics import accuracy_score
            summary["accuracy_vs_ground_truth"] = round(
                float(accuracy_score(actual_class, pred_class)) * 100, 1
            )

    return {"summary": summary, "predictions": results}


@app.get("/api/download-sample-batch-csv")
async def download_sample_batch_csv():
    """Return a curated 15-reaction sample CSV ready for batch screening ingestion."""
    df, _, _, _ = _ensure_data()
    sample_cols = [
        "Reactant SMILES", "Ligand SMILES", "Solvent SMILES",
        "Pressure/atm", "Temperature/C", "S/C", "Metal", "Additive SMILES", "ee"
    ]
    avail_cols = [c for c in sample_cols if c in df.columns]
    sample_df = df[avail_cols].head(15)
    csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chiralmeta_sample_batch.csv"}
    )



# ═══════════════════════════════════════════════════════════════════════════
# PRESET REACTIONS — 16 curated benchmarks
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = [
    {
        "id": 1, "name": "Co High — NNN-Pincer di-sub (99% ee)",
        "metal": "Co", "scaffold": "NNN-pincer", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1cccc(C(F)(F)F)c1)c1ccccc1Cl",
        "ligand_smiles":   "C/C(=N\\c1c(C(C)C)cccc1C(C)C)c1cccc(C2=N[C@@H](C(C)(C)C)CO2)n1",
        "solvent_smiles":  "Cc1ccccc1", "solvent_name": "Toluene",
        "pressure": 1.0, "temperature": 25.0, "sc_ratio": 20.0,
        "additive_smiles": "None", "ee_actual": 99.0, "ee_class": 1,
    },
    {
        "id": 2, "name": "Co Low — NNN-Pincer di-sub (5% ee)",
        "metal": "Co", "scaffold": "NNN-pincer", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1cc(OC)c(OC)c(OC)c1)c1ccc2ccccc2c1",
        "ligand_smiles":   "C/C(=N\\c1c(C(C)C)cccc1C(C)C)c1cccc(/C(C)=N/[C@@H](C)C2CCCCC2)n1",
        "solvent_smiles":  "Cc1ccccc1", "solvent_name": "Toluene",
        "pressure": 1.0, "temperature": 25.0, "sc_ratio": 20.0,
        "additive_smiles": "None", "ee_actual": 5.0, "ee_class": 0,
    },
    {
        "id": 3, "name": "Co High — Bisphosphine di-sub (99% ee)",
        "metal": "Co", "scaffold": "bisphosphine", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(C#Cc1ccccc1)NC(C)=O",
        "ligand_smiles":   "CP(c1nc2ccccc2nc1P(C)C(C)(C)C)C(C)(C)C",
        "solvent_smiles":  "CC#N", "solvent_name": "Acetonitrile",
        "pressure": 40.0, "temperature": 50.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 99.0, "ee_class": 1,
    },
    {
        "id": 4, "name": "Co Low — Bisphosphine tri-sub (0% ee)",
        "metal": "Co", "scaffold": "bisphosphine", "olefin_type": "tri-sub",
        "reactant_smiles": "CC(=O)N/C(C)=C/c1ccc(C)cc1",
        "ligand_smiles":   "CP(c1ccccc1P(C)C(C)(C)C)C(C)(C)C",
        "solvent_smiles":  "OCC(F)(F)F", "solvent_name": "TFE",
        "pressure": 40.0, "temperature": 70.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 0.0, "ee_class": 0,
    },
    {
        "id": 5, "name": "Co High — PPO Oxazoline di-sub (99% ee)",
        "metal": "Co", "scaffold": "phosphine-pyridine-oxazoline(PPO)", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1cc(C)c(C)cc1C)[SiH](c1ccccc1)c1ccccc1",
        "ligand_smiles":   "CC(C)(C)[C@H]1COC(c2cccc(CP(C(C)(C)C)C(C)(C)C)n2)=N1",
        "solvent_smiles":  "CCCCC", "solvent_name": "Pentane",
        "pressure": 2.0, "temperature": 25.0, "sc_ratio": 20.0,
        "additive_smiles": "None", "ee_actual": 99.0, "ee_class": 1,
    },
    {
        "id": 6, "name": "Co Moderate — Ferrocene Bisphosphine tri-sub (77% ee)",
        "metal": "Co", "scaffold": "ferrocene bisphosphine", "olefin_type": "tri-sub",
        "reactant_smiles": "CC(=O)N/C(C)=C/c1ccc(C)cc1",
        "ligand_smiles":   "C[C@@H](C1=C(P(c2ccccc2)c2ccccc2)C([Fe]C2C=CC=C2)C=C1)P(C1CCCCC1)C1CCCCC1",
        "solvent_smiles":  "OCC(F)(F)F", "solvent_name": "TFE",
        "pressure": 40.0, "temperature": 70.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 77.0, "ee_class": 0,
    },
    {
        "id": 7, "name": "Ir High — P,N-Oxazoline di-sub (94% ee)",
        "metal": "Ir", "scaffold": "P,N-oxazoline", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1ccc(C(F)(F)F)cc1)C(C)(C)C",
        "ligand_smiles":   "CC(C)[C@H](NP(C)C(C)(C)C)C1=N[C@H](C(C)(C)C)CO1",
        "solvent_smiles":  "ClCCl", "solvent_name": "DCM",
        "pressure": 1.0, "temperature": 23.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 94.0, "ee_class": 1,
    },
    {
        "id": 8, "name": "Ir Low — P,N-Oxazoline di-sub (15% ee)",
        "metal": "Ir", "scaffold": "P,N-oxazoline", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1ccccc1)C(C)(C)C",
        "ligand_smiles":   "CC(C)[C@H]1COC([C@@H](NP(C)C(C)(C)C)C(C)C)=N1",
        "solvent_smiles":  "ClCCl", "solvent_name": "DCM",
        "pressure": 1.0, "temperature": 25.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 15.0, "ee_class": 0,
    },
    {
        "id": 9, "name": "Ir High — P,N-Oxazoline tetra-sub (99% ee)",
        "metal": "Ir", "scaffold": "P,N-oxazoline", "olefin_type": "tetra-sub",
        "reactant_smiles": "CC(=O)/C(C)=C(\\C)c1ccccc1",
        "ligand_smiles":   "Cc1ccccc1P(NC1=N[C@@H](C(C)(C)C)CO1)c1ccccc1C",
        "solvent_smiles":  "ClCCl", "solvent_name": "DCM",
        "pressure": 100.0, "temperature": 23.0, "sc_ratio": 50.0,
        "additive_smiles": "None", "ee_actual": 99.0, "ee_class": 1,
    },
    {
        "id": 10, "name": "Ir Low — P,N-Oxazoline tetra-sub (25% ee)",
        "metal": "Ir", "scaffold": "P,N-oxazoline", "olefin_type": "tetra-sub",
        "reactant_smiles": "CC(C)=C(C)c1ccccc1",
        "ligand_smiles":   "CC(C)[C@H](NP(C)C(C)(C)C)C1=N[C@H](C(C)C)CO1",
        "solvent_smiles":  "ClCCl", "solvent_name": "DCM",
        "pressure": 74.02, "temperature": 25.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 25.0, "ee_class": 0,
    },
    {
        "id": 11, "name": "Rh High — Ferrocene Bisphosphine tri-sub (99.9% ee)",
        "metal": "Rh", "scaffold": "ferrocene bisphosphine", "olefin_type": "tri-sub",
        "reactant_smiles": "O=c1cc(C(F)(F)F)oc2ccccc12",
        "ligand_smiles":   "C1=CC([Fe]C2C=CC=C2P2Cc3cccc4c3C3(CC4)CCc4cccc(c43)C2)C(P2Cc3cccc4c3C3(CC4)CCc4cccc(c43)C2)=C1",
        "solvent_smiles":  "CO", "solvent_name": "Methanol",
        "pressure": 10.0, "temperature": 25.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 99.9, "ee_class": 1,
    },
    {
        "id": 12, "name": "Rh Low — Phosphoramidite di-sub (0% ee)",
        "metal": "Rh", "scaffold": "phosphoramidite", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(CC(=O)c1ccccc1)S(=O)(=O)c1ccccc1",
        "ligand_smiles":   "CN(C)P1Cc2ccc3ccccc3c2-c2c(ccc3ccccc23)C1",
        "solvent_smiles":  "C1CCOC1", "solvent_name": "THF",
        "pressure": 80.0, "temperature": 25.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 0.0, "ee_class": 0,
    },
    {
        "id": 13, "name": "Co — P,N Bisphosphite tetra-sub (High)",
        "metal": "Co", "scaffold": "P,N", "olefin_type": "tetra-sub",
        "reactant_smiles": "CC(=C(C)c1ccccc1)C(C)(C)C",
        "ligand_smiles":   "CC(C)(C)OP(Oc1ccccc1-c1ccccc1N)OC(C)(C)C",
        "solvent_smiles":  "Cc1ccccc1", "solvent_name": "Toluene",
        "pressure": 50.0, "temperature": 40.0, "sc_ratio": 50.0,
        "additive_smiles": "None", "ee_actual": 88.0, "ee_class": 1,
    },
    {
        "id": 14, "name": "Ir — Phosphoramidite di-sub (High)",
        "metal": "Ir", "scaffold": "phosphoramidite", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(c1ccccc1)c1ccccc1",
        "ligand_smiles":   "CN(C)P1Cc2ccc3ccccc3c2-c2c(ccc3ccccc23)C1",
        "solvent_smiles":  "ClCCl", "solvent_name": "DCM",
        "pressure": 50.0, "temperature": 25.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 92.0, "ee_class": 1,
    },
    {
        "id": 15, "name": "Co — Ferrocene Bisphosphine di-sub (High)",
        "metal": "Co", "scaffold": "ferrocene bisphosphine", "olefin_type": "di-sub",
        "reactant_smiles": "C=C(NC(=O)c1ccccc1)c1ccccc1",
        "ligand_smiles":   "C[C@@H](C1=C(P(c2ccccc2)c2ccccc2)C([Fe]C2C=CC=C2)C=C1)P(c2ccccc2)c2ccccc2",
        "solvent_smiles":  "OCC(F)(F)F", "solvent_name": "TFE",
        "pressure": 40.0, "temperature": 50.0, "sc_ratio": 100.0,
        "additive_smiles": "None", "ee_actual": 95.0, "ee_class": 1,
    },
    {
        "id": 16, "name": "Rh — NNN-Pincer tri-sub (Low)",
        "metal": "Rh", "scaffold": "NNN-pincer", "olefin_type": "tri-sub",
        "reactant_smiles": "C(/C=C/c1ccc(C)cc1)(=N)NC(=O)OC(C)(C)C",
        "ligand_smiles":   "C/C(=N/c1ccccc1)c1cccc(/C(C)=N/c2ccccc2)n1",
        "solvent_smiles":  "Cc1ccccc1", "solvent_name": "Toluene",
        "pressure": 1.0, "temperature": 25.0, "sc_ratio": 20.0,
        "additive_smiles": "None", "ee_actual": 12.0, "ee_class": 0,
    },
]


@app.get("/api/presets")
async def get_presets():
    return {"presets": PRESETS, "count": len(PRESETS)}


@app.get("/api/presets/{preset_id}")
async def get_preset(preset_id: int):
    for p in PRESETS:
        if p["id"] == preset_id:
            return p
    raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    log.info("ChiralMeta API starting …")
    try:
        _ensure_data()
        log.info("  Dataset loaded ✓")
    except Exception as e:
        log.warning(f"  Dataset pre-load skipped: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
