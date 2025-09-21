import os, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

# -------- CLI --------
ap = argparse.ArgumentParser(description="Train service model (baseline or engineered).")
ap.add_argument("--target", default=os.getenv("TARGET_COL", "soc_est"),
                help="Target column (ground-truth SoC), e.g. soc_est")
ap.add_argument("--W", type=int, default=int(os.getenv("W", 30)), help="Window size")
ap.add_argument("--H", type=int, default=int(os.getenv("H", 1)), help="Horizon (steps ahead)")
ap.add_argument("--engineered", type=int, default=int(os.getenv("ENGINEERED", 0)),
                help="0: only [V,I,T], 1: engineered features")
ap.add_argument("--ma", type=int, default=int(os.getenv("MA_N", 5)),
                help="Rolling mean/std window (only if engineered=1)")
ap.add_argument("--est", default=os.getenv("EST", "hgb"),
                choices=["hgb", "ridge"], help="Estimator: hgb (HistGBR) or ridge")
ap.add_argument("--outdir", default=os.getenv("OUTDIR", "models"),
                help="Where to write model.joblib and meta.json")
args = ap.parse_args()

TARGET_COL = args.target
W, H = args.W, args.H
ENGINEERED = bool(args.engineered)
MA_N = args.ma
EST = args.est
OUTDIR = Path(args.outdir)

# -------- helpers --------
def load_df(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    if "cycle_type" in df.columns:
        df = df[df["cycle_type"] == "discharge"].copy()
    return df.reset_index(drop=True)

def pick_target_col(df: pd.DataFrame) -> str:
    if TARGET_COL in df.columns:
        return TARGET_COL
    low = {c.lower(): c for c in df.columns}
    if TARGET_COL.lower() in low:
        return low[TARGET_COL.lower()]
    raise KeyError(f"TARGET_COL='{TARGET_COL}' not found. Columns={list(df.columns)}")

def _rolling_mean_std_series(a: np.ndarray, n: int):
    a = a.astype(np.float32, copy=False)
    if n <= 1:
        return a.copy(), np.zeros_like(a, dtype=np.float32)
    c  = np.cumsum(a, dtype=np.float64)
    c2 = np.cumsum(a*a, dtype=np.float64)
    sums_n  = c[n-1:]  - np.concatenate(([0.], c[:-n]))
    sums2_n = c2[n-1:] - np.concatenate(([0.], c2[:-n]))
    means_n = sums_n / n
    vars_n  = np.maximum(sums2_n / n - means_n**2, 0.0)
    stds_n  = np.sqrt(vars_n)
    m = np.empty_like(a, dtype=np.float32)
    s = np.empty_like(a, dtype=np.float32)
    m[:n-1] = [a[:k+1].mean() for k in range(n-1)]
    s[:n-1] = [a[:k+1].std()  if k>0 else 0.0 for k in range(n-1)]
    m[n-1:] = means_n.astype(np.float32, copy=False)
    s[n-1:] = stds_n.astype(np.float32,  copy=False)
    return m, s

def features_list(engineered: bool):
    if not engineered:
        return ["voltage_v","current_a","temp_c"]
    return ["voltage_v","current_a","temp_c",
            "dV","V_ma","V_std","I_ma","I_std","T_ma","T_std"]

def make_window_Xy(df: pd.DataFrame, W: int, H: int, engineered: bool, ma_n: int):
    need = {"voltage_v","current_a","temp_c"}
    assert need.issubset(df.columns), f"Eksik kolon(lar): {need - set(df.columns)}"

    v = df["voltage_v"].to_numpy(np.float32, copy=False)
    i = df["current_a"].to_numpy(np.float32, copy=False)
    t = df["temp_c"].to_numpy(np.float32, copy=False)
    y_all = df[pick_target_col(df)].to_numpy(np.float32, copy=False)

    windows_count = len(v) - W + 1
    target_start  = W - 1 + H
    N = min(windows_count, len(v) - target_start)
    if N <= 0:
        return None, None

    V = swv(v, W)[:N]
    I = swv(i, W)[:N]
    T = swv(t, W)[:N]
    y = y_all[target_start:target_start+N]

    if not engineered:
        X = np.stack([V,I,T], axis=2).reshape(N, -1).astype(np.float32, copy=False)
        return X, y

    dV_full = np.zeros_like(v, dtype=np.float32)
    if len(v) > 1:
        dV_full[1:] = np.diff(v)
    dV = swv(dV_full, W)[:N]

    V_ma_series, V_std_series = _rolling_mean_std_series(v, ma_n)
    I_ma_series, I_std_series = _rolling_mean_std_series(i, ma_n)
    T_ma_series, T_std_series = _rolling_mean_std_series(t, ma_n)

    V_ma  = swv(V_ma_series,  W)[:N]
    V_std = swv(V_std_series, W)[:N]
    I_ma  = swv(I_ma_series,  W)[:N]
    I_std = swv(I_std_series, W)[:N]
    T_ma  = swv(T_ma_series,  W)[:N]
    T_std = swv(T_std_series, W)[:N]

    feats_3d = np.stack([V,I,T,dV,V_ma,V_std,I_ma,I_std,T_ma,T_std], axis=2)  # (N,W,10)
    X = feats_3d.reshape(N, -1).astype(np.float32, copy=False)
    return X, y

def make_estimator(est_name: str):
    if est_name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=1.0))  # Ridge random_state almaz
        ])
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", HistGradientBoostingRegressor(
            max_depth=None, max_bins=255, learning_rate=0.1,
            max_iter=200, early_stopping=True, validation_fraction=0.1,
            random_state=42
        ))
    ])

# -------- train --------
paths = sorted(glob.glob("data/processed/B*.parquet"))
X_all, y_all = [], []
for p in paths:
    df = load_df(Path(p))
    X,y = make_window_Xy(df, W, H, ENGINEERED, MA_N)
    if X is None:
        continue
    X_all.append(X); y_all.append(y)

X = np.vstack(X_all); y = np.concatenate(y_all)
print(f"[train] shapes: X={X.shape}, y={y.shape} | engineered={ENGINEERED} W={W} H={H} MA_N={MA_N} est={EST}")

pipe = make_estimator(EST)
pipe.fit(X, y)

# -------- save (JOBLIB!) --------
OUTDIR.mkdir(parents=True, exist_ok=True)

joblib.dump(pipe, OUTDIR / "model.joblib")  # <-- JOBLIB OLARAK KAYDET
meta = {
    "window": W,
    "horizon": H,
    "features": features_list(ENGINEERED),
    "engineered": ENGINEERED,
    "ma_n": MA_N,
    "estimator": EST
}
(OUTDIR/"meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[save] {OUTDIR/'model.joblib'}")
print(f"[save] {OUTDIR/'meta.json'}")
print("[done]")
