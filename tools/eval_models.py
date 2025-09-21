# tools/eval_models.py
import os, glob, argparse
from pathlib import Path
import numpy as np, pandas as pd
from math import sqrt

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------- CLI / ENV ----------
parser = argparse.ArgumentParser()
parser.add_argument("--target", default=os.getenv("TARGET_COL", ""), help="Hedef (gerçek SoC) sütun adı (örn: soc_est)")
args = parser.parse_args()
TARGET_COL = args.target.strip()

# ---------- Yardımcılar ----------
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "cycle_type" in df.columns:
        df = df[df["cycle_type"] == "discharge"].copy()
    return df.reset_index(drop=True)

def pick_target_col(df: pd.DataFrame) -> str:
    # 1) Kullanıcıdan gelen isim
    if TARGET_COL:
        if TARGET_COL in df.columns:
            return TARGET_COL
        low = {c.lower(): c for c in df.columns}
        if TARGET_COL.lower() in low:
            return low[TARGET_COL.lower()]
        raise KeyError(f"TARGET_COL='{TARGET_COL}' bulunamadı. Kolonlar: {list(df.columns)}")

    # 2) Otomatik adaylar (case-insensitive)
    candidates = ["soc_est", "soc", "soc_true", "target", "y", "soc_pct", "soc_ratio", "state_of_charge"]
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    raise KeyError(
        "Hedef sütun bulunamadı. Adaylar: soc_est / soc / soc_true / target / y / soc_pct / soc_ratio / state_of_charge\n"
        f"Mevcut kolonlar: {list(df.columns)}\n"
        "İpucu: 'python tools/eval_models.py --target soc_est' şeklinde hedefi elle geçebilirsin."
    )

# ---- Vektörize pencereleme parçaları ----
from numpy.lib.stride_tricks import sliding_window_view as swv

def _rolling_mean_std_series(a: np.ndarray, n: int):
    """
    Causal (geçmişe bakan) hareketli ort/standart sapma.
    İlk n-1 noktada kısa pencere kullanır.
    """
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
    # baştaki kısa pencereler
    m[:n-1] = [a[:k+1].mean() for k in range(n-1)]
    s[:n-1] = [a[:k+1].std()  if k>0 else 0.0 for k in range(n-1)]
    # geri kalanı sabit n
    m[n-1:] = means_n.astype(np.float32, copy=False)
    s[n-1:] = stds_n.astype(np.float32,  copy=False)
    return m, s

def make_window_Xy(df: pd.DataFrame, W: int, H: int, engineered: bool=False, ma_n: int=5):
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
        raise ValueError(f"Yetersiz uzunluk: len={len(v)}, W={W}, H={H}")

    V = swv(v, W)[:N]   # (N, W)
    I = swv(i, W)[:N]
    T = swv(t, W)[:N]
    y = y_all[target_start:target_start+N]  # (N,)

    if not engineered:
        X = np.stack([V, I, T], axis=2).reshape(N, -1).astype(np.float32, copy=False)
        return X, y

    # engineered özellikler
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

    feats_3d = np.stack([V, I, T, dV, V_ma, V_std, I_ma, I_std, T_ma, T_std], axis=2)
    X = feats_3d.reshape(N, -1).astype(np.float32, copy=False)
    return X, y

# ---------- Veri ----------
DATA = sorted(glob.glob("data/processed/B*.parquet"))
BATS = [Path(p).stem for p in DATA]
print("[data]", BATS)

# ---------- Deney kurulumları ----------
SETUPS = [
    {"name":"base_W30_H1", "W":30, "H":1, "engineered":False, "ma_n":5},
    {"name":"eng_W60_H5", "W":60, "H":5, "engineered":True,  "ma_n":5},
]

# ---------- Model + Hızlı eğitim ----------
def make_pipe():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", HistGradientBoostingRegressor(
            max_depth=None,
            max_bins=255,
            learning_rate=0.1,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])

def eval_split(X_tr, y_tr, X_te, y_te, pipe, max_train=80000):
    # büyük veri için rastgele küçült
    if len(X_tr) > max_train:
        idx = np.random.RandomState(42).choice(len(X_tr), size=max_train, replace=False)
        X_tr = X_tr[idx]; y_tr = y_tr[idx]
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    return mae, rmse, r2

# ---------- Değerlendirme ----------
rows = []

# 1) Holdout (tüm veriden karışık train/test)
for S in SETUPS:
    X_all, y_all = [], []
    for p in DATA:
        df = load_df(Path(p))
        X,y = make_window_Xy(df, S["W"], S["H"], S["engineered"], S["ma_n"])
        X_all.append(X); y_all.append(y)
    X = np.vstack(X_all); y = np.concatenate(y_all)
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    mae,rmse,r2 = eval_split(Xtr,ytr,Xte,yte, make_pipe())
    rows.append(["HOLDOUT", "ALL→ALL", S["name"], mae, rmse, r2])

# 2) Cross-battery (birini tamamen testte bırak)
for S in SETUPS:
    for leave in range(len(DATA)):
        X_tr,y_tr,X_te,y_te = [],[],[],[]
        for idx,p in enumerate(DATA):
            df = load_df(Path(p))
            X,y = make_window_Xy(df, S["W"], S["H"], S["engineered"], S["ma_n"])
            if idx==leave:
                X_te.append(X); y_te.append(y)
            else:
                X_tr.append(X); y_tr.append(y)
        Xtr,ytr = np.vstack(X_tr), np.concatenate(y_tr)
        Xte,yte = np.vstack(X_te), np.concatenate(y_te)
        mae,rmse,r2 = eval_split(Xtr,ytr,Xte,yte, make_pipe())
        rows.append(["XBAT", f"TRAIN != {BATS[leave]}", S["name"], mae, rmse, r2])

out = pd.DataFrame(rows, columns=["scenario","split","setup","MAE","RMSE","R2"]).sort_values(["scenario","split","setup"])
print("\n=== RESULTS ===")
print(out.to_string(index=False))

Path("models").mkdir(exist_ok=True, parents=True)
out.to_csv("models/eval_summary.csv", index=False)
print("\n[save] models/eval_summary.csv")
