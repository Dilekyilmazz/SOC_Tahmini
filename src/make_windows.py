# src/make_windows.py
import argparse, os, json, numpy as np, pandas as pd

def make_windows_from_df(df, features, W=30, H=1):
    # sadece discharge ve SoC mevcut satırlar
    df = df[(df["cycle_type"] == "discharge") & (df["soc_est"].notna())].copy()
    df = df.sort_values(["file", "cycle_index", "t_s"])
    X_list, y_list = [], []

    for (f, c), g in df.groupby(["file", "cycle_index"], sort=False):
        g = g.reset_index(drop=True)
        M = len(g)
        last = M - (W + H) + 1
        if last <= 0:
            continue
        F = g[features].values.astype("float32")
        Y = g["soc_est"].values.astype("float32")
        for i in range(last):
            x = F[i:i+W]
            y = Y[i+W-1+H]
            if np.isnan(x).any() or np.isnan(y):
                continue
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        return np.empty((0, W, len(features)), dtype="float32"), np.empty((0,), dtype="float32")
    return np.stack(X_list), np.array(y_list)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b5", default="data/processed/B0005.parquet")
    ap.add_argument("--b6", default="data/processed/B0006.parquet")
    ap.add_argument("--b18", default="data/processed/B0018.parquet")
    ap.add_argument("--out", default="artifacts/windows.npz")
    ap.add_argument("--meta", default="artifacts/meta.json")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    features = ["voltage_v", "current_a", "temp_c"]
    W, H = args.window, args.horizon

    b5  = pd.read_parquet(args.b5)
    b6  = pd.read_parquet(args.b6)
    b18 = pd.read_parquet(args.b18)

    # train/val: B0005 (zaman sıralı split)
    X_tr_all, y_tr_all = make_windows_from_df(b5, features, W, H)
    n = len(X_tr_all)
    n_val = max(1, int(0.2 * n))
    X_tr, y_tr = X_tr_all[:-n_val], y_tr_all[:-n_val]
    X_val, y_val = X_tr_all[-n_val:], y_tr_all[-n_val:]

    # test: B0006 + B0018
    X6, y6   = make_windows_from_df(b6, features, W, H)
    X18, y18 = make_windows_from_df(b18, features, W, H)
    if len(X6) and len(X18):
        X_te = np.concatenate([X6, X18], axis=0)
        y_te = np.concatenate([y6, y18], axis=0)
    elif len(X6):
        X_te, y_te = X6, y6
    else:
        X_te, y_te = X18, y18

    np.savez_compressed(args.out,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,   y_val=y_val,
        X_test=X_te,   y_test=y_te)

    meta = {
        "features": features, "window": W, "horizon": H,
        "shapes": {
            "train": [len(X_tr), W, len(features)],
            "val":   [len(X_val), W, len(features)],
            "test":  [len(X_te), W, len(features)]
        }
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved:", args.out)
    print(meta)

if __name__ == "__main__":
    main()
