# src/train_baseline.py
import os, json, argparse, numpy as np, joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def flatten(X):  # (N, W, F) -> (N, W*F)
    if X.ndim != 3: return X
    N, W, F = X.shape
    return X.reshape(N, W*F)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/windows.npz")
    ap.add_argument("--meta", default="artifacts/meta.json")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    data = np.load(args.inp)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xv,  yv  = data["X_val"],   data["y_val"]
    Xte, yte = data["X_test"],  data["y_test"]

    Xtr_f, Xv_f, Xte_f = map(flatten, [Xtr, Xv, Xte])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_f)
    Xv_s  = scaler.transform(Xv_f)
    Xte_s = scaler.transform(Xte_f)

    model = Ridge(alpha=args.alpha)
    model.fit(Xtr_s, ytr)

    def report(name, Xs, y):
        if len(y) == 0:
            print(f"{name}: boş split")
            return None
        yp = model.predict(Xs)
        mae = mean_absolute_error(y, yp)
        mse_val = mean_squared_error(y, yp)   # MSE
        rmse = float(np.sqrt(mse_val))        # RMSE = sqrt(MSE)
        print(f"{name:>5} -> MAE: {mae:.4f}  RMSE: {rmse:.4f}")
        return mae, rmse

    print("Shapes:", Xtr_s.shape, Xv_s.shape, Xte_s.shape)
    report("train", Xtr_s, ytr)
    report(" val ",  Xv_s,  yv)
    report("test ",  Xte_s, yte)

    # kaydet
    joblib.dump(model, os.path.join(args.models_dir, "soc_model.joblib"))
    joblib.dump(scaler, os.path.join(args.models_dir, "scaler.joblib"))

    # meta -> modele koyalım (API için lazım olacak)
    if os.path.exists(args.meta):
        with open(args.meta, "r", encoding="utf-8") as f: meta = json.load(f)
    else:
        meta = {}
    with open(os.path.join(args.models_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("saved models to:", args.models_dir)

if __name__ == "__main__":
    main()
