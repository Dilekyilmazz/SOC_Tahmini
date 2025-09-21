# src/preprocess.py
import argparse, os, numpy as np, pandas as pd
from scipy.io import loadmat

def _to_str(x):
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray) and x.size == 1:
        return _to_str(x.item())
    return str(x)

def _ravel_attr(obj, name, default=np.nan):
    if hasattr(obj, name):
        v = getattr(obj, name)
        try:
            a = np.ravel(v).astype(float)
            return a
        except Exception:
            pass
    return np.array([default], dtype=float)

def extract_cycles(mat_path, only='discharge'):
    md = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    base = os.path.splitext(os.path.basename(mat_path))[0]

    # Üst seviye struct'ı bul
    root = md.get(base, None)
    if root is None:
        # fallback: içinde 'cycle' alanı olan ilk struct
        for k, v in md.items():
            if not k.startswith("__") and hasattr(v, "cycle"):
                root = v
                break
    if root is None or not hasattr(root, "cycle"):
        raise RuntimeError(f"'cycle' bulunamadı: {mat_path}")

    cycles = np.atleast_1d(root.cycle)
    frames = []

    for i in range(cycles.size):
        c = cycles[i]
        ctype = _to_str(c.type)
        if only != 'all' and ctype != only:
            continue

        d = c.data
        # Temel ölçümler
        t   = _ravel_attr(d, 'Time')
        vm  = _ravel_attr(d, 'Voltage_measured')
        im  = _ravel_attr(d, 'Current_measured')
        tm  = _ravel_attr(d, 'Temperature_measured')
        cap = _ravel_attr(d, 'Capacity')  # bazı setlerde tek değer olur

        n = int(min(len(t), len(vm), len(im), len(tm)))
        t, vm, im, tm = t[:n], vm[:n], im[:n], tm[:n]

        # zaman farkı (saniye)
        dt = np.diff(t, prepend=t[0])
        if len(dt): dt[0] = 0.0

        # Coulomb sayımıyla SoC (sadece discharge için anlamlı)
        soc = np.full(n, np.nan, dtype=float)
        cap_ah = float(cap[-1]) if len(cap) and np.isfinite(cap[-1]) else np.nan
        if ctype == 'discharge':
            Id = np.where(im < 0.0, -im, 0.0)  # deşarj akımını (+) yap
            q_as = np.cumsum(Id * dt)         # As (Coulomb)
            cap_as = cap_ah * 3600.0 if (np.isfinite(cap_ah) and cap_ah > 0) else (q_as[-1] if q_as[-1] > 0 else np.nan)
            if np.isfinite(cap_as) and cap_as > 0:
                soc = 1.0 - (q_as / cap_as)
                soc = np.clip(soc, 0.0, 1.0)

        frames.append(pd.DataFrame({
            "file": base,
            "cycle_index": i + 1,
            "cycle_type": ctype,
            "t_s": t,
            "voltage_v": vm,
            "current_a": im,
            "temp_c": tm,
            "dt_s": dt,
            "soc_est": soc,
            "capacity_ah": cap_ah if np.isfinite(cap_ah) else np.nan
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="B0005.mat / B0006.mat / B0018.mat yolu")
    ap.add_argument("--out", required=True, help="Çıkış parquet yolu")
    ap.add_argument("--only", default="discharge", choices=["discharge","charge","all"])
    ap.add_argument("--csv", action="store_true", help="Parquet yanında CSV de kaydet")
    args = ap.parse_args()

    df = extract_cycles(args.mat, args.only)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    if args.csv:
        import os as _os
        df.to_csv(_os.path.splitext(args.out)[0] + ".csv", index=False, encoding="utf-8")

    print(df.head())
    print("saved:", args.out, "rows:", len(df))

if __name__ == "__main__":
    main()
