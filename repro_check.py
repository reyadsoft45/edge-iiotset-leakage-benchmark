#!/usr/bin/env python3
"""
Reproducibility evidence + train/test overlap check (answers audit items 2 and 6).

Writes to --out:
  repro_manifest.json   dataset SHA-256, row counts, Python/library versions, OS, CPU
  splits/<task>_seed<k>_test_idx.csv   test-row indices (Protocol B rows) for every split used
  overlap.csv           per task/seed: test rows whose exact feature vector also appears in
                        the training split, and how many of those carry a different label

Usage:
  python repro_check.py --csv ML-EdgeIIoT-dataset.csv --out results_repro
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

import edge_ids_experiments as E


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def row_keys(X):
    return pd.util.hash_pandas_object(X, index=False).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_repro")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)

    df = E.load_raw(args.csv)
    XB, y_bin, y_mc, groups, dedup = E.protocol_b(df, print)
    C1 = ["tcp.checksum", "icmp.checksum", "udp.stream", "tcp.ack_raw", "mbtcp.trans_id"]
    C2 = C1 + ["tcp.seq", "tcp.ack", "icmp.seq_le"]
    drop = lambda srcs: XB.drop(columns=[f for s_ in srcs if s_ in groups for f in groups[s_] if f in XB.columns])
    feature_sets = {"B": XB, "C1": drop(C1), "C2": drop(C2)}

    # (a) exact-feature overlap between train and test, for every feature set used in the paper
    rows = []
    for fs_name, Xf in feature_sets.items():
        keys = row_keys(Xf)
        for task, y in (("binary", y_bin), ("multiclass", y_mc)):
            for seed in range(args.seeds):
                idx = np.arange(len(XB))
                tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
                if fs_name == "B":
                    pd.Series(np.sort(te), name="row").to_csv(
                        os.path.join(args.out, "splits", f"{task}_seed{seed}_test_idx.csv"), index=False)
                train_labels = {}
                for i in tr:
                    train_labels.setdefault(keys[i], set()).add(y[i])
                shared = [i for i in te if keys[i] in train_labels]
                conflict = [i for i in shared if y[i] not in train_labels[keys[i]]]
                rows.append({"features": fs_name, "task": task, "seed": seed, "n_test": len(te),
                             "test_rows_with_feature_vector_in_train": len(shared),
                             "pct_of_test": round(100 * len(shared) / len(te), 3),
                             "of_which_different_label": len(conflict)})
                print(rows[-1])
    ov = pd.DataFrame(rows)

    # (b) are the label-free type decisions the same when made on training rows only?
    d = df.drop(columns=[c for c in E.DROP_COLS if c in df.columns]).reset_index(drop=True)
    Xraw = d.drop(columns=[E.LABEL_BIN, E.LABEL_MC])
    full_types = dedup["col_types"]
    type_rows = []
    for seed in range(args.seeds):
        idx = np.arange(len(Xraw))
        tr, _ = train_test_split(idx, test_size=0.2, stratify=d[E.LABEL_MC].values, random_state=seed)
        t = E.decide_types(Xraw.iloc[tr])
        diff = [c for c in full_types if t[c] != full_types[c]]
        type_rows.append({"seed": seed, "columns_with_different_type": len(diff), "which": ";".join(diff)})
        print(type_rows[-1])
    pd.DataFrame(type_rows).to_csv(os.path.join(args.out, "type_decisions_train_vs_full.csv"), index=False)

    # (c) would train-only constant-column removal differ? (columns constant in a training split
    #     but not in the full data are the only way global constant-removal could matter)
    const_rows = []
    for task, y in (("binary", y_bin), ("multiclass", y_mc)):
        for seed in range(args.seeds):
            tr, _ = train_test_split(np.arange(len(XB)), test_size=0.2, stratify=y, random_state=seed)
            const_in_train = [c for c in XB.columns if XB[c].values[tr].min() == XB[c].values[tr].max()]
            const_rows.append({"task": task, "seed": seed,
                               "columns_constant_in_train_only": len(const_in_train),
                               "which": ";".join(const_in_train)})
            print(const_rows[-1])
    pd.DataFrame(const_rows).to_csv(os.path.join(args.out, "constant_columns_train_vs_full.csv"), index=False)
    ov.to_csv(os.path.join(args.out, "overlap.csv"), index=False)

    manifest = {
        "dataset_file": os.path.basename(args.csv),
        "dataset_sha256": sha256(args.csv),
        "rows_raw": int(len(df)), "rows_after_dedup": int(len(XB)),
        "label_conflict_rows": dedup["label_conflicts"],
        "n_features_protocol_B": int(XB.shape[1]),
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "os": platform.platform(), "cpu": cpu_name(),
        "split": "stratified 80/20, sklearn train_test_split, random_state = seed",
        "max_overlap_pct_of_test_by_features": ov.groupby("features").pct_of_test.max().to_dict(),
        "type_decisions_differ_train_vs_full": int(sum(r["columns_with_different_type"] for r in type_rows)),
        "columns_constant_in_some_train_split": int(sum(r["columns_constant_in_train_only"] for r in const_rows)),
    }
    with open(os.path.join(args.out, "repro_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
