#!/usr/bin/env python3
"""
Reviewer checks (E7, E8). Needs edge_ids_experiments.py in the same folder.

E7  Isolate the serialization artifact.
    A   : published-style recipe ("0" and "0.0" are different one-hot categories)
    A*  : identical to A except that the absence spellings in the 7 one-hot fields are
          collapsed to one value BEFORE one-hot encoding (one change only)
    B   : corrected protocol (structural encoding, dedup on processed features, ...)
    A -> A*  = effect of the placeholder spelling alone
    A* -> B  = effect of all other preprocessing differences

E8  Feature-vector collisions under B / C1 / C2 (row-level, 15-class).
    For every test record: is its feature vector present in the training split, and if so,
    is it ambiguous there (seen with >1 label)? A majority-label "lookup" shows how much a
    perfect memorizer could gain from collisions. XGBoost accuracy and macro-F1 are reported
    separately on test records that collide with training and on those that do not.

Usage:
  python reviewer_checks.py --csv ML-EdgeIIoT-dataset.csv --out results_review
  python reviewer_checks.py --csv ML-EdgeIIoT-dataset.csv --out results_review_quick --quick
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import edge_ids_experiments as E

C1_DROP = ["tcp.checksum", "icmp.checksum", "udp.stream", "tcp.ack_raw", "mbtcp.trans_id"]
C2_DROP = C1_DROP + ["tcp.seq", "tcp.ack", "icmp.seq_le"]


def protocol_a_star(df):
    """Protocol A with ONE change: absence spellings in the one-hot fields are unified."""
    d = df.copy()
    for c in E.DUMMY_COLS:
        if c in d.columns:
            d[c] = d[c].where(~d[c].isin(E.ABSENT - {""}), "0")
    return E.protocol_a(d)


def run_e7(df, seeds, log):
    XA, yAb, yAm, _ = E.protocol_a(df)
    XS, ySb, ySm, gS = protocol_a_star(df)
    XB, yBb, yBm, gB, _ = E.protocol_b(df, log)
    auditS = E.leakage_audit(XS, ySb, gS)
    log("[E7] A* single-column audit (top 6):\n" + auditS.head(6).to_string(index=False))
    rows = []
    for name, X, yb, ym in (("A", XA, yAb, yAm), ("A*", XS, ySb, ySm), ("B", XB, yBb, yBm)):
        log(f"[E7] {name}: {X.shape[0]} rows x {X.shape[1]} features")
        for task, y in (("binary", yb), ("15-class", ym)):
            yy = LabelEncoder().fit_transform(y)
            k = len(np.unique(yy))
            for seed in seeds:
                Xtr, Xte, ytr, yte = E.split(X.values, yy, seed)
                for mname in ("DT", "XGB"):
                    m = E.make_models(k, seed)[mname]
                    if mname == "XGB":
                        m.set_params(n_jobs=-1)
                    m.fit(Xtr, ytr)
                    p = m.predict(Xte)
                    r = {"protocol": name, "task": task, "model": mname, "seed": seed,
                         "n_rows": X.shape[0], "n_features": X.shape[1],
                         "accuracy": accuracy_score(yte, p), "macro_f1": f1_score(yte, p, average="macro")}
                    rows.append(r)
                    log(f"[E7] {name:2s} {task:8s} {mname:3s} seed={seed} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")
    return pd.DataFrame(rows), auditS


def run_e8(df, seeds, log):
    XB, y_bin, y_mc, groups, _ = E.protocol_b(df, log)
    drop = lambda srcs: XB.drop(columns=[f for s in srcs if s in groups for f in groups[s] if f in XB.columns])
    sets = {"B": XB, "C1": drop(C1_DROP), "C2": drop(C2_DROP)}
    le = LabelEncoder().fit(y_mc)
    yy = le.transform(y_mc)
    k = len(le.classes_)
    rows = []
    for name, X in sets.items():
        keys = pd.util.hash_pandas_object(X, index=False).values
        for seed in seeds:
            idx = np.arange(len(X))
            Xtr, Xte, ytr, yte = E.split(X.values, yy, seed)
            tr, te = E.split(idx, yy, seed)[0:2]
            table = {}
            for i in tr:
                table.setdefault(keys[i], Counter())[yy[i]] += 1
            shared = np.array([keys[i] in table for i in te])
            ambiguous = np.array([keys[i] in table and len(table[keys[i]]) > 1 for i in te])
            lookup_ok = np.array([keys[i] in table and table[keys[i]].most_common(1)[0][0] == yy[i] for i in te])
            m = E.make_models(k, seed)["XGB"]
            m.set_params(n_jobs=-1)
            m.fit(Xtr, ytr)
            p = m.predict(Xte)
            r = {"features": name, "seed": seed, "n_features": X.shape[1], "n_test": len(te),
                 "pct_test_shared": 100 * shared.mean(),
                 "pct_test_ambiguous": 100 * ambiguous.mean(),
                 "lookup_acc_on_shared": lookup_ok[shared].mean() if shared.any() else np.nan,
                 "xgb_acc_all": accuracy_score(yte, p), "xgb_macro_f1_all": f1_score(yte, p, average="macro"),
                 "xgb_acc_shared": accuracy_score(yte[shared], p[shared]) if shared.any() else np.nan,
                 "xgb_acc_unshared": accuracy_score(yte[~shared], p[~shared]) if (~shared).any() else np.nan,
                 "xgb_macro_f1_unshared": f1_score(yte[~shared], p[~shared], average="macro") if (~shared).any() else np.nan}
            rows.append(r)
            log("[E8] " + " ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_review")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log_review.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    seeds = [0] if args.quick else list(range(args.seeds))
    df = E.load_raw(args.csv)
    e7, auditS = run_e7(df, seeds, log)
    e7.to_csv(os.path.join(args.out, "e7_raw.csv"), index=False)
    auditS.to_csv(os.path.join(args.out, "audit_protocolAstar.csv"), index=False)
    s7 = e7.groupby(["task", "model", "protocol"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4)
    s7.to_csv(os.path.join(args.out, "table_e7_isolation.csv"))
    log("E7 summary:\n" + s7.to_string())
    e8 = run_e8(df, seeds[:3], log)
    e8.to_csv(os.path.join(args.out, "e8_raw.csv"), index=False)
    s8 = e8.groupby("features").mean(numeric_only=True).round(4)
    s8.to_csv(os.path.join(args.out, "table_e8_collisions.csv"))
    log("E8 summary:\n" + s8.to_string())
    log("Done. Results in " + args.out)


if __name__ == "__main__":
    main()
