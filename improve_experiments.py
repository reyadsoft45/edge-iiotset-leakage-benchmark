#!/usr/bin/env python3
"""
Experiment E5: (1) reproduce the "published-style" numbers, (2) raise honest results legitimately.

Part 1  Same recipe as most published papers (Protocol A) on the 15-class and 6-class
        tasks -> shows our models reach the same high numbers when evaluated the same way.
Part 2  Honest improvement on corrected data (Protocol B and C1):
          - XGBoost tuned on a VALIDATION split carved from the training data
            (never on the test split), with early stopping
          - optional class-balanced sample weights (helps rare attacks: MITM, Fingerprinting)
          - the 6-class grouping defined by the dataset authors, as well as 2 and 15 classes
Reports accuracy AND macro-F1, so comparisons with papers that report only accuracy are fair.

Usage:
  python improve_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results_e5
  python improve_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results_e5_quick --quick
Needs edge_ids_experiments.py in the same folder.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import edge_ids_experiments as E

C1_DROP = ["tcp.checksum", "icmp.checksum", "udp.stream", "tcp.ack_raw", "mbtcp.trans_id"]

# 6-class grouping: the five attack categories named by the dataset authors (DoS/DDoS,
# Information gathering, MITM, Injection, Malware) + Normal. The dataset page does not
# state where Password and Uploading belong; placing them here is OUR assignment and must
# be described as such in the paper unless confirmed in Ferrag et al. (2022).
SIX = {
    "Normal": "Normal",
    "DDoS_UDP": "DoS/DDoS", "DDoS_ICMP": "DoS/DDoS", "DDoS_TCP": "DoS/DDoS", "DDoS_HTTP": "DoS/DDoS",
    "Port_Scanning": "Information gathering", "Vulnerability_scanner": "Information gathering",
    "Fingerprinting": "Information gathering",
    "MITM": "Man-in-the-middle",
    "XSS": "Injection", "SQL_injection": "Injection", "Uploading": "Injection",
    "Backdoor": "Malware", "Password": "Malware", "Ransomware": "Malware",
}

GRID = [{"max_depth": d, "balanced": b} for d in (6, 10, 14) for b in (False, True)]


def xgb(k, seed, depth, n_est=1000, early=True):
    kw = dict(n_estimators=n_est, max_depth=depth, learning_rate=0.1, tree_method="hist",
              subsample=0.9, colsample_bytree=0.9, n_jobs=-1, random_state=seed, verbosity=0)
    if early:
        kw["early_stopping_rounds"] = 30
    if k > 2:
        kw.update(objective="multi:softprob", num_class=k, eval_metric="mlogloss")
    else:
        kw.update(eval_metric="logloss")
    return XGBClassifier(**kw)


def scores(y, p):
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted")}


def tune(Xtr, ytr, k, seed, log, tag):
    """Pick depth / class weighting on a validation split of the TRAINING data only."""
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=seed)
    best = None
    for g in GRID:
        sw = compute_sample_weight("balanced", ya) if g["balanced"] else None
        m = xgb(k, seed, g["max_depth"])
        m.fit(Xa, ya, sample_weight=sw, eval_set=[(Xv, yv)], verbose=False)
        f = f1_score(yv, m.predict(Xv), average="macro")
        n = int(getattr(m, "best_iteration", m.n_estimators - 1)) + 1
        log(f"[E5 tune] {tag} depth={g['max_depth']} balanced={g['balanced']} "
            f"val macroF1={f:.4f} trees={n}")
        if best is None or f > best["val_macro_f1"]:
            best = {**g, "n_estimators": n, "val_macro_f1": f}
    return best


def run_tuned(X, y, task, proto, seeds, log):
    le = LabelEncoder().fit(y)
    yy = le.transform(y)
    k = len(le.classes_)
    tag = f"{proto:3s} {task:10s}"
    rows = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = E.split(X.values, yy, seed)
        # nested selection: tune on a validation split of THIS seed's training data only,
        # so no outer test row of any seed is ever used for hyperparameter selection
        best = tune(Xtr, ytr, k, seed, log, f"{tag} seed={seed}")
        log(f"[E5 best] {tag} seed={seed} {best}")
        sw = compute_sample_weight("balanced", ytr) if best["balanced"] else None
        m = xgb(k, seed, best["max_depth"], n_est=best["n_estimators"], early=False)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr, sample_weight=sw)
        fit_s = time.perf_counter() - t0
        p = m.predict(Xte)
        m.set_params(n_jobs=1)  # single-core latency, as in the main benchmark
        r = {"protocol": proto, "task": task, "model": "XGB-tuned", "seed": seed,
             **scores(yte, p), "size_kb": E.model_size_kb(m),
             "latency_us": E.single_sample_latency_us(m, Xte, seed=seed), "fit_s": fit_s,
             "max_depth": best["max_depth"], "balanced": best["balanced"],
             "n_estimators": best["n_estimators"]}
        rows.append(r)
        log(f"[E5] {tag} XGB-tuned seed={seed} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
            f"size={r['size_kb']:.0f}KB")
        # small, edge-sized baseline under the same split
        dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=seed,
                                    class_weight="balanced" if best["balanced"] else None)
        r2 = E.evaluate(dt, Xtr, ytr, Xte, yte, seed)
        rows.append({"protocol": proto, "task": task, "model": "DT", "seed": seed,
                     **{kk: r2[kk] for kk in ("accuracy", "macro_f1", "weighted_f1", "size_kb", "latency_us", "fit_s")}})
    return rows


def run_recipe(XA, y, task, seeds, log):
    """Protocol A (as distributed) with the same fixed models as the main benchmark."""
    le = LabelEncoder().fit(y)
    yy = le.transform(y)
    k = len(le.classes_)
    rows = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = E.split(XA.values, yy, seed)
        for name in ("DT", "XGB"):
            m = E.make_models(k, seed)[name]
            m.fit(Xtr, ytr)
            r = scores(yte, m.predict(Xte))
            rows.append({"protocol": "A (recipe)", "task": task, "model": name, "seed": seed, **r})
            log(f"[E5 recipe] {task:10s} {name:4s} seed={seed} acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_e5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_recipe", action="store_true",
                    help="skip Part 1 (Protocol A); its results do not depend on tuning")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log_e5.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    seeds = [0] if args.quick else list(range(args.seeds))
    df = E.load_raw(args.csv)
    XA, yA_bin, yA_mc, _ = E.protocol_a(df)
    XB, y_bin, y_mc, groups, _ = E.protocol_b(df, log)
    leaky = E.train_only_audit_report(XB, y_bin, groups, log)  # diagnostic only, never drops
    XC1 = XB.drop(columns=[f for s in C1_DROP if s in groups for f in groups[s] if f in XB.columns])
    y6 = np.array([SIX.get(v, v) for v in y_mc])
    yA6 = np.array([SIX.get(v, v) for v in yA_mc])
    unknown = sorted(set(y_mc) - set(SIX))
    if unknown:
        log(f"WARNING: classes not in the 6-class map (kept as-is): {unknown}")

    rows = []
    if not args.skip_recipe:
        rows += run_recipe(XA, yA_mc, "15-class", seeds, log)
        rows += run_recipe(XA, yA6, "6-class", seeds, log)
    for proto, X in (("B", XB), ("C1", XC1)):
        for task, y in (("binary", y_bin), ("6-class", y6), ("15-class", y_mc)):
            rows += run_tuned(X, y, task, proto, seeds, log)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "e5_raw.csv"), index=False)
    E.summarize(res, ["protocol", "task", "model"], os.path.join(args.out, "table_e5.csv"))
    s = res.groupby(["protocol", "task", "model"])[["accuracy", "macro_f1"]].mean().round(4)
    log("Summary (mean over seeds):\n" + s.to_string())
    with open(os.path.join(args.out, "meta_e5.json"), "w") as f:
        json.dump({"seeds": seeds, "grid": GRID, "six_class_map": SIX, "C1_drop": C1_DROP,
                   "flagged_leaky_columns_not_removed": leaky}, f, indent=1)
    log("Done. Results in " + args.out)


if __name__ == "__main__":
    main()
