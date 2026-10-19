#!/usr/bin/env python3
"""
Leakage-aware benchmark of lightweight intrusion detectors for IoT edge devices
on Edge-IIoTset (ML-EdgeIIoT-dataset.csv).

Experiments
  E1  Leakage audit: the as-distributed preprocessing recipe ("Protocol A")
      vs. a corrected protocol ("Protocol B", after Galal, arXiv:2608.15761).
  E2  Edge benchmark under Protocol B: macro-F1, accuracy, serialized model size
      and single-sample CPU latency for six lightweight classifiers,
      binary and multiclass, mean +/- std over repeated stratified splits.
  E3  Feature budget: features ranked by mean |SHAP| (XGBoost TreeSHAP,
      computed on the TRAINING split only), retrain on the top-k features.

Usage (Google Colab or local):
  pip install xgboost scikit-learn pandas matplotlib
  python edge_ids_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results
  python edge_ids_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results --quick   # 1 seed smoke test

All hyperparameters are fixed a priori (edge-sized configurations); nothing is
tuned on the test split.
"""
import argparse
import json
import math
import os
import pickle
import time
import warnings
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")  # single-core latency, like an edge CPU

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier, DMatrix

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Dataset constants (Edge-IIoTset Readme, preprocessing step list)
# ----------------------------------------------------------------------------
LABEL_BIN = "Attack_label"
LABEL_MC = "Attack_type"
DROP_COLS = [  # identifiers / payloads removed by the distributed recipe
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]
DUMMY_COLS = [  # one-hot encoded by the distributed recipe
    "http.request.method", "http.referer", "http.request.version",
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
]
ABSENT = {"", "0", "0.0", "0.00", "nan", "NaN", "None", "none", "null", "NULL", "-"}
LEAK_THRESHOLD = 0.999

# Print-friendly palette (validated categorical order) + markers as 2nd encoding
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#898781", "axes.linewidth": 0.6,
        "grid.color": "#e4e3df", "grid.linewidth": 0.5,
        "savefig.dpi": 600, "savefig.bbox": "tight", "figure.dpi": 100,
    })


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_raw(path):
    """Read every column as text so that '0' and '0.0' stay distinguishable."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False,
                     encoding="utf-8", encoding_errors="replace")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df[LABEL_BIN] = df[LABEL_BIN].astype(float).astype(int)
    return df


# ----------------------------------------------------------------------------
# Protocol A: the recipe as distributed
# ----------------------------------------------------------------------------
def protocol_a(df):
    d = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    d = d.replace({"": np.nan}).dropna(axis=0, how="any")
    d = d.drop_duplicates(keep="first")
    y_bin = d[LABEL_BIN].values
    y_mc = d[LABEL_MC].values
    X = d.drop(columns=[LABEL_BIN, LABEL_MC])
    groups = {}
    parts = []
    for c in X.columns:
        if c in DUMMY_COLS:
            dummies = pd.get_dummies(X[c].astype(str), prefix=c).astype(np.float32)
            groups[c] = list(dummies.columns)
            parts.append(dummies)
        else:
            col = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.float32)
            groups[c] = [c]
            parts.append(col.rename(c).to_frame())
    Xn = pd.concat(parts, axis=1)
    return Xn, y_bin, y_mc, groups


# ----------------------------------------------------------------------------
# Protocol B: corrected preprocessing
# ----------------------------------------------------------------------------
def _parse_num(v):
    if v in ABSENT:
        return 0.0
    try:
        if v.lower().startswith("0x"):
            return float(int(v, 16))
        return float(v)
    except (ValueError, AttributeError):
        return np.nan


def _entropy(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum((k / n) * math.log2(k / n) for k in Counter(s).values())


def _text_features(series, name):
    s = series.where(~series.isin(ABSENT), "")
    uniq = pd.Series(s.unique())
    feats = pd.DataFrame({
        "v": uniq,
        f"{name}__present": (uniq != "").astype(np.float32),
        f"{name}__len": uniq.str.len().astype(np.float32),
        f"{name}__digit_ratio": uniq.map(lambda x: sum(ch.isdigit() for ch in x) / len(x) if x else 0.0),
        f"{name}__alpha_ratio": uniq.map(lambda x: sum(ch.isalpha() for ch in x) / len(x) if x else 0.0),
        f"{name}__entropy": uniq.map(_entropy),
    }).set_index("v")
    return feats.loc[s.values].reset_index(drop=True).astype(np.float32)


def decide_types(X):
    """Label-free schema decision: a column is numeric if >= 99% of its values parse as numbers
    (the 7 recipe text columns are always treated as text). Returns {column: 'numeric'|'text'}."""
    types = {}
    for c in X.columns:
        if c in DUMMY_COLS:
            types[c] = "text"
            continue
        mapping = {v: _parse_num(v) for v in X[c].unique()}
        frac_num = 1.0 - X[c].map(mapping).isna().mean()
        types[c] = "numeric" if frac_num >= 0.99 else "text"
    return types


def protocol_b(df, log):
    d = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).reset_index(drop=True)
    y_bin = d[LABEL_BIN].values
    y_mc = d[LABEL_MC].values
    X = d.drop(columns=[LABEL_BIN, LABEL_MC])
    types = decide_types(X)
    parts, groups = [], {}
    for c in X.columns:
        if types[c] == "numeric":
            mapping = {v: _parse_num(v) for v in X[c].unique()}
            col = X[c].map(mapping).fillna(-1).astype(np.float32).rename(c).to_frame()
        else:  # text column -> structural features, never identity encoding
            col = _text_features(X[c], c)
        groups[c] = list(col.columns)
        parts.append(col)
    Xn = pd.concat(parts, axis=1)
    Xn = Xn.loc[:, Xn.nunique() > 1]
    groups = {k: [f for f in v if f in Xn.columns] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if v}

    # deduplicate on (features, label) BEFORE any split
    full = Xn.copy()
    full["_y"] = y_mc
    n0 = len(full)
    full = full.drop_duplicates()
    n1 = len(full)
    # rows whose features are identical but labels differ (irreducible noise)
    feat_dup = full.drop(columns="_y").duplicated(keep=False)
    conflicts = int(feat_dup.sum())
    log(f"[B] rows {n0} -> {n1} after dedup ({n0 - n1} duplicates removed); "
        f"{conflicts} rows share features with a different label (kept)")
    keep = full.index.values
    return (Xn.loc[keep].reset_index(drop=True), y_bin[keep], y_mc[keep], groups,
            {"rows_raw": n0, "rows_dedup": n1, "label_conflicts": conflicts,
             "keep_idx": keep, "col_types": types})


# ----------------------------------------------------------------------------
# Leakage audit: single-source-column separability
# ----------------------------------------------------------------------------
def leakage_audit(X, y, groups, seed=0, max_rows=60000):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    Xs, ys = X.iloc[idx], y[idx]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for src, cols in groups.items():
        clf = DecisionTreeClassifier(max_depth=3, random_state=seed)
        sc = cross_val_score(clf, Xs[cols].values, ys, cv=cv, scoring="balanced_accuracy")
        rows.append({"source_column": src, "n_features": len(cols),
                     "single_column_balanced_acc": float(sc.mean())})
    return pd.DataFrame(rows).sort_values("single_column_balanced_acc", ascending=False)


def train_only_audit_report(X, y_bin, groups, log, seed=0):
    """Label-based leakage audit computed on the TRAINING rows of seed 0 only.
    Diagnostic: it reports columns that alone separate the classes, but drops nothing, so
    the feature set is fixed by a predefined, label-independent policy."""
    idx = np.arange(len(X))
    tr, _ = train_test_split(idx, test_size=0.2, stratify=y_bin, random_state=seed)
    audit = leakage_audit(X.iloc[tr].reset_index(drop=True), y_bin[tr], groups, seed=seed)
    flagged = audit.loc[audit.single_column_balanced_acc >= LEAK_THRESHOLD, "source_column"].tolist()
    log(f"[audit train-only] max single-column balanced acc = "
        f"{audit.single_column_balanced_acc.max():.4f}; flagged (not dropped): {flagged}")
    return flagged


# ----------------------------------------------------------------------------
# Models (edge-sized, fixed a priori)
# ----------------------------------------------------------------------------
def make_models(n_classes, seed):
    xgb_kw = dict(n_estimators=100, max_depth=6, learning_rate=0.2, tree_method="hist",
                  n_jobs=1, random_state=seed, verbosity=0)
    if n_classes > 2:
        xgb_kw.update(objective="multi:softprob", num_class=n_classes)
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, n_jobs=1)),
        "GNB": GaussianNB(),
        "DT": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, random_state=seed),
        "RF": RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=1, random_state=seed),
        "XGB": XGBClassifier(**xgb_kw),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                             early_stopping=True, max_iter=200, random_state=seed)),
    }


def model_size_kb(model):
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0


def single_sample_latency_us(model, X_test, n=300, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    for i in idx[:20]:  # warm-up
        model.predict(X_test[i:i + 1])
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X_test[i:i + 1])
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e6)


def evaluate(model, Xtr, ytr, Xte, yte, seed, latency=True):
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    pred = model.predict(Xte)
    return {
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "weighted_f1": f1_score(yte, pred, average="weighted"),
        "balanced_acc": balanced_accuracy_score(yte, pred),
        "size_kb": model_size_kb(model),
        "latency_us": single_sample_latency_us(model, Xte, seed=seed) if latency else np.nan,
        "fit_s": fit_s,
    }


def split(X, y, seed):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


# ----------------------------------------------------------------------------
# SHAP ranking via XGBoost's built-in TreeSHAP (no extra dependency)
# ----------------------------------------------------------------------------
def shap_ranking(Xtr, ytr, n_classes, feature_names, seed, max_rows=5000):
    m = make_models(n_classes, seed)["XGB"]
    m.fit(Xtr, ytr)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Xtr), size=min(max_rows, len(Xtr)), replace=False)
    contrib = m.get_booster().predict(DMatrix(Xtr[idx]), pred_contribs=True)
    contrib = np.abs(np.asarray(contrib))
    if contrib.ndim == 3:  # (n, classes, features+1)
        imp = contrib[:, :, :-1].mean(axis=0).sum(axis=0)
    else:
        imp = contrib[:, :-1].mean(axis=0)
    order = np.argsort(-imp)
    return [(feature_names[i], float(imp[i])) for i in order]


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def run_e1(XA, yA_bin, XB, yB_bin, seeds, log):
    """Recipe vs corrected, binary task, macro-F1."""
    rows = []
    for seed in seeds:
        for proto, X, y in (("A (recipe)", XA, yA_bin), ("B (corrected)", XB, yB_bin)):
            Xtr, Xte, ytr, yte = split(X.values, y, seed)
            for name, m in make_models(2, seed).items():
                r = evaluate(m, Xtr, ytr, Xte, yte, seed, latency=False)
                rows.append({"protocol": proto, "model": name, "seed": seed, **r})
                log(f"[E1] {proto:13s} {name:4s} seed={seed} macroF1={r['macro_f1']:.4f}")
    return pd.DataFrame(rows)


def run_e2(XB, y, task, seeds, log):
    rows = []
    le = LabelEncoder().fit(y)
    yy = le.transform(y)
    k = len(le.classes_)
    for seed in seeds:
        Xtr, Xte, ytr, yte = split(XB.values, yy, seed)
        for name, m in make_models(k, seed).items():
            r = evaluate(m, Xtr, ytr, Xte, yte, seed)
            rows.append({"task": task, "model": name, "seed": seed, **r})
            log(f"[E2] {task:10s} {name:4s} seed={seed} macroF1={r['macro_f1']:.4f} "
                f"size={r['size_kb']:.1f}KB lat={r['latency_us']:.0f}us")
    return pd.DataFrame(rows)


def run_e3(XB, y, task, seeds, budgets, log):
    rows, rankings = [], {}
    le = LabelEncoder().fit(y)
    yy = le.transform(y)
    k = len(le.classes_)
    names = list(XB.columns)
    for seed in seeds:
        Xtr, Xte, ytr, yte = split(XB.values, yy, seed)
        rank = shap_ranking(Xtr, ytr, k, names, seed)  # training data only
        rankings[seed] = rank
        order = [names.index(f) for f, _ in rank]
        for b in budgets:
            kk = len(names) if b == "all" else min(int(b), len(names))
            cols = order[:kk]
            for name in ("DT", "RF", "XGB"):
                m = make_models(k, seed)[name]
                r = evaluate(m, Xtr[:, cols], ytr, Xte[:, cols], yte, seed)
                rows.append({"task": task, "model": name, "seed": seed, "k": kk, **r})
                log(f"[E3] {task:10s} k={kk:3d} {name:4s} seed={seed} macroF1={r['macro_f1']:.4f}")
    return pd.DataFrame(rows), rankings


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
MODEL_ORDER = ["LR", "GNB", "DT", "RF", "XGB", "MLP"]


def fig_e1(e1, out):
    g = e1.groupby(["protocol", "model"])["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    for j, (proto, hatch) in enumerate((("A (recipe)", ""), ("B (corrected)", "////"))):
        s = g[g.protocol == proto].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + (j - 0.5) * w, s["mean"], w * 0.92, yerr=s["std"], capsize=1.5,
               color=COLORS[j], hatch=hatch, edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 0.6}, label=f"Protocol {proto}")
    ax.set_xticks(x, MODEL_ORDER)
    ax.set_ylabel("Macro-F1 (binary)")
    lo = max(0.0, float(g["mean"].min()) - 0.1)
    ax.set_ylim(lo, 1.01)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.2, columnspacing=1.0)
    fig.savefig(os.path.join(out, "fig_e1_leakage.png"))
    plt.close(fig)


def fig_e2_pareto(e2, task, out):
    g = e2[e2.task == task].groupby("model")[["macro_f1", "size_kb", "latency_us"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, col, lab in ((axes[0], "size_kb", "Serialized model size (KB, log)"),
                         (axes[1], "latency_us", "Single-sample latency (µs, log)")):
        for i, m in enumerate(MODEL_ORDER):
            if m not in g.index:
                continue
            ax.scatter(g.loc[m, col], g.loc[m, "macro_f1"], s=36, marker=MARKERS[i],
                       color=COLORS[i], edgecolor="white", linewidth=0.8, zorder=3, label=m)
        ax.set_xscale("log")
        span = np.log10(g[col].max() / max(g[col].min(), 1e-9))
        subs = (1.0, 2.0, 5.0) if span < 2.5 else (1.0,)
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=subs))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(lab)
        ax.grid(True, which="major")
        ax.set_axisbelow(True)
    axes[0].set_ylabel(f"Macro-F1 ({task})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98),
               ncol=len(labels), handletextpad=0.3, columnspacing=1.2)
    fig.savefig(os.path.join(out, f"fig_e2_pareto_{task}.png"))
    plt.close(fig)


def fig_shap(rank, task, out, top=15):
    names = [f for f, _ in rank[:top]][::-1]
    vals = [v for _, v in rank[:top]][::-1]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh(names, vals, color=COLORS[0], height=0.7)
    ax.set_xlabel("Mean |SHAP value| (training split)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(os.path.join(out, f"fig_e3_shap_{task}.png"))
    plt.close(fig)


def fig_budget(e3, task, out):
    g = e3[e3.task == task].groupby(["model", "k"])["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, m in enumerate(("DT", "RF", "XGB")):
        s = g[g.model == m].sort_values("k")
        ci = MODEL_ORDER.index(m)
        ax.errorbar(s["k"], s["mean"], yerr=s["std"], marker=MARKERS[ci], ms=4,
                    color=COLORS[ci], lw=1.6, capsize=1.5, elinewidth=0.6, label=m)
    ax.set_xlabel("Number of SHAP-ranked features (k)")
    ax.set_ylabel(f"Macro-F1 ({task})")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(out, f"fig_e3_budget_{task}.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------
def summarize(df, keys, out_csv):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "balanced_acc", "size_kb", "latency_us", "fit_s"]
    metrics = [m for m in metrics if m in df.columns]
    s = df.groupby(keys)[metrics].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    s.to_csv(out_csv, index=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to ML-EdgeIIoT-dataset.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget_seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="1 seed, smoke test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_plot_style()
    logf = open(os.path.join(args.out, "log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    seeds = [0] if args.quick else list(range(args.seeds))
    bseeds = [0] if args.quick else list(range(args.budget_seeds))
    budgets = [3, 5, 8, 10, 15, 20, "all"]

    df = load_raw(args.csv)
    log(f"Loaded {len(df)} rows x {df.shape[1]} columns")
    log("Class counts:\n" + df[LABEL_MC].value_counts().to_string())

    XA, yA_bin, yA_mc, gA = protocol_a(df)
    log(f"[A] matrix {XA.shape}")
    XB, yB_bin, yB_mc, gB, dedup = protocol_b(df, log)
    log(f"[B] matrix {XB.shape}")

    auditA = leakage_audit(XA, yA_bin, gA)   # descriptive: shows the artifact in Protocol A
    auditB = leakage_audit(XB, yB_bin, gB)   # descriptive only
    auditA.to_csv(os.path.join(args.out, "audit_protocolA.csv"), index=False)
    log("[audit A] top columns:\n" + auditA.head(10).to_string(index=False))
    # No label-based column dropping: the Protocol B feature set is fixed by a predefined,
    # label-independent policy. The train-only audit is reported for transparency.
    leaky = train_only_audit_report(XB, yB_bin, gB, log)
    auditB.to_csv(os.path.join(args.out, "audit_protocolB.csv"), index=False)
    log("[audit B] top columns:\n" + auditB.head(10).to_string(index=False))
    log(f"[B] final feature matrix {XB.shape}")

    e1 = run_e1(XA, yA_bin, XB, yB_bin, seeds, log)
    e1.to_csv(os.path.join(args.out, "e1_raw.csv"), index=False)
    summarize(e1, ["protocol", "model"], os.path.join(args.out, "table_e1_leakage.csv"))
    fig_e1(e1, args.out)

    e2 = pd.concat([run_e2(XB, yB_bin, "binary", seeds, log),
                    run_e2(XB, yB_mc, "multiclass", seeds, log)])
    e2.to_csv(os.path.join(args.out, "e2_raw.csv"), index=False)
    summarize(e2, ["task", "model"], os.path.join(args.out, "table_e2_edge.csv"))
    for t in ("binary", "multiclass"):
        fig_e2_pareto(e2, t, args.out)

    e3_parts, ranks = [], {}
    for t, y in (("binary", yB_bin), ("multiclass", yB_mc)):
        e3t, r = run_e3(XB, y, t, bseeds, budgets, log)
        e3_parts.append(e3t)
        ranks[t] = {str(s): v for s, v in r.items()}
        fig_shap(r[bseeds[0]], t, args.out)
        fig_budget(e3t, t, args.out)
    e3 = pd.concat(e3_parts)
    e3.to_csv(os.path.join(args.out, "e3_raw.csv"), index=False)
    summarize(e3, ["task", "model", "k"], os.path.join(args.out, "table_e3_budget.csv"))

    meta = {"dedup": {k: v for k, v in dedup.items() if k != "keep_idx"}, "protocolA_shape": list(XA.shape), "protocolB_shape": list(XB.shape),
            "flagged_leaky_columns_not_removed": leaky, "seeds": seeds, "budget_seeds": bseeds,
            "shap_rankings": ranks, "features_B": list(XB.columns)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log("Done. Tables (CSV) and 600-dpi figures are in " + args.out)


if __name__ == "__main__":
    main()
