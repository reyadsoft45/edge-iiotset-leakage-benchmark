#!/usr/bin/env python3
"""
Experiment E4: are the top SHAP features behavioural, or capture identifiers?

Under Protocol B the highest-ranked SHAP features include Wireshark's stream
index, packet checksums and raw TCP sequence/ack numbers. These fields describe
*which capture/session a packet came from*, not *how the traffic behaves*, so a
detector that relies on them may again be learning provenance.

This script compares three feature sets on the same deduplicated rows:
  B   : corrected protocol (as in edge_ids_experiments.py)
  C1  : B minus clear identifiers (checksums, stream index, raw ack, Modbus transaction id)
  C2  : C1 minus flow-position counters (relative TCP seq/ack, ICMP sequence number)
and reports macro-F1, size, latency, per-class F1 and the new SHAP ranking.

Usage:
  python protocol_c_ablation.py --csv ML-EdgeIIoT-dataset.csv --out results_c
  python protocol_c_ablation.py --csv ML-EdgeIIoT-dataset.csv --out results_c_quick --quick
Needs edge_ids_experiments.py in the same folder.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

import edge_ids_experiments as E

C1_DROP = ["tcp.checksum", "icmp.checksum", "udp.stream", "tcp.ack_raw", "mbtcp.trans_id"]
C2_DROP = C1_DROP + ["tcp.seq", "tcp.ack", "icmp.seq_le"]
MODELS = ["DT", "RF", "XGB"]


def drop_sources(X, groups, sources):
    cols = [f for s in sources if s in groups for f in groups[s]]
    return X.drop(columns=[c for c in cols if c in X.columns])


def run(X, y, task, proto, seeds, log):
    le = LabelEncoder().fit(y)
    yy = le.transform(y)
    k = len(le.classes_)
    rows, pcs = [], []
    for seed in seeds:
        Xtr, Xte, ytr, yte = E.split(X.values, yy, seed)
        for name in MODELS:
            m = E.make_models(k, seed)[name]
            r = E.evaluate(m, Xtr, ytr, Xte, yte, seed)
            rows.append({"protocol": proto, "task": task, "model": name, "seed": seed,
                         "n_features": X.shape[1], **r})
            log(f"[E4] {proto:3s} {task:10s} {name:4s} seed={seed} macroF1={r['macro_f1']:.4f} "
                f"size={r['size_kb']:.1f}KB lat={r['latency_us']:.0f}us")
            if task == "multiclass":
                pred = m.predict(Xte)
                f = f1_score(yte, pred, average=None, labels=np.arange(k))
                for ci, cname in enumerate(le.classes_):
                    pcs.append({"protocol": proto, "model": name, "seed": seed,
                                "class": cname, "f1": f[ci]})
    return rows, pcs


def fig_ablation(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    protos = ["B", "C1", "C2"]
    x = np.arange(len(protos))
    for ax, task in zip(axes, ("binary", "multiclass")):
        g = df[df.task == task].groupby(["model", "protocol"])["macro_f1"].agg(["mean", "std"])
        for m in MODELS:
            ci = E.MODEL_ORDER.index(m)
            s = g.loc[m].reindex(protos)
            ax.errorbar(x, s["mean"], yerr=s["std"], marker=E.MARKERS[ci], ms=4, lw=1.6,
                        color=E.COLORS[ci], capsize=1.5, elinewidth=0.6, label=m)
        ax.set_xticks(x, ["B (corrected)", "C1 (\u2212 identifiers)", "C2 (\u2212 counters)"])
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(task.capitalize())
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.savefig(os.path.join(out, "fig_e4_ablation.png"))
    plt.close(fig)


def fig_perclass(pc, out):
    g = pc[pc.model == "XGB"].groupby(["class", "protocol"])["f1"].mean().unstack()
    g = g.sort_values("B")
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y = np.arange(len(g))
    offs = {"B": 0.22, "C1": 0.0, "C2": -0.22}
    for j, p in enumerate(["B", "C1", "C2"]):
        if p in g.columns:
            ax.scatter(g[p], y + offs[p], marker=E.MARKERS[j], color=E.COLORS[j], s=16, zorder=3,
                       label=f"Protocol {p}")
    ax.set_yticks(y, g.index)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Per-class F1 (XGBoost, mean of seeds)")
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              handletextpad=0.2, columnspacing=0.8)
    fig.savefig(os.path.join(out, "fig_e4_perclass.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_c")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    E.set_plot_style()
    logf = open(os.path.join(args.out, "log_c.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    seeds = [0] if args.quick else list(range(args.seeds))
    df = E.load_raw(args.csv)
    XB, y_bin, y_mc, groups, dedup = E.protocol_b(df, log)
    leaky = E.train_only_audit_report(XB, y_bin, groups, log)  # diagnostic only, never drops
    sets = {"B": XB, "C1": drop_sources(XB, groups, C1_DROP), "C2": drop_sources(XB, groups, C2_DROP)}
    for p, X in sets.items():
        log(f"[E4] protocol {p}: {X.shape[1]} features")

    rows, pcs = [], []
    for p, X in sets.items():
        for task, y in (("binary", y_bin), ("multiclass", y_mc)):
            r, pc = run(X, y, task, p, seeds, log)
            rows += r
            pcs += pc
    res = pd.DataFrame(rows)
    pc = pd.DataFrame(pcs)
    res.to_csv(os.path.join(args.out, "e4_raw.csv"), index=False)
    pc.to_csv(os.path.join(args.out, "e4_perclass_raw.csv"), index=False)
    E.summarize(res, ["protocol", "task", "model"], os.path.join(args.out, "table_e4_ablation.csv"))
    pc.groupby(["class", "protocol", "model"])["f1"].agg(["mean", "std"]).reset_index() \
        .to_csv(os.path.join(args.out, "table_e4_perclass.csv"), index=False)

    # single-column audit and SHAP ranking under C2
    groups_c2 = {k: [f for f in v if f in sets["C2"].columns] for k, v in groups.items()}
    groups_c2 = {k: v for k, v in groups_c2.items() if v}
    audit_c2 = E.leakage_audit(sets["C2"], y_bin, groups_c2)
    audit_c2.to_csv(os.path.join(args.out, "audit_protocolC2.csv"), index=False)
    log("[audit C2] top columns:\n" + audit_c2.head(10).to_string(index=False))
    le = LabelEncoder().fit(y_mc)
    Xtr, _, ytr, _ = E.split(sets["C2"].values, le.transform(y_mc), 0)
    rank = E.shap_ranking(Xtr, ytr, len(le.classes_), list(sets["C2"].columns), 0)
    E.fig_shap(rank, "multiclass_C2", args.out)
    with open(os.path.join(args.out, "meta_c.json"), "w") as f:
        json.dump({"C1_drop": C1_DROP, "C2_drop": C2_DROP, "flagged_leaky_columns_not_removed": leaky,
                   "n_features": {p: X.shape[1] for p, X in sets.items()},
                   "shap_rank_multiclass_C2": rank, "seeds": seeds}, f, indent=1)

    fig_ablation(res, args.out)
    fig_perclass(pc, args.out)
    s = res.groupby(["task", "protocol", "model"])["macro_f1"].mean().unstack("protocol").round(4)
    log("Macro-F1 summary:\n" + s.to_string())
    log("Done. Results in " + args.out)


if __name__ == "__main__":
    main()
