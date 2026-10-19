#!/usr/bin/env python3
"""
Experiment E6: does performance hold when packets of the same connection cannot straddle
train and test? (connection-grouped evaluation; see flow_ids for exactly how groups are built)

Row-level random splits (E1-E5) can put packets of the same network connection in both
training and test data. Here every packet is assigned to a bidirectional FLOW
(source IP:port <-> destination IP:port, from the raw columns that the models never see),
and whole flows are kept on one side of the split with StratifiedGroupKFold.
The same 5-fold protocol with ordinary StratifiedKFold (row-level) is run alongside, so the
two can be compared like for like.

Packets with no IP endpoints (e.g. ARP) cannot be assigned to a flow and are kept as
singleton groups; their share is reported.

Usage:
  python group_split_experiment.py --csv ML-EdgeIIoT-dataset.csv --out results_e6
  python group_split_experiment.py --csv ML-EdgeIIoT-dataset.csv --out results_e6_quick --quick
Needs edge_ids_experiments.py in the same folder.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import edge_ids_experiments as E

warnings.filterwarnings("ignore")
C1_DROP = ["tcp.checksum", "icmp.checksum", "udp.stream", "tcp.ack_raw", "mbtcp.trans_id"]
MODELS = ["DT", "RF", "XGB"]


def _col(df, name):
    return df[name].astype(str).str.strip() if name in df.columns else pd.Series([""] * len(df))


def flow_ids(df):
    """Connection-grouping key per packet, built only from fields the dataset provides
    (it has no udp.srcport/udp.dstport, ip.proto or tcp.stream):
      TCP  : endpoint pair  (ip:tcp.srcport <-> ip:tcp.dstport), bidirectional
      UDP  : endpoint pair + udp.port + udp.stream (Wireshark's UDP conversation index)
      other IP (e.g. ICMP): endpoint pair only -> coarser groups, i.e. a STRICTER split
      ARP  : ARP sender/target protocol addresses
      none of the above: singleton group (cannot be linked; share is reported)
    Coarse grouping can only make the split stricter; under-grouping is limited to singletons."""
    src, dst = _col(df, "ip.src_host"), _col(df, "ip.dst_host")
    sp, dp = _col(df, "tcp.srcport"), _col(df, "tcp.dstport")
    up, us = _col(df, "udp.port"), _col(df, "udp.stream")
    asrc, adst = _col(df, "arp.src.proto_ipv4"), _col(df, "arp.dst.proto_ipv4")
    has_ip = ~(src.isin(E.ABSENT).values | dst.isin(E.ABSENT).values)
    is_tcp = ~sp.isin(E.ABSENT).values
    is_udp = ~is_tcp & ~up.isin(E.ABSENT).values
    is_arp = ~has_ip & ~(asrc.isin(E.ABSENT).values | adst.isin(E.ABSENT).values)

    def pair(a, b):
        a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
        return np.where(a < b, a + "<>" + b, b + "<>" + a)

    tcp_key = "tcp|" + pair(src.values + ":" + sp.values, dst.values + ":" + dp.values)
    udp_key = "udp|" + pair(src.values, dst.values) + "|p" + up.values + "|s" + us.values
    oth_key = "ip|" + pair(src.values, dst.values)
    arp_key = "arp|" + pair(asrc.values, adst.values)
    key = np.where(has_ip & is_tcp, tcp_key,
          np.where(has_ip & is_udp, udp_key,
          np.where(has_ip, oth_key,
          np.where(is_arp, arp_key, None))))
    single = key == None  # noqa: E711
    key[single] = ["single_" + str(i) for i in np.where(single)[0]]
    kind = np.where(has_ip & is_tcp, "tcp", np.where(has_ip & is_udp, "udp",
           np.where(has_ip, "other-ip", np.where(is_arp, "arp", "singleton"))))
    return key.astype(str), kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_e6")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 folds, smoke test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log_e6.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    folds = 2 if args.quick else args.folds
    df = E.load_raw(args.csv)
    XB, y_bin, y_mc, groups, info = E.protocol_b(df, log)
    keep = info["keep_idx"]
    flows_all, kind_all = flow_ids(df)
    flows, kind = flows_all[keep], kind_all[keep]
    no_ip = kind == "singleton"
    XC1 = XB.drop(columns=[f for s in C1_DROP if s in groups for f in groups[s] if f in XB.columns])

    # describe the grouping
    g = pd.DataFrame({"flow": flows, "cls": y_mc})
    per_cls = g.groupby("cls")["flow"].nunique().sort_values()
    log(f"[E6] {len(g)} packets in {g.flow.nunique()} connection groups; "
        f"{no_ip.mean() * 100:.2f}% packets could not be linked (singleton groups)")
    log("[E6] packets by grouping rule:\n" + pd.Series(kind).value_counts().to_string())
    gs = g.flow.value_counts()
    sizes = gs.describe(percentiles=[0.5, 0.9, 0.99]).round(1)
    log("[E6] group size distribution (packets per group):\n" + sizes.to_string())
    gs.describe(percentiles=[0.5, 0.9, 0.99]).to_csv(os.path.join(args.out, "group_size_distribution.csv"))
    pd.crosstab(g.cls, pd.Series(kind, name="rule")).to_csv(os.path.join(args.out, "class_by_grouping_rule.csv"))
    log("[E6] flows per class:\n" + per_cls.to_string())
    biggest = g.flow.value_counts().iloc[0] / len(g) * 100
    log(f"[E6] largest single flow holds {biggest:.2f}% of all packets")
    per_cls.to_csv(os.path.join(args.out, "flows_per_class.csv"))

    rows, coverage, fold_counts = [], [], []
    for task, y in (("binary", y_bin), ("multiclass", y_mc)):
        yy = LabelEncoder().fit_transform(y)
        k = len(np.unique(yy))
        splitters = {
            "row-level": StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(XB, yy),
            "flow-grouped": StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0).split(XB, yy, flows),
        }
        for split_name, it in splitters.items():
            for fold, (tr, te) in enumerate(it):
                shared_flows = len(set(flows[tr]) & set(flows[te]))
                missing = sorted(set(range(k)) - set(np.unique(yy[te])))
                missing_tr = sorted(set(range(k)) - set(np.unique(yy[tr])))
                coverage.append({"task": task, "split": split_name, "fold": fold,
                                 "n_train": len(tr), "n_test": len(te), "flows_in_both": shared_flows,
                                 "classes_missing_from_test": len(missing),
                                 "classes_missing_from_train": len(missing_tr)})
                for c in range(k):
                    fold_counts.append({"task": task, "split": split_name, "fold": fold, "class": c,
                                        "n_train": int((yy[tr] == c).sum()), "n_test": int((yy[te] == c).sum())})
                for proto, X in (("B", XB), ("C1", XC1)):
                    for name in MODELS:
                        m = E.make_models(k, fold)[name]
                        if name in ("RF", "XGB"):
                            m.set_params(n_jobs=-1)
                        m.fit(X.values[tr], yy[tr])
                        p = m.predict(X.values[te])
                        r = {"task": task, "split": split_name, "protocol": proto, "model": name,
                             "fold": fold, "accuracy": accuracy_score(yy[te], p),
                             # macro-F1 over classes present in this fold (sklearn default)
                             "macro_f1": f1_score(yy[te], p, average="macro"),
                             # macro-F1 over ALL classes; absent classes count as 0 (stricter)
                             "macro_f1_all_labels": f1_score(yy[te], p, average="macro",
                                                             labels=np.arange(k), zero_division=0),
                             "classes_missing_from_test": len(missing),
                             "classes_missing_from_train": len(missing_tr)}
                        rows.append(r)
                        log(f"[E6] {task:10s} {split_name:12s} {proto:2s} {name:3s} fold={fold} "
                            f"acc={r['accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
                            f"flows_in_both={shared_flows} missing_classes={len(missing)}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "e6_raw.csv"), index=False)
    pd.DataFrame(coverage).to_csv(os.path.join(args.out, "e6_fold_coverage.csv"), index=False)
    lab_names = {t: LabelEncoder().fit(y).classes_ for t, y in (("binary", y_bin), ("multiclass", y_mc))}
    fc = pd.DataFrame(fold_counts)
    fc["class_name"] = [lab_names[t][c] for t, c in zip(fc.task, fc["class"])]
    fc.to_csv(os.path.join(args.out, "e6_fold_class_counts.csv"), index=False)
    s = res.groupby(["task", "protocol", "model", "split"])[["accuracy", "macro_f1", "macro_f1_all_labels"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.reset_index().to_csv(os.path.join(args.out, "table_e6.csv"), index=False)
    piv = res.groupby(["task", "protocol", "model", "split"])["macro_f1"].mean().unstack("split").round(4)
    piv["drop"] = (piv["row-level"] - piv["flow-grouped"]).round(4)
    log("Macro-F1, row-level vs flow-grouped (mean over folds):\n" + piv.to_string())
    with open(os.path.join(args.out, "meta_e6.json"), "w") as f:
        json.dump({"folds": folds, "n_groups": int(g.flow.nunique()),
                   "packets_by_rule": pd.Series(kind).value_counts().to_dict(),
                   "pct_no_ip": float(no_ip.mean() * 100), "largest_flow_pct": float(biggest),
                   "C1_drop": C1_DROP}, f, indent=1)
    log("Done. Results in " + args.out)


if __name__ == "__main__":
    main()
