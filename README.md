# Leakage-aware benchmark of lightweight intrusion detectors on Edge-IIoTset

Code for the paper *"Hidden Shortcuts Inflate IoT Intrusion Detection: A Leakage-Aware Edge Benchmark"* (ICCIT-2026).

## Data
The dataset is **not** included. Download `ML-EdgeIIoT-dataset.csv` (folder "Selected dataset for ML and DL") from
[Edge-IIoTset on IEEE DataPort](https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)
(Ferrag et al., IEEE Access 10, 40281–40306, 2022). Do not open or re-save it in Excel, because that merges "0" and "0.0".

## Setup
```
pip install -r requirements.txt
```

## Experiments
| Script | Paper section | Command |
|---|---|---|
| `edge_ids_experiments.py` | E1–E3: leakage audit, edge benchmark, SHAP feature budget (Sec. 4.1, 4.3) | `python edge_ids_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results_full` |
| `protocol_c_ablation.py` | E4: identifier ablation B / C1 / C2 (Sec. 4.2) | `python protocol_c_ablation.py --csv ML-EdgeIIoT-dataset.csv --out results_c` |
| `improve_experiments.py` | E5: published-style recipe, nested tuning, 6-class (Sec. 4.1) | `python improve_experiments.py --csv ML-EdgeIIoT-dataset.csv --out results_e5` |
| `group_split_experiment.py` | E6: connection-grouped evaluation (Sec. 4.4) | `python group_split_experiment.py --csv ML-EdgeIIoT-dataset.csv --out results_e6` |
| `reviewer_checks.py` | E7: single-change protocol A′; E8: feature-vector collisions (Sec. 4.1, 4.2) | `python reviewer_checks.py --csv ML-EdgeIIoT-dataset.csv --out results_review` |
| `repro_check.py` | Dataset hash, versions, split indices, overlap checks | `python repro_check.py --csv ML-EdgeIIoT-dataset.csv --out results_repro` |

Add `--quick` to any script for a short smoke test. All splits are seeded (`random_state` = seed), so the results are reproducible.

## Results
The `results/` folder contains the raw per-split CSV files, summary tables, logs and figures behind every number in the paper. `results_repro/` (dataset SHA-256, library versions, CPU and the test-row indices of every split) is produced by `repro_check.py`.

## Citation
If you use this code, please cite the paper (citation details will be added after publication).

## License
MIT. See `LICENSE`.
