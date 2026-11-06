# Requirement-to-Evidence Matrix (AFGML Problem 1)

| Requirement ID | Source Requirement | Config and Command | Artifact and Key Finding | Status / Limitation | Report & Slide Location |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C-GM** | Exact VE and clique-tree agreement on tractable instance | `python -m pytest tests/test_causal_pipeline.py -k test_inference_equivalence` | Max abs error: $1.11 \times 10^{-16} \le 10^{-6}$ across $G_1$--$G_4$ | Complete | Report Sec. 2 & App. A; Slide 5 & 12 |
| **C-STAT** | Paired uncertainty with valid sampling unit (1,000 bootstrap reps) | `python experiments/run_test_audit.py` | Paired effect: worst shift $-0.095 \; [-0.112, -0.078]$ ($p < 0.001$, Holm) | Complete | Report Sec. 4; Slide 8 |
| **C-SEED** | Algorithmic variance across at least 5 random seeds | `python experiments/run_multiseed_audit.py` | 5 seeds: AUROC $0.6997 \pm 0.0171$, Worst CF Shift $0.0759 \pm 0.0160$ | Complete | Report App. B; Slide 8 & 12 |
| **C-LEAK** | Zero test leakage in preprocessing and discretization | `python -m pytest tests/test_causal_pipeline.py -k test_zero_leakage` | Thresholds fit on train; train/val/test $= 18\text{k}/6\text{k}/6\text{k}$ | Complete | Report Sec. 2; Slide 4 |
| **C-GROUP** | Small-group audit; flag cells where $N < 30$ | `python -m pytest tests/test_causal_pipeline.py -k test_small_groups` | All 6 intersectional strata $N \ge 102$; Small-Cell Flag = `False` | Complete | Report App. B; Slide 4 & 12 |
| **P1-E1** | Frozen test split audit of baseline GBT vs Blind vs Minimax | `python experiments/run_test_audit.py` | Unconstrained worst shift $0.168$ vs Minimax $0.073$ at matched threshold | Complete | Report Sec. 4, Table 1; Slide 8 |
| **P1-E2** | Matched-utility Pareto trade-off curve across penalty $\lambda$ | `python experiments/run_matched_utility.py` | Pareto knee at $\lambda=0.60$ with AUROC loss $\le 1.9\%$ | Complete | Report Sec. 4, Fig. 1; Slide 6 |
| **P1-E3** | Cross-graph audit discrepancy across DAGs $G_1$--$G_4$ | `python experiments/run_graph_shift.py` | Minimax cuts cross-graph variance by 34.2\% vs single-graph penalty | Complete | Report Sec. 4, Fig. 2; Slide 7 |
| **P1-E4** | Semi-synthetic SCM stress test under minority demographic thinning | `python experiments/run_simulation.py` | Thinning to $5\%$ minority support inflates empirical estimation gap | Complete | Report Sec. 5, Fig. 3; Slide 9 |
| **P1-E5** | Document failure case under structural graph misspecification | `python experiments/run_simulation.py` | Unobserved stress $U_{\text{stress}}$ violates causal sufficiency, error $\uparrow 3.4\times$ | Complete | Report Sec. 4; Slide 9 |
| **P1-A1** | Required ablation: Minimax vs Single-Graph vs Average-Graph | `python experiments/run_graph_shift.py` | Single fails on $G_3$ ($0.120$); Average fails worst-case ($0.108$) | Complete | Report Sec. 4, Table 1; Slide 9 |
| **P1-R1** | Robustness check: age band cutoffs and prior smoothing strengths | `python -m pytest tests/test_causal_pipeline.py` | Dirichlet smoothing ($\alpha \in \{0.5, 1.0, 2.0\}$) preserves graph ranking | Complete | Report Sec. 3; Slide 9 |
