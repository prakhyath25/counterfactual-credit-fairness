# Two-Student Accountability and Contributions

## 1. Concrete Work Item Matrix

| Work Item | Lead Student | Reviewer / Cross-Check | Artifact / Link | What Was Verified |
| :--- | :--- | :--- | :--- | :--- |
| **Causal DAG Formulation ($G_1$--$G_4$)** | Prakhyath Shetty | Shri Hari S | `src/causal_dags.py` | Structural validity, BDeu smoothing, no cycles |
| **Exact Inference Proof (VE vs. CT)** | Prakhyath Shetty | Shri Hari S | `tests/test_causal_pipeline.py` | Numerical equivalence ($\vert{}\text{VE} - \text{CT}\vert{} \le 1.11 \times 10^{-16}$) |
| **Pearl Abduction-Action Engine** | Prakhyath Shetty | Shri Hari S | `src/counterfactual.py` | Correct 3-step abduction on discrete CPTs |
| **PyTorch Minimax Classifier** | Shri Hari S | Prakhyath Shetty | `src/models.py` | Correct forward pass and worst-case graph penalty |
| **Data Pipeline & Leakage Controls** | Shri Hari S | Prakhyath Shetty | `src/data_loader.py` | Zero test leakage; frozen 60/20/20 splits |
| **Mandatory Experiments (1--5)** | Shri Hari S | Prakhyath Shetty | `experiments/` | Metric correctness, Pareto frontier, SCM simulator |
| **Bootstrap & Multi-Seed Protocol** | Shri Hari S | Prakhyath Shetty | `src/metrics.py` | 1,000 paired resamples, Holm family adjustments |
| **Inspection Notebook & Visuals** | Prakhyath Shetty | Shri Hari S | `notebooks/`, `figures/` | Clean cell execution and vector PDF generation |
| **Manuscript & Presentation Slides** | Joint | Joint | `report/`, `slides/` | Rubric compliance and cross-verification |

---

## 2. Individual Reflections

### Prakhyath Shetty (Roll No: 261090830014)
* **Technical Decision:** Implemented BDeu Dirichlet smoothing ($\alpha = 1.0$) on conditional probability tables to prevent zero-probability crashes during counterfactual abduction on sparse demographic cells.
* **Difficulty / Failure:** Initially, conditioning on all downstream features during single counterfactual queries yielded identical posteriors due to intermediate d-separation blocking. Resolved this by propagating interventional shifts downstream through each DAG's specific proxy pathway.
* **Cross-Check of Partner's Work:** Independently audited the PyTorch minimax loss implementation in `src/models.py`, confirming that gradient backpropagation correctly routes through the $\operatorname{argmax}_{G_k}$ branch during batch training.

### Shri Hari S (Roll No: 261090830005)
* **Technical Decision:** Adopted a frozen client-level 60/20/20 split and computed all bin thresholds solely on training data to guarantee zero leakage into the test audit.
* **Difficulty / Failure:** In Experiment 4, severe minority support thinning ($< 5\%$) caused the paired bootstrap variance to widen significantly, initially masking method differentiation. Resolved by reporting exact denominators and isolating sampling variance from structural graph uncertainty.
* **Cross-Check of Partner's Work:** Re-ran the Variable Elimination and Clique Tree inference routines across all candidate DAGs, independently validating that calibration differences stayed below machine precision ($10^{-16}$).
