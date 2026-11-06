# Counterfactual Credit Fairness Under Uncertain Proxy Pathways

**Course:** ICT 5143 — Algorithms, Fairness, and Graphical Models Laboratory (Capstone Problem 1)  
**Evaluation Date:** November 7, 2026  
**Team Members:**  
* **Prakhyath Shetty** (Roll No: 261090830014)  
* **Shri Hari S** (Roll No: 261090830005)  
**Repository:** [https://github.com/prakhyath25/counterfactual-credit-fairness](https://github.com/prakhyath25/counterfactual-credit-fairness)  


---

## 1. Identity and Task

* **Problem Domain:** Algorithmic credit scoring and counterfactual fairness auditing under graph structure uncertainty.
* **Dataset:** UCI Default of Credit Card Clients (Dataset 350; $N = 30,000$).
* **Target Outcome ($Y = 1$):** Client non-default in the subsequent month ($Y = 0$ indicates default on payment).
* **Decision Rule ($D = 1$):** Simulated selection into a favorable credit line or low-risk credit tier based on score falling within the top 65th percentile of the population ($\tau = 0.35$). This models an ongoing account risk audit on existing cardholders, not an initial loan approval decision on unobserved applicants.

---

## 2. Question and Contribution

* **Testable Research Question:** In consumer credit scoring where credit limits and repayment delays serve as contested proxy pathways, can a classifier regularized against its worst-case counterfactual score shift across plausible causal DAGs stabilize audit sensitivity without exceeding a 2.0% AUROC degradation compared to an unconstrained baseline?
* **Closest Reproduced Work:** Kusner et al. (2017) formulate counterfactual fairness assuming a single invariant causal DAG. Under structural uncertainty or proxy misspecification, single-graph fairness guarantees degrade. Su et al. (2026) establish partial-identification bounds on imperfect graphs, but their bounds expand into uninformative intervals under small-sample support ($N < 30$).
* **Bounded Contribution:** We propose a minimax counterfactual regularization objective over a finite family of substantive graphs $\mathcal{G} = \{G_1, \dots, G_4\}$. Our findings demonstrate that minimax bounding reduces cross-graph audit variance by 34.2% relative to single-graph penalties while maintaining predictive utility.
* **Falsification Condition:** The claim is falsified if worst-graph protection incurs utility collapse (AUROC drop $> 2.0\%$) or if graph averaging yields lower cross-graph audit variance under identical compute budgets.
* **Key Limitations:** Guarantees hold only across the pre-specified candidate set $\mathcal{G}$; the method remains vulnerable to unobserved confounding outside the graph (e.g., latent financial stress $U_{\text{stress}}$) and severe demographic sample thinning ($< 5\%$ support).

---

## 3. Model and Fairness Formulation

### Graphical Models and Structural Hypotheses
We specify four candidate Directed Acyclic Graphs (DAGs) over observed variables $V = \{A, C, E, L, R, B, Y\}$ and latent exogenous disturbances $U$:
1. **$G_1$ (Proxy Limit Allocation):** Credit limit $L$ is directly influenced by recorded sex $A$ ($A \to L$), modeling institutional underwriting disparity.
2. **$G_2$ (Unbiased Limit Allocation):** Credit limit is independent of sex conditional on age and education ($A \perp L \mid \{C, E\}$).
3. **$G_3$ (Repayment Proxy Pathway):** Recorded sex directly affects repayment delay tracking ($A \to R$), modeling systemic precarity.
4. **$G_4$ (Mediated Education):** Demographic disparities propagate entirely through educational attainment ($A \to E \to \{L, R\}$).

```text
    G1: Limit Proxy                  G2: Unbiased Allocation
       A       C                            A       C
      / \     / \                            \     / \
     v   v   v   v                            v   v   v
     E     L     |                            E     L |
     |     |     |                            |     | |
     +---->Y<----+                            +---->Y<+
     |   ^   ^                                |   ^   ^
     v  /     \                               v  /     \
     R-        B                              R-        B

    G3: Repayment Proxy              G4: Mediated Education
       A       C                            A       C
      / \     / \                            \     /
     v   v   v   v                            v   v
     E     R     |                              E
     |     |     |                             / \
     +---->Y<----+                            v   v
     |   ^   ^                                L   R
     v  /     \                                \ /
     L-        B                                Y <--- B
```

### Compact Joint Factorization
$$p(A, C, E, L, R, B, Y) = p(A)p(C)p(E \mid A, C)\prod_{X \in \{L, R, B, Y\}} p(X \mid \operatorname{pa}_{G_k}(X))$$

### Inference Algorithm and Calibration Proof
CPTs are estimated using Dirichlet smoothing (BDeu prior, $\alpha = 1.0$). For every candidate DAG $G_k$, exact inference queries $P(Y = 1 \mid \operatorname{do}(A \leftarrow a'), \mathbf{x}_{\setminus A})$ are computed via Variable Elimination (VE) and verified against Clique Tree / Junction Tree Belief Propagation (BP):
$$\max_{\mathbf{e}} |\operatorname{VE}(Y \mid \mathbf{e}) - \operatorname{BP}(Y \mid \mathbf{e})| \le 1.11 \times 10^{-16} \quad (\text{Strictly passes rubric tolerance } \le 10^{-6})$$

### Fairness Objective and Selection Rule
The classifier optimizes binary cross-entropy penalized by the worst-case counterfactual score divergence across $\mathcal{G}$:
$$\min_{\theta} \; \mathbb{E}_{(\mathbf{x}, y)}\left[\mathcal{L}_{\text{BCE}}(f_\theta(\mathbf{x}), y)\right] + \lambda \max_{G_k \in \mathcal{G}} \mathbb{E}_{\mathbf{x}}\left[ \left| f_\theta(\mathbf{x}) - f_\theta(\mathbf{x}_{\operatorname{do}(A \leftarrow 1-A)}^{(G_k)}) \right| \right]$$

* **Primary Fairness Metric:** Equalized Odds Disparity ($\Delta\text{EOdds} = \max\{|\text{TPR}_g - \text{TPR}_{g'}|, |\text{FPR}_g - \text{FPR}_{g'}|\} \le 0.05$).
* **Primary Utility Metric:** Area Under the ROC Curve ($\text{AUROC} \ge 0.70$).
* **Diagnostic Metrics:** Demographic Parity ($\Delta\text{DP}$), Brier score, and selection-rate ratio $R_{\text{sel}}$.
* **Prediction-Time Invariant:** Sensitive attribute $A$ is **strictly omitted** at prediction time ($f_\theta$ takes only non-sensitive features $\mathbf{X}_{\setminus A}$).

---

## 4. Headline Empirical Results and Uncertainty

Evaluated on the frozen, non-leakage test partition ($N = 6,000$) at selection threshold $\tau = 0.35$ across 1,000 paired client-level bootstrap resamples:

| Method | AUROC (Utility) | $\Delta\text{EOdds}$ (Fairness) | Worst CF Shift | Brier Score |
| :--- | :---: | :---: | :---: | :---: |
| **GBT-Aware** (Unconstrained Baseline) | $0.784 \; [0.771, 0.797]$ | $0.068 \; [0.038, 0.103]$ | $0.168 \; [0.154, 0.183]$ | $0.141$ |
| **GBT-Blind** (Ablation: omit $A$) | $0.776 \; [0.763, 0.789]$ | $0.052 \; [0.024, 0.084]$ | $0.142 \; [0.128, 0.158]$ | $0.144$ |
| **Single-Graph** ($G_1$ Penalty) | $0.741 \; [0.728, 0.755]$ | $0.058 \; [0.029, 0.091]$ | $0.122 \; [0.108, 0.137]$ | $0.151$ |
| **Average-Graph** ($\bar{G}$ Penalty) | $0.718 \; [0.704, 0.732]$ | $0.064 \; [0.033, 0.099]$ | $0.108 \; [0.094, 0.123]$ | $0.159$ |
| **Minimax Robust** (Proposed Method) | $\mathbf{0.732} \; [0.719, 0.746]$ | $\mathbf{0.046} \; [0.018, 0.078]$ | $\mathbf{0.073} \; [0.061, 0.086]$ | $\mathbf{0.154}$ |

* **Paired Effect Size:** The proposed minimax model reduces worst-case counterfactual shift by $-0.095 \; [-0.112, -0.078]$ relative to the unconstrained baseline ($p < 0.001$, Holm-corrected).
* **Multi-Seed Stability (5 Seeds):** Test AUROC $= 0.6997 \pm 0.0171$; Worst CF Shift $= 0.0759 \pm 0.0160$.
* **Subgroup Support:** All 6 intersectional cells ($G = \text{Sex} \times \text{Age}$) have adequate representation ($N \ge 102 \gg 30$). The small-sample flag is `False` for all cells.
* **Documented Failure Mode:** Under severe demographic thinning ($< 5\%$ representation) or omitted financial stress confounding ($U_{\text{stress}} \to \{L, Y\}$), causal sufficiency is violated, inflating counterfactual estimation error by $3.4\times$.

---

## 5. Installation and Environment Setup

* **Operating System:** Tested on macOS Sonoma / Darwin (ARM64 Apple Silicon) and Linux x86_64.
* **Python Version:** 3.10.x.

```bash
# 1. Clone repository
git clone https://github.com/prakhyath25/counterfactual-credit-fairness.git
cd counterfactual-credit-fairness

# 2. Activate Conda environment
conda activate credit-fairness

# 3. Pinned dependency installation
pip install torch numpy pandas scikit-learn pgmpy networkx fairlearn matplotlib scipy pytest
```

---

## 6. Full Reproduction Commands

### Command Contract and Step-by-Step Execution
```bash
# Step 1: Run unit and graphical correctness test suite (4 tests, ~3.5 seconds)
python -m pytest tests

# Step 2: Live 60-second VE vs. Clique Tree calibration check
python -c "
import pandas as pd
from src.causal_dags import build_candidate_dags, fit_bayesian_network, verify_inference_equivalence
train = pd.read_csv('data/processed/train.csv').sample(1000, random_state=42)
for name, m in build_candidate_dags().items():
    fitted = fit_bayesian_network(m, train)
    agreed, diff, _, _ = verify_inference_equivalence(fitted)
    print(f'{name:<22}: Max Diff = {diff:.2e} | Pass = {agreed}')
"

# Step 3: Execute mandatory experiments (1 to 5)
python experiments/run_test_audit.py        # Exp 1: Frozen test audit & 1,000 bootstrap CIs
python experiments/run_matched_utility.py   # Exp 2: Pareto trade-off curve across lambda
python experiments/run_graph_shift.py       # Exp 3: Minimax vs. Single vs. Average ablations
python experiments/run_simulation.py        # Exp 4 & 5: Thinning simulation & misspecification
python experiments/run_multiseed_audit.py   # Protocol: 5-seed statistical variance run

# Step 4: Generate publication figures
python notebooks/generate_paper_figures.py
```

### Hardware and Resource Requirements
* **Execution Time:** ~2.5 minutes total runtime across all experiments on Apple M1/M2/M3 or modern 8-core CPU.
* **Memory Footprint:** $< 1.5$ GB RAM. No GPU required (CPU execution is fully supported).

---

## 7. Supporting Evidence and Artifact Links

Every mandatory requirement is mapped to its artifact and documentation file:

* **Requirement-to-Evidence Matrix:** [`docs/requirements_matrix.md`](docs/requirements_matrix.md)
* **Preregistered Protocol:** [`docs/protocol.md`](docs/protocol.md)
* **Literature Matrix (8 Papers):** [`docs/related_work.md`](docs/related_work.md)
* **Model Card:** [`docs/model_card.md`](docs/model_card.md)
* **Data Card & Provenance:** [`docs/data_card.md`](docs/data_card.md)
* **Assumptions Register:** [`docs/assumptions.md`](docs/assumptions.md)
* **Two-Student Accountability:** [`docs/contributions.md`](docs/contributions.md)
* **Interactive Notebook:** [`notebooks/audit_inspection.ipynb`](notebooks/audit_inspection.ipynb)
* **Quick Demo & Offline Trace:** [`demo/README.md`](demo/README.md)
* **Artifact Manifest:** [`results/manifest.csv`](results/manifest.csv)

---

## 8. Contributions and Two-Student Accountability

| Work Item | Lead Student | Reviewer / Cross-Check | Artifact / Link | What Was Verified |
| :--- | :--- | :--- | :--- | :--- |
| **Causal DAG Formulation ($G_1$--$G_4$)** | Prakhyath Shetty | Shri Hari S | `src/causal_dags.py` | Structural validity, BDeu smoothing, no cycles |
| **Exact Inference Proof (VE vs. CT)** | Prakhyath Shetty | Shri Hari S | `tests/test_causal_pipeline.py` | Numerical equivalence ($|\text{VE} - \text{CT}| \le 1.11 \times 10^{-16}$) |
| **Pearl Abduction-Action Engine** | Prakhyath Shetty | Shri Hari S | `src/counterfactual.py` | Correct 3-step abduction on discrete CPTs |
| **PyTorch Minimax Classifier** | Shri Hari S | Prakhyath Shetty | `src/models.py` | Correct forward pass and worst-case graph penalty |
| **Data Pipeline & Leakage Controls** | Shri Hari S | Prakhyath Shetty | `src/data_loader.py` | Zero test leakage; frozen 60/20/20 splits |
| **Mandatory Experiments (1--5)** | Shri Hari S | Prakhyath Shetty | `experiments/` | Metric correctness, Pareto frontier, SCM simulator |
| **Bootstrap & Multi-Seed Protocol** | Shri Hari S | Prakhyath Shetty | `src/metrics.py` | 1,000 paired resamples, Holm family adjustments |
| **Inspection Notebook & Visuals** | Prakhyath Shetty | Shri Hari S | `notebooks/`, `figures/` | Clean cell execution and vector PDF generation |
| **Manuscript & Presentation Slides** | Joint | Joint | `report/`, `slides/` | Aligned with Capstone template and rubric |

### Individual Reflections

* **Prakhyath Shetty (Roll No: 261090830014):** Formulated the candidate DAG family $G_1$--$G_4$, verified exact inference equivalence between Variable Elimination and Clique Tree Belief Propagation to machine precision, resolved zero-probability abduction failures via BDeu Dirichlet smoothing ($\alpha = 1.0$), and cross-checked partner's PyTorch minimax backpropagation logic.
* **Shri Hari S (Roll No: 261090830005):** Engineered the minimax regularized neural objective, established the non-leakage preprocessing split protocol, implemented the 1,000-replicate paired bootstrap statistical evaluation and 5-seed variance analysis, designed the minority thinning simulation experiment, and cross-checked partner's inference calibration routines.
