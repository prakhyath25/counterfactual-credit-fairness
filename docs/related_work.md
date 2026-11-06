# Literature Comparison Table & Research Gap Positioning

## 1. Comparative Analysis Matrix

| Reference | Graph Assumptions | Graphical Model Family | Fairness Definitions | Inference & Estimation Methods | Evaluation Datasets | Limitations & Fragility |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Koller & Friedman (2009) [1]** | Exact DAG factorization; Markov properties; Positivity. | Discrete Bayesian & Markov Networks, Clique Trees. | Probabilistic conditional independence (X ⟂ Y | Z). | Variable Elimination, Junction Tree calibration, BDeu parameter estimation. | Synthetic / Benchmark networks (Asia, Alarm). | Methodological foundation; lacks causal fairness objectives and decision-rule audits. |
| **Kusner et al. (2017) [6]** | Single known causal DAG; additive exogenous disturbances (U ⟂ A). | Single Structural Causal Model (SCM). | Counterfactual Fairness: P(Y_{A ← a} | X) = P(Y_{A ← a'} | X) | MCMC parameter abduction; detec forward propagation. | Law School (LSAC), COMPAS. | Complete failure under graph misspecification; assumes graph is known without uncertainty. |
| **Valerio et al. (2026) [7]** | Graph uncertainty represented via distribution over candidate DAGs or partial DAGs. | Causal Bayesian Network ensemble. | Soft Counterfactual Fairness across graph posterior. | Bayesian model averaging across graph posteriors; posterior sampling. | Semi-synthetic datasets. | Averaging washes out severe worst-case proxy pathways; sensitive to prior graph distribution. |
| **Su et al. (2026) [8]** | Imperfect causal graphs with unknown or missing directed edges. | Partially Directed / Local Directed Graphs. | Bounded Counterfactual Fairness (local causal bounds). | Linear programming / partial identification bounds on causal effects. | Adult, German Credit. | Bounds become overly wide and vacuous when demographic groups have sparse support (N < 30). |
| **Our Proposed Approach** | Finite candidate set G = {G1, ..., GK} of substantively grounded DAGs; exogenous disturbance abduction. | Discrete Bayesian Networks with Dirichlet-smoothed CPTs. | Minimax Counterfactual Discrepancy: max_{G_k} E[\|f(X) - f(X^(G_k)_{cf})\|] + Equalized Odds (Delta EOdds). | Exact Variable Elimination & Clique Tree calibrated inference (<1e-6 diff); PyTorch minimax regularizer. | UCI Default of Credit Card Clients (350) [9] + Parametric SCM Simulator. | Minimax bounds hold over G; vulnerable if unobserved confounding violates causal sufficiency outside G. |

---

## 2. Formal Research Gap

Existing frameworks for counterfactual fairness under graph uncertainty either average over graph models (which dilutes localized proxy penalties) or construct partial identification bounds (which expand into uninformative intervals under small-sample support). No prior work evaluates whether counterfactual audit rankings remain stable when proxy pathway ambiguity intersects directly with empirical sample sparsity in small demographic cells (N < 30).

## 3. Falsifiable Contribution Sentence

> *"Penalizing a predictive classifier by its worst-case counterfactual score discrepancy across a bounded set of plausible proxy graphs reduces empirical cross-graph audit variance by more than 30% compared to single-graph and average-graph penalties while maintaining an AUROC degradation within 2.0% on the UCI Credit Card benchmark."*
