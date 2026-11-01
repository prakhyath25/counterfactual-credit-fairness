# Model Card: Minimax Counterfactual Fair Credit Classifier

## 1. Model Overview
* **Architecture:** Compact feed-forward neural network (`CompactCreditNet`: 6 inputs, 16 hidden units [ReLU], 8 hidden units [ReLU], 1 output unit [Sigmoid]).
* **Objective Function:**
  `min_theta  L_BCE(Y, Y_hat) + lambda * max_{G_k in G} E[ |f(X, A) - f(X^(G_k)_{A <- 1-A}, 1-A)| ]`
* **Optimizer:** Adam (lr = 0.005, weight_decay = 1e-4), batch size = full batch / 2,000 subsample.
* **Optimal Operating Point:** `lambda = 0.40`, operating at the top 65% selection rate on validation probabilities.

## 2. Intended Use & Scope
* **Application Domain:** Lending risk audit simulators and counterfactual fairness stress-testing under model uncertainty.
* **Out of Scope:** Real-world autonomous credit underwriting, automated adverse lending rejection, or downstream pricing without human adjudication.

## 3. Quantitative Performance Summary (Frozen Test Set)

| Evaluation Metric | GBT (Aware) Baseline | GBT (Blind - A removed) | Proposed Minimax Classifier |
| :--- | :--- | :--- | :--- |
| **AUROC** | 0.7507 | 0.7481 | 0.7401 |
| **Average Precision (AP)** | 0.8987 | 0.8966 | 0.8842 |
| **Delta EOdds** | 0.2660 | 0.2184 | 0.1742 |
| **Delta DP** | 0.1499 | 0.1249 | 0.0952 |
| **Worst-Graph CF Shift** | 0.1669 | 0.1580 | 0.1073 |
| **95% Bootstrap CI (Delta EOdds)** | Baseline | [-0.081, +0.024] | [-0.152, -0.041] (Statistically Significant) |
