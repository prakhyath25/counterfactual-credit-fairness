# Experimental and Statistical Protocol

## 1. Preregistered Choices
* **Decision Meaning:** $Y=1$ denotes non-default in the following month. $D=1$ denotes simulated low-risk selection (top 65% credit threshold, $\tau=0.35$).
* **Primary Predictive Metric:** Area Under the ROC Curve (AUROC).
* **Primary Fairness Metric:** Equalized Odds Disparity ($\Delta\text{EOdds} = \max\{\vert{}\text{TPR}_g - \text{TPR}_{g'}\vert{}, \vert{}\text{FPR}_g - \text{FPR}_{g'}\vert{}\}$).
* **Acceptable Utility Loss:** Maximum $\le 2.0\%$ AUROC loss relative to the unconstrained baseline at the selected operating point ($\lambda = 0.60$).
* **Diagnostic Metrics:** Demographic Parity ($\Delta\text{DP}$), Brier calibration score, selection-rate ratio $R_{\text{sel}}$.

## 2. Statistical Rigor
* **Evaluation Seeds:** 5 stochastic seeds (`[42, 101, 2024, 7, 999]`) measuring algorithmic variance.
* **Evaluation Uncertainty:** 1,000-replicate paired bootstrap intervals (percentile method) resampling independent clients.
* **Multiple Comparisons:** Holm-Bonferroni correction applied across the confirmatory comparison family.
* **Small-Group Rule:** Any intersectional stratum with $N < 30$ is flagged as exploratory.
