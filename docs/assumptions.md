# Explicit Assumptions Register (Protocol Item 2 & Causal Safeguards)

## 1. Causal Model Assumptions
1. **Causal Sufficiency within G:** Exogenous disturbances `U_i` are assumed mutually independent conditional on the parent set for candidate DAGs `G1, ..., G4`. (Experiment 5 evaluates the breakdown of this assumption under unobserved confounding).
2. **Positivity:** For all observed conditioning configurations `X = x`, `0 < P(A = a | X = x) < 1`.
3. **Invariance of Non-Descendants:** Interventions on recorded sex `do(A = a')` modify only topological descendants of `A` in each respective graph; mechanisms for non-descendant context features (`C`) are preserved.

## 2. Statistical & Inference Assumptions
1. **Dirichlet Prior Smoothing:** Tabular conditional probability tables use symmetric BDeu Dirichlet priors (`α = 1.0`) to guarantee non-zero posterior densities during the abduction step.
2. **Inference Equivalence:** Exact equality between Variable Elimiation and Clique Tree (Belief Propagation) algorithms (`|Marginal_VE - Marginal_CT| < 1e-6`).
3. **Resampling Unit:** Bootstrap confidence intervals are computed over independent borrower records (`N = 1,000` replicates), evaluating evaluation uncertainty rather than population sampling variance.
