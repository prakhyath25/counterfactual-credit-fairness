import os
import sys
import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal_dags import build_candidate_dags, fit_bayesian_network
from src.counterfactual import CounterfactualEngine
from src.models import StandardBaselines, MinimaxCounterfactualClassifier
from src.metrics import compute_fairness_and_utility, paired_bootstrap_intervals


def run_experiment_1():
    print("=" * 80)
    print("EXPERIMENT 1: Frozen Test Set Audit & Intersectional Disparity Analysis")
    print("=" * 80)

    # 1. Load frozen datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # 2. Fit candidate Bayesian Networks
    print("\n[1/4] Fitting candidate causal DAGs on training split...")
    dags = build_candidate_dags()
    fitted_dags = {name: fit_bayesian_network(model, train_df) for name, model in dags.items()}

    # 3. Pre-generate counterfactual datasets across candidate graphs for train and test splits
    print("[2/4] Generating counterfactual pairs across all candidate graphs...")
    # Use a representative sample of train data for regularized training speed
    train_sample = train_df.sample(n=3000, random_state=42)
    train_cfs = {}
    for name, model in fitted_dags.items():
        engine = CounterfactualEngine(model)
        train_cfs[name] = engine.generate_counterfactual_dataset(train_sample)

    # 4. Fit Models: Baselines vs. Minimax Robust Model
    print("[3/4] Fitting predictive baselines and minimax counterfactual classifier...")
    # Baseline 1: Standard GBT with Protected Attribute
    gbt_aware = StandardBaselines(use_protected=True)
    gbt_aware.fit(train_df)

    # Baseline 2: Attribute Blind (A removed)
    gbt_blind = StandardBaselines(use_protected=False)
    gbt_blind.fit(train_df)

    # Proposed Method: Minimax worst-graph regularizer
    minimax_clf = MinimaxCounterfactualClassifier(
        input_dim=6, penalty_lambda=2.0, penalty_type="minimax", epochs=40
    )
    minimax_clf.fit(train_sample, train_cfs)

    # 5. Evaluate on Frozen Test Split
    print("[4/4] Evaluating fairness & predictive utility on frozen test set...")
    preds_aware = gbt_aware.predict_proba(test_df)["GBT"]
    preds_blind = gbt_blind.predict_proba(test_df)["GBT"]
    preds_minimax = minimax_clf.predict_proba(test_df)

    eval_models = {
        "GBT (Aware)": preds_aware,
        "GBT (Blind - A removed)": preds_blind,
        "Minimax CF-Penalized": preds_minimax
    }

    results = []
    y_test = test_df["Y"].values
    g_test = test_df["G"].values

    for model_name, probas in eval_models.items():
        metrics = compute_fairness_and_utility(y_test, probas, g_test, threshold=0.5)
        results.append({
            "Model": model_name,
            "AUROC": f"{metrics['AUROC']:.4f}",
            "AP": f"{metrics['AP']:.4f}",
            "Delta_EOdds": f"{metrics['Delta_EOdds']:.4f}",
            "Delta_DP": f"{metrics['Delta_DP']:.4f}",
            "R_sel": f"{metrics['R_sel']:.4f}"
        })

    results_table = pd.DataFrame(results)
    print("\n" + results_table.to_string(index=False))

    # Bootstrap paired difference between Minimax and Baseline GBT on Delta EOdds
    mean_diff, (ci_low, ci_high) = paired_bootstrap_intervals(
        y_test, preds_aware, preds_minimax, g_test, n_bootstraps=500
    )
    print("\nPaired Bootstrap Analysis (Minimax vs. GBT Aware on Delta EOdds):")
    print(f"  Mean Shift: {mean_diff:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print("\nExperiment 1 completed successfully.")


if __name__ == "__main__":
    run_experiment_1()