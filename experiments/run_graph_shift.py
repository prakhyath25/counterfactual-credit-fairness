import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal_dags import build_candidate_dags, fit_bayesian_network
from src.counterfactual import CounterfactualEngine
from src.models import MinimaxCounterfactualClassifier
from src.metrics import compute_fairness_and_utility


def run_experiment_3_and_ablations():
    print("=" * 95)
    print("EXPERIMENT 3 & ABLATION STUDY: Minimax vs. Single vs. Average Graph Penalties")
    print("=" * 95)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    dags = build_candidate_dags()
    fitted_dags = {name: fit_bayesian_network(model, train_df) for name, model in dags.items()}

    print("\n[1/3] Generating counterfactual pairs per candidate graph...")
    train_sample = train_df.sample(n=3000, random_state=42)
    test_sample = test_df.sample(n=1500, random_state=42)

    train_cfs = {}
    test_cfs = {}
    for name, model in fitted_dags.items():
        engine = CounterfactualEngine(model)
        train_cfs[name] = engine.generate_counterfactual_dataset(train_sample)
        test_cfs[name] = engine.generate_counterfactual_dataset(test_sample)

    LAMBDA_OPT = 0.40
    models = {
        "Unconstrained Baseline": MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=0.0, epochs=45, lr=0.005),
        "Single-Graph Penalty (G1)": MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=LAMBDA_OPT, penalty_type="single", epochs=45, lr=0.005),
        "Average-Graph Penalty": MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=LAMBDA_OPT, penalty_type="average", epochs=45, lr=0.005),
        "Minimax Worst-Graph (Proposed)": MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=LAMBDA_OPT, penalty_type="minimax", epochs=45, lr=0.005),
    }

    print("\n[2/3] Training ablation variants with identical budget...")
    for label, clf in models.items():
        clf.fit(train_sample, train_cfs)

    print("\n[3/3] Auditing method stability across each candidate graph...")
    graph_names = list(dags.keys())
    audit_rows = []
    y_test = test_sample["Y"].values
    g_test = test_sample["G"].values

    for model_name, clf in models.items():
        preds_factual = clf.predict_proba(test_sample)
        tau = float(np.percentile(preds_factual, 35))
        metrics = compute_fairness_and_utility(y_test, preds_factual, g_test, threshold=tau)
        
        row = {
            "Model Strategy": model_name,
            "AUROC": round(metrics["AUROC"], 4)
        }
        shifts_across_graphs = []

        for g_name in graph_names:
            preds_cf = clf.predict_proba(test_cfs[g_name])
            shift = float(np.mean(np.abs(preds_factual - preds_cf)))
            shifts_across_graphs.append(shift)
            row[f"Shift_{g_name[:8]}"] = round(shift, 4)

        row["Worst_Shift"] = round(max(shifts_across_graphs), 4)
        row["Audit_Variance"] = round(float(np.var(shifts_across_graphs)), 6)
        audit_rows.append(row)

    audit_table = pd.DataFrame(audit_rows)
    print("\nCross-Graph Audit Discrepancy & Utility Table:")
    print("-" * 105)
    print(audit_table.to_string(index=False))
    print("-" * 105)
    print("Experiment 3 and Ablation Study completed successfully.")


if __name__ == "__main__":
    run_experiment_3_and_ablations()