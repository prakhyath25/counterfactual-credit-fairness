import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal_dags import build_candidate_dags, fit_bayesian_network
from src.counterfactual import CounterfactualEngine
from src.models import MinimaxCounterfactualClassifier
from src.metrics import compute_fairness_and_utility


def run_experiment_2():
    print("=" * 80)
    print("EXPERIMENT 2: Matched-Utility Fairness Trade-off Frontier (Calibrated Threshold)")
    print("=" * 80)

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    dags = build_candidate_dags()
    fitted_dags = {name: fit_bayesian_network(model, train_df) for name, model in dags.items()}

    print("Generating counterfactual samples for training and test evaluation...")
    train_sample = train_df.sample(n=3000, random_state=42)
    test_sample = test_df.sample(n=1500, random_state=42)
    
    train_cfs = {}
    test_cfs = {}
    for name, model in fitted_dags.items():
        engine = CounterfactualEngine(model)
        train_cfs[name] = engine.generate_counterfactual_dataset(train_sample)
        test_cfs[name] = engine.generate_counterfactual_dataset(test_sample)

    lambdas = [0.0, 0.05, 0.15, 0.3, 0.6, 1.2]
    tradeoff_records = []

    y_test = test_sample["Y"].values
    g_test = test_sample["G"].values

    for lam in lambdas:
        clf = MinimaxCounterfactualClassifier(
            input_dim=6, penalty_lambda=lam, penalty_type="minimax", epochs=50, lr=0.005
        )
        clf.fit(train_sample, train_cfs)
        
        preds_factual = clf.predict_proba(test_sample)
        
        # Determine operating threshold on validation set (Top 65% lowest-risk selection rule)
        operating_threshold = float(np.percentile(preds_factual, 35))
        
        # Calculate worst-case counterfactual score shift across candidate graphs
        cf_shifts = []
        cf_flips = []
        for name, cf_df in test_cfs.items():
            preds_cf = clf.predict_proba(cf_df)
            shift = np.mean(np.abs(preds_factual - preds_cf))
            flips = np.mean((preds_factual >= operating_threshold) != (preds_cf >= operating_threshold))
            cf_shifts.append(shift)
            cf_flips.append(flips)
            
        worst_cf_shift = max(cf_shifts)
        worst_cf_flip = max(cf_flips)

        metrics = compute_fairness_and_utility(
            y_test, preds_factual, g_test, threshold=operating_threshold
        )
        
        tradeoff_records.append({
            "Lambda": lam,
            "AUROC": round(metrics["AUROC"], 4),
            "Worst_CF_Shift": round(worst_cf_shift, 4),
            "Worst_CF_Flip": round(worst_cf_flip, 4),
            "Delta_EOdds": round(metrics["Delta_EOdds"], 4),
            "Delta_DP": round(metrics["Delta_DP"], 4),
            "R_sel": round(metrics["R_sel"], 4)
        })

    tradeoff_df = pd.DataFrame(tradeoff_records)
    print("\nRealistic Matched-Utility Frontier (Simulated Top 65% Selection):")
    print("-" * 85)
    print(tradeoff_df.to_string(index=False))
    print("-" * 85)
    print("Experiment 2 completed successfully.")


if __name__ == "__main__":
    run_experiment_2()