import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.causal_dags import build_candidate_dags, fit_bayesian_network
from src.counterfactual import CounterfactualEngine
from src.models import MinimaxCounterfactualClassifier
from src.metrics import compute_fairness_and_utility

SEEDS = [42, 101, 2024, 7, 999]

def run_multiseed_evaluation():
    print("=" * 80)
    print("STATISTICAL PROTOCOL: Multi-Seed Robustness Evaluation (5 Seeds)")
    print("=" * 80)
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    dags = build_candidate_dags()
    fitted_dags = {name: fit_bayesian_network(model, train_df) for name, model in dags.items()}
    
    train_sample = train_df.sample(n=2000, random_state=42).reset_index(drop=True)
    test_sample = test_df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    train_cfs = {
        name: CounterfactualEngine(model).generate_counterfactual_dataset(train_sample)
        for name, model in fitted_dags.items()
    }
    test_cfs = {
        name: CounterfactualEngine(model).generate_counterfactual_dataset(test_sample)
        for name, model in fitted_dags.items()
    }
    
    seed_records = []
    
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        
        clf = MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=0.40, epochs=30, lr=0.005)
        clf.fit(train_sample, train_cfs)
        
        preds_fac = clf.predict_proba(test_sample)
        tau = float(np.percentile(preds_fac, 35))
        
        shifts = []
        for g_name in dags.keys():
            preds_cf = clf.predict_proba(test_cfs[g_name])
            shifts.append(float(np.mean(np.abs(preds_fac - preds_cf))))
            
        m = compute_fairness_and_utility(test_sample["Y"].values, preds_fac, test_sample["G"].values, threshold=tau)
        
        seed_records.append({
            "Seed": s,
            "AUROC": m["AUROC"],
            "Delta_EOdds": m["Delta_EOdds"],
            "Delta_DP": m["Delta_DP"],
            "Worst_CF_Shift": max(shifts)
        })
        
    df_seeds = pd.DataFrame(seed_records)
    print("\nMulti-Seed Empirical Results:")
    print("-" * 80)
    print(df_seeds.to_string(index=False))
    print("-" * 80)
    print("Aggregate Summary (Mean ± Std):")
    print(df_seeds.drop(columns="Seed").agg(["mean", "std"]).round(4).to_string())
    print("-" * 80)

if __name__ == "__main__":
    run_multiseed_evaluation()
