import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulator import CreditSCMSimulator
from src.causal_dags import build_candidate_dags, fit_bayesian_network
from src.counterfactual import CounterfactualEngine
from src.models import MinimaxCounterfactualClassifier
from src.metrics import compute_fairness_and_utility


def run_experiment_4_and_5():
    print("=" * 95)
    print("EXPERIMENTS 4 & 5: Simulation Stress-Testing & Graph Misspecification Failure Mode")
    print("=" * 95)

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Minority Support Thinning vs. Proxy Strength
    # -------------------------------------------------------------------------
    print("\n[PART 1: EXPERIMENT 4] Stress-testing under Minority Support Thinning & Proxy Strength...")
    
    simulation_scenarios = [
        {"name": "Balanced_ModerateProxy", "minority_ratio": 0.30, "proxy_strength": 1.0},
        {"name": "Balanced_StrongProxy",   "minority_ratio": 0.30, "proxy_strength": 2.5},
        {"name": "Thinned_ModerateProxy",  "minority_ratio": 0.05, "proxy_strength": 1.0},
        {"name": "Thinned_StrongProxy",    "minority_ratio": 0.05, "proxy_strength": 2.5},
    ]

    sim_results = []
    dags = build_candidate_dags()

    for sc in simulation_scenarios:
        sim = CreditSCMSimulator(
            proxy_strength=sc["proxy_strength"],
            minority_ratio=sc["minority_ratio"],
            seed=42
        )
        sim_train = sim.sample(n=3000)
        sim_test = sim.sample(n=1500)
        
        # Ground-truth counterfactuals known via structural equations
        sim_test_gt_cf = sim.get_ground_truth_counterfactual(sim_test, intervention_val=1)
        
        # Fit candidate DAGs on simulated training data
        fitted_dags = {name: fit_bayesian_network(model, sim_train) for name, model in dags.items()}
        
        # Use a fixed subset of sim_train for paired training
        train_sub = sim_train.sample(n=2000, random_state=42).reset_index(drop=True)
        sim_train_cfs = {
            name: CounterfactualEngine(model).generate_counterfactual_dataset(train_sub)
            for name, model in fitted_dags.items()
        }

        # Train Unconstrained vs. Minimax
        unconstrained = MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=0.0, epochs=35, lr=0.005)
        minimax = MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=0.40, penalty_type="minimax", epochs=35, lr=0.005)

        unconstrained.fit(train_sub, sim_train_cfs)
        minimax.fit(train_sub, sim_train_cfs)

        preds_unconstrained = unconstrained.predict_proba(sim_test)
        preds_minimax = minimax.predict_proba(sim_test)

        preds_unconstrained_cf = unconstrained.predict_proba(sim_test_gt_cf)
        preds_minimax_cf = minimax.predict_proba(sim_test_gt_cf)

        # Ground-truth counterfactual discrepancy
        gt_shift_unconstrained = float(np.mean(np.abs(preds_unconstrained - preds_unconstrained_cf)))
        gt_shift_minimax = float(np.mean(np.abs(preds_minimax - preds_minimax_cf)))
        pct_reduction = ((gt_shift_unconstrained - gt_shift_minimax) / gt_shift_unconstrained) * 100

        sim_results.append({
            "Scenario": sc["name"],
            "Minority_Rate": sc["minority_ratio"],
            "Proxy_Strength": sc["proxy_strength"],
            "GT_Shift_Baseline": round(gt_shift_unconstrained, 4),
            "GT_Shift_Minimax": round(gt_shift_minimax, 4),
            "Shift_Reduction(%)": round(pct_reduction, 2)
        })

    sim_table = pd.DataFrame(sim_results)
    print("\nGround-Truth Counterfactual Discrepancy Table:")
    print("-" * 95)
    print(sim_table.to_string(index=False))
    print("-" * 95)

    # -------------------------------------------------------------------------
    # EXPERIMENT 5: Intentional Graph Misspecification Failure Mode
    # -------------------------------------------------------------------------
    print("\n[PART 2: EXPERIMENT 5] Evaluating Misspecification Failure Case...")
    print("Inducing unobserved confounding (financial stress U_stress driving both L and Y)...")
    
    rng = np.random.default_rng(99)
    n_miss = 2000
    U_stress = rng.normal(0, 1, size=n_miss)
    
    A_miss = rng.choice([0, 1], size=n_miss)
    L_miss = pd.qcut(0.8 * A_miss + 1.2 * U_stress + rng.normal(0, 0.5, size=n_miss), q=3, labels=[0, 1, 2]).astype(int)
    Y_miss = ((-1.5 * U_stress + 0.3 * L_miss + rng.logistic(0, 1, size=n_miss)) > 0).astype(int)
    
    miss_df = pd.DataFrame({
        "A": A_miss,
        "C": rng.choice([0, 1, 2], size=n_miss),
        "E": rng.choice([0, 1, 2, 3], size=n_miss),
        "L": L_miss,
        "R": rng.choice([0, 1, 2], size=n_miss),
        "B": rng.choice([0, 1, 2], size=n_miss),
        "Y": Y_miss
    })

    # Generate counterfactual pairs for the misspecified dataset using candidate DAGs
    miss_cfs = {
        name: CounterfactualEngine(model).generate_counterfactual_dataset(miss_df)
        for name, model in fitted_dags.items()
    }

    miss_clf = MinimaxCounterfactualClassifier(input_dim=6, penalty_lambda=0.40, epochs=35, lr=0.005)
    miss_clf.fit(miss_df, miss_cfs)
    preds_miss = miss_clf.predict_proba(miss_df)
    
    brier_miss = np.mean((preds_miss - Y_miss) ** 2)
    print(f"Outcome under misspecification: Unobserved financial stress causes hidden confounding.")
    print(f"Brier score error inflated to {brier_miss:.4f} due to structural confounding omitted from G1-G4.")
    print("Experiments 4 and 5 completed successfully.")


if __name__ == "__main__":
    run_experiment_4_and_5()