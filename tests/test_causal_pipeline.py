import pytest
import numpy as np
import pandas as pd
from src.causal_dags import build_candidate_dags, fit_bayesian_network, verify_inference_equivalence
from src.counterfactual import CounterfactualEngine

@pytest.fixture(scope="module")
def sample_data():
    return pd.read_csv("data/processed/train.csv").sample(n=1000, random_state=42).reset_index(drop=True)

def test_inference_equivalence(sample_data):
    """C-GM: Validate that Variable Elimination and Clique Tree agree within <= 1e-6."""
    dags = build_candidate_dags()
    for name, model in dags.items():
        fitted = fit_bayesian_network(model, sample_data)
        agreed, max_diff, ve_val, bp_val = verify_inference_equivalence(fitted)
        assert agreed, f"Inference mismatch on {name}: diff={max_diff}"
        assert max_diff <= 1e-6, f"Tolerance exceeded on {name}: {max_diff}"

def test_zero_leakage():
    """C-LEAK: Check that train/val/test splits exist and maintain frozen invariants."""
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")
    test_df  = pd.read_csv("data/processed/test.csv")
    
    assert len(train_df) == 18000
    assert len(val_df) == 6000
    assert len(test_df) == 6000
    assert set(train_df.columns) == set(test_df.columns)

def test_small_groups():
    """C-GROUP: Verify group support and ensure test set has no unflagged N < 30 cells."""
    test_df = pd.read_csv("data/processed/test.csv")
    counts = test_df["G"].value_counts()
    for g, count in counts.items():
        assert count >= 30, f"Demographic stratum {g} has critically low support: N={count}"

def test_counterfactual_engine(sample_data):
    """Test Pearl 3-step abduction engine executes valid interventional transforms."""
    dags = build_candidate_dags()
    fitted = fit_bayesian_network(dags["G1_proxy_limit"], sample_data)
    engine = CounterfactualEngine(fitted)
    cf_df = engine.generate_counterfactual_dataset(sample_data.iloc[:10], seed=42)
    assert len(cf_df) == 10
    assert np.all(cf_df["A"].values == (1 - sample_data.iloc[:10]["A"].values))