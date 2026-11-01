import warnings
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation

# Silence external pgmpy structure score warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def build_candidate_dags():
    """
    Defines 4 substantive causal graphs:
      G1 (Direct & Indirect Proxy): Sex (A) affects Limit (L) directly, plus downstream R, B, and Y.
      G2 (Unbiased Allocation): A does NOT cause Limit L (A is connected via demographic context A -> E without an edge to L).
      G3 (Behavioral Proxy): Demographics affect behavioral repayment directly.
      G4 (Fully Mediated Proxy): A influences E, which cascades to L, R, B, and outcome Y.
    """
    dag_structures = {
        # Graph 1: Standard proxy pathway (A -> L)
        "G1_proxy_limit": [
            ("A", "L"), ("C", "L"), ("E", "L"),
            ("L", "R"), ("L", "B"),
            ("R", "Y"), ("B", "Y")
        ],
        # Graph 2: Unbiased allocation: A is separated from L
        "G2_unbiased_allocation": [
            ("A", "E"), ("C", "E"),
            ("C", "L"),
            ("L", "R"), ("L", "B"),
            ("R", "Y"), ("B", "Y")
        ],
        # Graph 3: Demographics affect behavioral repayment directly
        "G3_proxy_repayment": [
            ("C", "L"), ("E", "L"),
            ("A", "R"), ("L", "R"), ("L", "B"),
            ("R", "Y"), ("B", "Y")
        ],
        # Graph 4: Fully mediated proxy through education
        "G4_mediated_education": [
            ("A", "E"), ("C", "E"),
            ("E", "L"), ("L", "R"), ("L", "B"),
            ("R", "Y"), ("B", "Y")
        ]
    }
    
    models = {}
    for name, edges in dag_structures.items():
        models[name] = DiscreteBayesianNetwork(edges)
    return models


def fit_bayesian_network(model, train_df, pseudo_counts=1.0):
    """
    Fits Bayesian Network CPTs with Dirichlet (Laplace) pseudo-count smoothing
    directly using TabularCPD to guarantee stability across all pgmpy versions.
    """
    sub_df = train_df[[col for col in model.nodes()]].copy()
    
    # Precompute cardinality of each node from observed unique values
    state_cards = {node: int(sub_df[node].nunique()) for node in model.nodes()}
    # Ensure minimum cardinality of 2 for binary nodes (like A, Y)
    for node in state_cards:
        if state_cards[node] < 2:
            state_cards[node] = 2

    model.cpds = []
    
    for node in model.nodes():
        parents = list(model.get_parents(node))
        node_card = state_cards[node]
        
        if not parents:
            # Prior distribution without parents: counts + pseudo_counts
            counts = sub_df[node].value_counts().to_dict()
            vals = np.array([counts.get(val, 0) + pseudo_counts for val in range(node_card)], dtype=float)
            probs = (vals / vals.sum()).reshape(-1, 1)
            
            cpd = TabularCPD(
                variable=node,
                variable_card=node_card,
                values=probs
            )
        else:
            # Conditional distribution: counts conditioned on parent configurations
            parent_cards = [state_cards[p] for p in parents]
            
            # Cross-tabulate node vs parents
            grouped = sub_df.groupby(parents + [node], observed=False).size().unstack(fill_value=0)
            
            # Ensure all node states exist
            for s in range(node_card):
                if s not in grouped.columns:
                    grouped[s] = 0
            grouped = grouped[sorted(grouped.columns)]
            
            # Generate all possible parent combinations for full multi-index alignment
            all_parent_tuples = pd.MultiIndex.from_product(
                [range(state_cards[p]) for p in parents], 
                names=parents
            )
            grouped = grouped.reindex(all_parent_tuples, fill_value=0)
            
            # Add Dirichlet pseudo-counts and normalize per parent configuration
            smoothed_counts = grouped.values.astype(float) + pseudo_counts
            probs = smoothed_counts / smoothed_counts.sum(axis=1, keepdims=True)
            
            # TabularCPD expects shape (node_card, num_parent_configurations)
            cpd = TabularCPD(
                variable=node,
                variable_card=node_card,
                values=probs.T,
                evidence=parents,
                evidence_card=parent_cards
            )
            
        model.add_cpds(cpd)
        
    assert model.check_model(), "Model validation failed after CPD assignment!"
    return model


def verify_inference_equivalence(model, test_evidence={"A": 1, "C": 1, "L": 1}, target="Y", tolerance=1e-6):
    """
    Mandatory check from Experiment Protocol E5/E6:
    Verifies exact mathematical agreement between Variable Elimination and Clique Tree (Belief Propagation).
    """
    ve = VariableElimination(model)
    bp = BeliefPropagation(model)
    
    valid_evidence = {k: int(v) for k, v in test_evidence.items() if k in model.nodes()}
    
    q_ve = ve.query(variables=[target], evidence=valid_evidence, show_progress=False)
    q_bp = bp.query(variables=[target], evidence=valid_evidence, show_progress=False)
    
    diff = np.max(np.abs(q_ve.values - q_bp.values))
    agrees = bool(diff < tolerance)
    return agrees, diff, q_ve.values, q_bp.values


if __name__ == "__main__":
    train_data = pd.read_csv("data/processed/train.csv")
    dags = build_candidate_dags()
    
    print("Fitting candidate Bayesian Networks and verifying inference correctness...")
    for name, model in dags.items():
        fitted = fit_bayesian_network(model, train_data)
        agreed, max_diff, ve_res, bp_res = verify_inference_equivalence(fitted)
        status = "PASSED" if agreed else "FAILED"
        print(f"[{status}] {name:25} | Max VE vs. CliqueTree Diff: {max_diff:.8e}")
        assert agreed, f"Inference divergence detected in {name}!"
    print("All graphical models passed inference validation.")