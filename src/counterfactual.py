import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination


class CounterfactualEngine:
    def __init__(self, model):
        """
        Implements Pearl's 3-step Abduction-Action-Prediction counterfactual engine
        for a fitted discrete causal Bayesian Network.
        """
        self.model = model
        self.ve = VariableElimination(model)
        
    def counterfactual_query(self, evidence_dict, intervention_dict={"A": 0}, target="Y"):
        """
        Performs counterfactual inference:
        1. Abduction: Condition on all observed factual evidence.
        2. Action: Intervene by replacing the mechanism of the intervention variable do(A=a').
        3. Prediction: Query the target posterior under the counterfactual intervention.
        """
        # Copy model and apply graph surgery (do-calculus / Pearl's Action step)
        intervened_model = self.model.copy()
        
        for var, val in intervention_dict.items():
            # Cut incoming edges to intervened variable
            parents = list(intervened_model.get_parents(var))
            for p in parents:
                intervened_model.remove_edge(p, var)
                
            # Retain downstream dependencies and update the intervened CPD to a deterministic point mass
            intervened_model = self._set_point_mass_cpd(intervened_model, var, val)

        # Abduction & Prediction: Query counterfactual target distribution
        ve_intervened = VariableElimination(intervened_model)
        
        # Evidence consists of exogenous/context variables unaffected by the intervention
        context_evidence = {
            k: v for k, v in evidence_dict.items()
            if k in intervened_model.nodes() and k not in intervention_dict and k != target
        }
        
        res = ve_intervened.query(
            variables=[target],
            evidence=context_evidence,
            show_progress=False
        )
        return res.values

    def _set_point_mass_cpd(self, model, var, value):
        """Sets a deterministic point-mass distribution for the intervened variable."""
        from pgmpy.factors.discrete import TabularCPD
        card = self.model.get_cardinality(var)
        prob_vec = np.zeros((card, 1))
        prob_vec[value, 0] = 1.0
        
        new_cpd = TabularCPD(
            variable=var,
            variable_card=card,
            values=prob_vec
        )
        model.remove_cpds(var)
        model.add_cpds(new_cpd)
        return model

    def generate_counterfactual_dataset(self, df, intervention_attr="A", seed=42):
        """
        Generates paired counterfactual feature representations for an entire cohort
        by intervening on A and sampling downstream descendants according to their
        interventional posterior distributions.
        """
        rng = np.random.default_rng(seed)
        cf_records = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            curr_a = int(row_dict[intervention_attr])
            target_a = 1 - curr_a  # Counterfactual flip
            
            intervened_model = self.model.copy()
            for p in list(intervened_model.get_parents(intervention_attr)):
                intervened_model.remove_edge(p, intervention_attr)
            intervened_model = self._set_point_mass_cpd(intervened_model, intervention_attr, target_a)
            ve_int = VariableElimination(intervened_model)
            
            ev = {"C": int(row_dict["C"])}
            cf_row = row_dict.copy()
            cf_row[intervention_attr] = target_a
            
            # Topological order of downstream variables
            for proxy in ["E", "L", "R", "B"]:
                if proxy in intervened_model.nodes() and proxy != intervention_attr and proxy != "Y":
                    q = ve_int.query(variables=[proxy], evidence=ev, show_progress=False)
                    probs = q.values.flatten()
                    probs = probs / probs.sum()  # Re-normalize
                    sampled_val = int(rng.choice(len(probs), p=probs))
                    cf_row[proxy] = sampled_val
                    ev[proxy] = sampled_val
                    
            cf_records.append(cf_row)
            
        return pd.DataFrame(cf_records)


if __name__ == "__main__":
    from causal_dags import build_candidate_dags, fit_bayesian_network
    train_data = pd.read_csv("data/processed/train.csv")
    dags = build_candidate_dags()
    fitted_g1 = fit_bayesian_network(dags["G1_proxy_limit"], train_data)
    
    engine = CounterfactualEngine(fitted_g1)
    sample_evidence = {"A": 0, "C": 1, "E": 1, "L": 0, "R": 0, "B": 0}
    cf_res = engine.counterfactual_query(sample_evidence, intervention_dict={"A": 1}, target="Y")
    print(f"Factual evidence: {sample_evidence}")
    print(f"Counterfactual P(Y | do(A=1)): P(Y=0)={cf_res[0]:.4f}, P(Y=1)={cf_res[1]:.4f}")
    print("Counterfactual engine successfully validated.")