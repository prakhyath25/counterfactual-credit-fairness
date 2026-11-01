import numpy as np
import pandas as pd


class CreditSCMSimulator:
    """
    Parametric Structural Causal Model (SCM) generator with known exogenous disturbances U
    and closed-form counterfactual capabilities.
    """
    def __init__(self, proxy_strength=1.5, minority_ratio=0.2, seed=42):
        self.proxy_strength = proxy_strength  # Controls weight of A -> L and A -> R
        self.minority_ratio = minority_ratio  # Controls P(A = 0)
        self.rng = np.random.default_rng(seed)

    def sample(self, n=5000):
        """Samples observational data while preserving exogenous noise states."""
        # 1. Exogenous disturbances
        U_A = self.rng.uniform(0, 1, size=n)
        U_C = self.rng.choice([0, 1, 2], p=[0.4, 0.45, 0.15], size=n)
        U_E = self.rng.choice([0, 1, 2, 3], p=[0.1, 0.35, 0.45, 0.1], size=n)
        U_L = self.rng.normal(0, 1, size=n)
        U_R = self.rng.normal(0, 1, size=n)
        U_B = self.rng.normal(0, 1, size=n)
        U_Y = self.rng.logistic(0, 1, size=n)

        # 2. Structural Equations
        # Protected attribute A
        A = (U_A > self.minority_ratio).astype(int)
        C = U_C
        E = U_E

        # Credit Limit L: influenced by Context and Proxy edge from A
        l_latent = 0.5 * C + 0.8 * E + self.proxy_strength * A + U_L
        L = pd.qcut(l_latent, q=3, labels=[0, 1, 2]).astype(int)

        # Repayment R: influenced by Limit and behavioral proxy
        r_latent = -0.6 * L + 0.3 * self.proxy_strength * A + U_R
        R = pd.qcut(r_latent, q=3, labels=[0, 1, 2]).astype(int)

        # Bills B
        b_latent = 0.7 * L - 0.4 * R + U_B
        B = pd.qcut(b_latent, q=3, labels=[0, 1, 2]).astype(int)

        # Non-default outcome Y
        y_logit = 1.2 + 0.8 * L - 1.5 * R - 0.5 * B + U_Y
        Y = (y_logit > 0).astype(int)

        # Intersectional Group G = A x C
        G = A * 3 + C

        df = pd.DataFrame({
            "A": A, "C": C, "G": G, "E": E, "L": L, "R": R, "B": B, "Y": Y,
            "U_L": U_L, "U_R": U_R, "U_B": U_B, "U_Y": U_Y
        })
        return df

    def get_ground_truth_counterfactual(self, df, intervention_val=1):
        """
        Computes exact ground-truth counterfactuals using structural equations
        and frozen factual exogenous noise U.
        """
        n = len(df)
        A_cf = np.full(n, intervention_val)
        C = df["C"].values
        E = df["E"].values
        
        # Abducted exogenous noise (preserved exactly)
        U_L = df["U_L"].values
        U_R = df["U_R"].values
        U_B = df["U_B"].values
        U_Y = df["U_Y"].values

        # Propagate forward under intervention do(A = intervention_val)
        l_latent_cf = 0.5 * C + 0.8 * E + self.proxy_strength * A_cf + U_L
        L_cf = pd.qcut(l_latent_cf, q=3, labels=[0, 1, 2]).astype(int)

        r_latent_cf = -0.6 * L_cf + 0.3 * self.proxy_strength * A_cf + U_R
        R_cf = pd.qcut(r_latent_cf, q=3, labels=[0, 1, 2]).astype(int)

        b_latent_cf = 0.7 * L_cf - 0.4 * R_cf + U_B
        B_cf = pd.qcut(b_latent_cf, q=3, labels=[0, 1, 2]).astype(int)

        y_logit_cf = 1.2 + 0.8 * L_cf - 1.5 * R_cf - 0.5 * B_cf + U_Y
        Y_cf = (y_logit_cf > 0).astype(int)

        cf_df = pd.DataFrame({
            "A": A_cf, "C": C, "G": A_cf * 3 + C, "E": E,
            "L": L_cf, "R": R_cf, "B": B_cf, "Y": Y_cf
        })
        return cf_df


if __name__ == "__main__":
    sim = CreditSCMSimulator(proxy_strength=2.0, minority_ratio=0.15)
    data = sim.sample(n=2000)
    cf_data = sim.get_ground_truth_counterfactual(data, intervention_val=1)
    
    flip_rate = np.mean(data["Y"] != cf_data["Y"])
    print(f"Synthetic SCM generated {len(data)} profiles.")
    print(f"Ground-truth counterfactual outcome flip rate under do(A=1): {flip_rate:.4f}")