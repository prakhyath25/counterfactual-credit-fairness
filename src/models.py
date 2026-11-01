import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


class StandardBaselines:
    """Logistic Regression and Gradient Boosted Trees predictive baselines."""
    def __init__(self, use_protected=True):
        self.use_protected = use_protected
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.gbt_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        
    def _get_x(self, df):
        cols = ["C", "E", "L", "R", "B"]
        if self.use_protected:
            cols = ["A"] + cols
        return df[cols].values

    def fit(self, train_df):
        X = self._get_x(train_df)
        y = train_df["Y"].values
        self.lr_model.fit(X, y)
        self.gbt_model.fit(X, y)

    def predict_proba(self, test_df):
        X = self._get_x(test_df)
        return {
            "LR": self.lr_model.predict_proba(X)[:, 1],
            "GBT": self.gbt_model.predict_proba(X)[:, 1]
        }


class CompactCreditNet(nn.Module):
    """Compact neural network for counterfactual-regularized credit scoring."""
    def __init__(self, input_dim=6, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class MinimaxCounterfactualClassifier:
    """
    Minimax model: optimizes cross-entropy loss plus the worst-case counterfactual
    sensitivity penalty across candidate graphs G_1, ..., G_K.
    """
    def __init__(self, input_dim=6, penalty_lambda=1.5, penalty_type="minimax", lr=0.01, epochs=60):
        self.penalty_lambda = penalty_lambda
        self.penalty_type = penalty_type  # 'minimax', 'single', or 'average'
        self.epochs = epochs
        self.lr = lr
        self.model = CompactCreditNet(input_dim=input_dim)
        self.features = ["A", "C", "E", "L", "R", "B"]

    def fit(self, train_df, cf_datasets_dict):
        """
        cf_datasets_dict: mapping from graph_name -> paired counterfactual DataFrame.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        bce_criterion = nn.BCELoss()

        X_factual = torch.tensor(train_df[self.features].values, dtype=torch.float32)
        y_factual = torch.tensor(train_df["Y"].values, dtype=torch.float32).unsqueeze(1)

        # Pre-convert counterfactual feature matrices
        X_cfs = {
            g_name: torch.tensor(cf_df[self.features].values, dtype=torch.float32)
            for g_name, cf_df in cf_datasets_dict.items()
        }

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds_factual = self.model(X_factual)
            loss_pred = bce_criterion(preds_factual, y_factual)

            # Compute counterfactual discrepancies |f(X) - f(X_cf)| across candidate graphs
            graph_penalties = []
            for g_name, X_cf in X_cfs.items():
                preds_cf = self.model(X_cf)
                discrepancy = torch.mean(torch.abs(preds_factual - preds_cf))
                graph_penalties.append(discrepancy)

            # Penalty selection rule
            if self.penalty_type == "minimax":
                fairness_penalty = torch.max(torch.stack(graph_penalties))
            elif self.penalty_type == "average":
                fairness_penalty = torch.mean(torch.stack(graph_penalties))
            elif self.penalty_type == "single":
                fairness_penalty = graph_penalties[0]
            else:
                fairness_penalty = torch.tensor(0.0)

            total_loss = loss_pred + self.penalty_lambda * fairness_penalty
            total_loss.backward()
            optimizer.step()

    def predict_proba(self, df):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(df[self.features].values, dtype=torch.float32)
            probs = self.model(X).squeeze(1).numpy()
        return probs


if __name__ == "__main__":
    train_data = pd.read_csv("data/processed/train.csv")
    
    # Quick sanity check with standard baselines
    baselines = StandardBaselines(use_protected=True)
    baselines.fit(train_data)
    preds = baselines.predict_proba(train_data)
    print("Baseline predictions generated:")
    print(f"  LR Mean Pred:  {np.mean(preds['LR']):.4f}")
    print(f"  GBT Mean Pred: {np.mean(preds['GBT']):.4f}")
    print("Models module verified.")