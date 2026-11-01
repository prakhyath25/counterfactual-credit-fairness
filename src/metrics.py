import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def compute_fairness_and_utility(y_true, y_prob, group_labels, threshold=0.5):
    """
    Computes primary predictive utility metrics and causal/group fairness disparities.
    D = 1 denotes simulated selection for low-risk credit clients.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    group_labels = np.asarray(group_labels)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Clip probabilities to prevent log(0) numerical overflow safely across all sklearn versions
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)

    # 1. Predictive Utility
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    ce_loss = log_loss(y_true, y_prob_clipped)
    
    # 2. Group Rates
    unique_groups = np.unique(group_labels)
    sg = {}  # P(D = 1 | G = g)
    tg = {}  # P(D = 1 | Y = 1, G = g) (True Positive Rate)
    fg = {}  # P(D = 1 | Y = 0, G = g) (False Positive Rate)
    support_flags = {}
    
    for g in unique_groups:
        idx_g = (group_labels == g)
        n_g = np.sum(idx_g)
        
        # Guardrail requirement 7: Flag small groups (<30 observations)
        support_flags[g] = bool(n_g < 30)
        
        sg[g] = np.mean(y_pred[idx_g]) if n_g > 0 else 0.0
        
        pos_idx = idx_g & (y_true == 1)
        tg[g] = np.mean(y_pred[pos_idx]) if np.sum(pos_idx) > 0 else 0.0
        
        neg_idx = idx_g & (y_true == 0)
        fg[g] = np.mean(y_pred[neg_idx]) if np.sum(neg_idx) > 0 else 0.0
        
    s_vals = list(sg.values())
    t_vals = list(tg.values())
    f_vals = list(fg.values())
    
    delta_dp = max(s_vals) - min(s_vals)
    delta_eopp = max(t_vals) - min(t_vals)
    delta_eodds = max(delta_eopp, max(f_vals) - min(f_vals))
    
    max_s = max(s_vals)
    min_s = min(s_vals)
    r_sel = (min_s / max_s) if max_s > 0 else np.nan
    
    return {
        "AUROC": auroc,
        "AP": ap,
        "Brier": brier,
        "LogLoss": ce_loss,
        "Delta_DP": delta_dp,
        "Delta_EOpp": delta_eopp,
        "Delta_EOdds": delta_eodds,
        "R_sel": r_sel,
        "Group_Support_Flags": support_flags
    }


def paired_bootstrap_intervals(y_true, y_prob_base, y_prob_novel, groups, n_bootstraps=1000, seed=42):
    """
    Computes paired difference and 95% bootstrap confidence intervals
    across independent client sampling units.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        m_base = compute_fairness_and_utility(y_true[idx], y_prob_base[idx], groups[idx])
        m_novel = compute_fairness_and_utility(y_true[idx], y_prob_novel[idx], groups[idx])
        # Track difference in primary fairness criterion (Delta EOdds)
        diffs.append(m_novel["Delta_EOdds"] - m_base["Delta_EOdds"])
        
    diffs = np.array(diffs)
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    return np.mean(diffs), (ci_lower, ci_upper)