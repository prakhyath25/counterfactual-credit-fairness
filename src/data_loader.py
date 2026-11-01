import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "raw", "default_of_credit_card_clients.xls")
PROCESSED_DIR = os.path.join("data", "processed")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


def download_uci_dataset(dest_path=RAW_DATA_PATH):
    """Downloads the official UCI Credit Card dataset if not locally present."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print("Downloading UCI Default of Credit Card Clients dataset...")
        urllib.request.urlretrieve(UCI_URL, dest_path)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")


def load_and_preprocess(dest_path=RAW_DATA_PATH):
    """Loads and standardizes variables into our 6-10 substantive causal system."""
    download_uci_dataset(dest_path)
    
    # Read Excel skipping initial descriptive header row
    df = pd.read_excel(dest_path, header=1)
    
    # Standardize column names
    df.columns = [c.strip().upper() for c in df.columns]
    
    # 1. Target Recoding: Non-default Y=1 (repaid), Y=0 (default)
    # Original: default payment next month (1 = default, 0 = no default)
    default_col = "DEFAULT PAYMENT NEXT MONTH" if "DEFAULT PAYMENT NEXT MONTH" in df.columns else "DEFAULT"
    df["Y"] = 1 - df[default_col].astype(int)
    
    # 2. Protected Attribute: A (Sex: 1 = Male, 2 = Female in raw UCI)
    # Recode to 0/1: 0 = Male, 1 = Female
    df["A"] = (df["SEX"] == 2).astype(int)
    
    # 3. Context / Age Bands: C (<35: 0, 35-54: 1, >=55: 2)
    conditions = [
        (df["AGE"] < 35),
        (df["AGE"] >= 35) & (df["AGE"] < 55),
        (df["AGE"] >= 55)
    ]
    df["C"] = np.select(conditions, [0, 1, 2], default=1).astype(int)
    
    # 4. Intersectional Group G = A x C (0 through 5)
    df["G"] = df["A"] * 3 + df["C"]
    
    # 5. Education E: Group 1=Grad, 2=Univ, 3=HighSchool, 0=Other
    df["E"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3] else 0).astype(int)
    
    # 6. Credit Limit L: Discretize into ordinal financial capacity bands
    df["LIMIT_BAL_RAW"] = df["LIMIT_BAL"]
    
    # 7. Consolidated Repayment Summary R (max repayment delay over last 6 months)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["R_RAW"] = df[pay_cols].max(axis=1)
    # 0 = No delay / paid in full, 1 = 1 month delay, 2 = 2+ months delay
    df["R"] = np.where(df["R_RAW"] <= 0, 0, np.where(df["R_RAW"] == 1, 1, 2))
    
    # 8. Consolidated Bill and Payment Summary B (Total Bill / Total Paid)
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["TOTAL_BILL"] = df[bill_cols].sum(axis=1).clip(lower=0)
    df["TOTAL_PAID"] = df[pay_amt_cols].sum(axis=1).clip(lower=0)
    
    # Retain primary 8 substantive causal variables
    core_cols = ["A", "C", "G", "E", "LIMIT_BAL_RAW", "R", "TOTAL_BILL", "TOTAL_PAID", "Y"]
    return df[core_cols]


def split_and_save(df, processed_dir=PROCESSED_DIR, random_state=42):
    """Splits into 60/20/20 Train, Validation, Test. Discretizes L and B using Train only."""
    os.makedirs(processed_dir, exist_ok=True)
    
    # Stratified split by Group (G) and Outcome (Y)
    strat_col = df["G"].astype(str) + "_" + df["Y"].astype(str)
    train_val, test_df = train_test_split(
        df, test_size=0.20, random_state=random_state, stratify=strat_col
    )
    
    strat_val = train_val["G"].astype(str) + "_" + train_val["Y"].astype(str)
    train_df, val_df = train_test_split(
        train_val, test_size=0.25, random_state=random_state, stratify=strat_val
    )  # 0.25 * 0.80 = 0.20
    
    # Bin continuous variables using training quantiles only to prevent data leakage
    # Discretize Limit L (Low: 0, Medium: 1, High: 2)
    l_bins = [0, train_df["LIMIT_BAL_RAW"].quantile(0.33), train_df["LIMIT_BAL_RAW"].quantile(0.66), np.inf]
    for split_data in [train_df, val_df, test_df]:
        split_data["L"] = pd.cut(split_data["LIMIT_BAL_RAW"], bins=l_bins, labels=[0, 1, 2]).astype(int)
        
    # Discretize Bill ratio B
    bill_ratio_train = (train_df["TOTAL_PAID"] + 1) / (train_df["TOTAL_BILL"] + 1)
    b_bins = [0, bill_ratio_train.quantile(0.33), bill_ratio_train.quantile(0.66), np.inf]
    for split_data in [train_df, val_df, test_df]:
        ratio = (split_data["TOTAL_PAID"] + 1) / (split_data["TOTAL_BILL"] + 1)
        split_data["B"] = pd.cut(ratio, bins=b_bins, labels=[0, 1, 2]).astype(int)
        
    # Drop raw intermediate variables
    final_cols = ["A", "C", "G", "E", "L", "R", "B", "Y"]
    train_final = train_df[final_cols].copy()
    val_final = val_df[final_cols].copy()
    test_final = test_df[final_cols].copy()
    
    train_final.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_final.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_final.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    
    print(f"Data successfully split and saved to {processed_dir}:")
    print(f"  Train: {train_final.shape}")
    print(f"  Val:   {val_final.shape}")
    print(f"  Test:  {test_final.shape}")
    return train_final, val_final, test_final


if __name__ == "__main__":
    raw_df = load_and_preprocess()
    split_and_save(raw_df)