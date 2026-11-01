# Data Card: UCI Default of Credit Card Clients (Dataset 350)

## 1. Dataset Provenance & Metadata
* **Dataset Identifier:** UCI Machine Learning Repository, ID 350 (Yeh & Lien, 2009) [9].
* **Collection Scope:** 30,000 credit card accounts from an issuance bank in Taiwan (April 2005 – September 2005).
* **Licensing:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Split Protocol:** Stratified 60% Train (18,000), 20% Validation (6,000), 20% Test (6,000) partitioned over intersectional groups G and target Y. All discretization and scaling parameters frozen solely on training splits.

## 2. Variable Construction & Recoding
* **Target Y (Binary):** Non-default next month (`Y = 1` indicates full non-default repayment; `Y = 0` denotes payment default). Recoded directly from raw default column (`1 - raw_default`).
* **Decision D in {0, 1}:** Simulated selection for a low-risk customer product tier. **This does not reflect original loan approval or originatn decisions.**
* **Protected Attribute A:** Recorded sex (`A = 0` for Male [raw 1], `A = 1` for Female [raw 2]).
* **Context Variable C:** Age bands (`0 = <35 yrs`, `1 = 35–54 yrs`, `2 = ≥55 yrs`).
* **Intersectional Strata G:** Cross-product `A x C in {0, 1, 2, 3, 4, 5}`.
* **Financial Features:**
  * **E:** Education level categorized into Graduate, University, High School, Other.
  * **L:** Discretized credit limit (quantiles on training set: Low, Medium, High).
  * **R:** Consolidated repayment delay score (maximum payment delay observed across past 6 months).
  * **B:** Consolidated financial burden (ratio of total payments made to total bill amount).

## 3. Demographic Representation & Small Group Safeguards
* **Group Proportions in UCI:**
  * Female (`A = 1`): ~60.4%
  * Male (`A = 0`): ~39.6%
* **Underrepresented Strata:** Group `G = 2` (Male ≥ 55) and Group `G = 5` (Female ≥ 55) represent < 5% of the cohort.
* **Support Rule:** In alignment with Protocol Item 7, any evaluation subset or bootratum with `N < 30` observations is explicitly flagged as exploratory and excluded from confirmatory disparity claims.

## 4. Ethical Nuance & Limitations
* **Proxy Nature:** Recorded sex is a biological and administrative categorization, not a complete measure of gender identity.
* **Selection Bias:** The dataset exclusively records existing account holders who were previously granted credit cards. It carries no information on rejected credit applicants.
