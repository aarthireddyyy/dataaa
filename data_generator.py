import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')
import json
import os


# ─────────────────────────────────────────────
# 1. GENERATE REALISTIC MESSY DATASET
#    Simulating Kaggle "E-Commerce Customer Churn"
# ─────────────────────────────────────────────
np.random.seed(42)
N = 15000

# Base signal
tenure = np.random.exponential(scale=24, size=N).clip(0, 120)
age    = np.random.normal(35, 12, N).clip(18, 80)
spend  = np.random.lognormal(mean=5.5, sigma=1.1, size=N).clip(10, 5000)
login_freq = np.random.poisson(8, N).clip(0, 60)

# Target (churn) — dependent on real signals
churn_prob = 1 / (1 + np.exp(0.05*tenure - 0.01*login_freq + 0.0003*spend - 0.8))
churn = (np.random.random(N) < churn_prob).astype(int)

data = {
    'customer_id':      [f'CUST_{i:05d}' for i in range(N)],
    'age':              age,
    'gender':           np.random.choice(['Male','Female','male','female','M','F','Other', None], N,
                                          p=[0.3,0.3,0.1,0.1,0.07,0.07,0.03,0.03]),
    'city':             np.random.choice(['Mumbai','Delhi','Bangalore','Chennai','Hyderabad',
                                          'Kolkata','Pune','Mumbai City','bengaluru','DELHI', np.nan], N,
                                          p=[0.18,0.17,0.13,0.1,0.1,0.08,0.08,0.04,0.04,0.04,0.04]),
    'tenure_months':    tenure,
    'contract_type':    np.random.choice(['Month-to-Month','One Year','Two Year', 'month-to-month', np.nan], N,
                                          p=[0.45,0.25,0.22,0.05,0.03]),
    'monthly_charges':  spend / (tenure.clip(1,120)),
    'total_spend':      spend,
    'num_products':     np.random.randint(1, 8, N),
    'login_frequency':  login_freq,
    'last_login_days':  np.random.exponential(15, N).clip(0, 365),
    'support_tickets':  np.random.poisson(1.5, N),
    'payment_method':   np.random.choice(['Credit Card','UPI','Net Banking','Debit Card',
                                          'credit card','COD', np.nan], N,
                                          p=[0.25,0.25,0.2,0.15,0.05,0.05,0.05]),
    'device':           np.random.choice(['Mobile','Desktop','Tablet','mobile', np.nan], N,
                                          p=[0.5,0.3,0.13,0.05,0.02]),
    'satisfaction_score': np.random.choice([1,2,3,4,5,99,-1, np.nan], N,
                                            p=[0.08,0.12,0.2,0.3,0.2,0.04,0.03,0.03]),
    'referral_source':  np.random.choice(['Organic','Social Media','Email','Paid Ad','Referral', np.nan], N,
                                          p=[0.25,0.2,0.2,0.18,0.12,0.05]),
    'churn':            churn
}

df_raw = pd.DataFrame(data)

# Inject more realistic messiness
# 1. Missing values in key numeric cols
for col, frac in [('age',0.06),('tenure_months',0.04),('total_spend',0.07),
                  ('monthly_charges',0.05),('login_frequency',0.03)]:
    mask = np.random.random(N) < frac
    df_raw.loc[mask, col] = np.nan

# 2. Outliers (data entry errors)
outlier_idx = np.random.choice(N, 120, replace=False)
df_raw.loc[outlier_idx[:40], 'age'] = np.random.choice([0, 150, 999, -5], 40)
df_raw.loc[outlier_idx[40:80], 'total_spend'] = np.random.uniform(50000, 200000, 40)
df_raw.loc[outlier_idx[80:], 'monthly_charges'] = np.random.uniform(-500, -1, 40)

# 3. Duplicates
dup_rows = df_raw.sample(200, random_state=1)
df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

# 4. Whitespace / mixed types in some fields
df_raw.loc[np.random.choice(len(df_raw), 150), 'city'] = '  '
df_raw.loc[np.random.choice(len(df_raw), 80), 'contract_type'] = ''

df_raw.to_csv('messy_ecommerce.csv', index=False)
print(f"Dataset shape: {df_raw.shape}")
print(df_raw.dtypes)
print("\nMissing values:\n", df_raw.isnull().sum())