#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


# In[5]:


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
base_dir = Path(r"filepath")

train_years_labels = [
    "2012-2013", "2013-2014", "2014-2015", "2015-2016",
    "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021"
]
test_years_labels = list(range(2014, 2024))  # 2014..2023 inclusive

# Build file map
files = {}

for label in train_years_labels:
    files[f"train_{label}"] = base_dir / f"train_{label}_final.csv"

for year in test_years_labels:
    files[f"test_{year}"] = base_dir / f"test_{year}_final.csv"

# ---------------------------------------------------------
# Create summary table: N unique patients + MPR counts
# ---------------------------------------------------------
rows = []
print(f"Looking for {len(files)} files in {base_dir}...")

for name, path in files.items():
    if not path.exists():
        print(f"  Missing file: {name}")
        rows.append({
            "dataset": name,
            "file": str(path),
            "n_rows": np.nan,
            "n_unique_patients": np.nan,
            "mpr_0": np.nan,
            "mpr_1": np.nan,
            "mpr_missing": np.nan
        })
        continue

    try:
        df = pd.read_csv(path, usecols=["ID", "MPR"], dtype={"ID": str})

        n_rows = len(df)
        n_unique = df["ID"].nunique(dropna=True)

        # Make sure MPR behaves like 0/1 (handle strings, floats, blanks)
        mpr = pd.to_numeric(df["MPR"], errors="coerce")

        mpr_0 = int((mpr == 0).sum())
        mpr_1 = int((mpr == 1).sum())
        mpr_missing = int(mpr.isna().sum())

        rows.append({
            "dataset": name,
            "file": str(path),
            "n_rows": n_rows,
            "n_unique_patients": n_unique,
            "mpr_0": mpr_0,
            "mpr_1": mpr_1,
            "mpr_missing": mpr_missing
        })

        print(f"  Loaded {name}: unique patients={n_unique}, MPR0={mpr_0}, MPR1={mpr_1}, missing={mpr_missing}")

    except Exception as e:
        print(f"  Error loading {name}: {e}")
        rows.append({
            "dataset": name,
            "file": str(path),
            "n_rows": np.nan,
            "n_unique_patients": np.nan,
            "mpr_0": np.nan,
            "mpr_1": np.nan,
            "mpr_missing": np.nan
        })

summary_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

# ---------------------------------------------------------
# Optional: add % columns
# ---------------------------------------------------------
summary_df["mpr_0_pct"] = (summary_df["mpr_0"] / summary_df["n_rows"] * 100).round(1)
summary_df["mpr_1_pct"] = (summary_df["mpr_1"] / summary_df["n_rows"] * 100).round(1)
summary_df["mpr_missing_pct"] = (summary_df["mpr_missing"] / summary_df["n_rows"] * 100).round(1)

# ---------------------------------------------------------
# Save table
# ---------------------------------------------------------
out_path = base_dir / "train_test_patient_mpr_summary.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nSaved summary table to: {out_path}")

summary_df


# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
base_dir = Path(r"file path")

# 1. Define the full ranges of years
train_years_labels = [
    "2012-2013", "2013-2014", "2014-2015", "2015-2016",
    "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021"
]
test_years_labels = list(range(2014, 2023))  # 2014 to 2023

# 2. Build the dictionary dynamically
files = {}

# Add Train Files
for label in train_years_labels:
    key = f"train_{label}"
    files[key] = base_dir / f"train_{label}_final.csv"

# Add Test Files
for year in test_years_labels:
    key = f"test_{year}"
    files[key] = base_dir / f"test_{year}_final.csv"

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
id_sets = {}
print(f"Looking for {len(files)} files in {base_dir}...")

for name, path in files.items():
    if path.exists():
        try:
            # optimize by reading only ID column
            df = pd.read_csv(path, usecols=['ID'])
            id_sets[name] = set(df['ID'])
            print(f"  Loaded {name}: {len(id_sets[name])} IDs")
        except Exception as e:
            print(f"  Error loading {name}: {e}")
    else:
        print(f"  Missing file: {name}")

# ---------------------------------------------------------
# 2. Generate Heatmap (All-to-All Overlap)
# ---------------------------------------------------------
# Filter keys to only those we successfully loaded
keys = [k for k in files.keys() if k in id_sets]
n = len(keys)

if n > 0:
    overlap_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            name_i = keys[i]
            name_j = keys[j]
            intersection = len(id_sets[name_i].intersection(id_sets[name_j]))
            overlap_matrix[i, j] = intersection

    # Increased figure size for readability with more files
    plt.figure(figsize=(20, 16)) 
    sns.heatmap(overlap_matrix, annot=True, fmt='g', 
                xticklabels=keys, yticklabels=keys, cmap='Blues', 
                cbar_kws={'label': 'Count of Shared IDs'})
    plt.title("Patient Overlap Matrix (All Train & Test Files)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('overlap_heatmap_full.png')
    print("\n✅ Generated 'overlap_heatmap_full.png'")
else:
    print("\n⚠️ No files loaded, skipping heatmap.")

# ---------------------------------------------------------
# 3. Generate Sequential Bar Charts (Retention Analysis)
# ---------------------------------------------------------
def plot_sequential_overlap(series_keys, series_name):
    valid_keys = [k for k in series_keys if k in id_sets]
    if len(valid_keys) < 2:
        print(f"Skipping {series_name} (not enough valid files found)")
        return
    
    stats = []
    for i in range(len(valid_keys) - 1):
        curr, next_ = valid_keys[i], valid_keys[i+1]
        ids_curr, ids_next = id_sets[curr], id_sets[next_]
        
        overlap = len(ids_curr.intersection(ids_next))
        lost = len(ids_curr - ids_next)  # Present in curr, missing in next
        new = len(ids_next - ids_curr)   # Missing in curr, present in next
        
        stats.append({
            "Transition": f"{curr} -> {next_}",
            "Retained (Overlap)": overlap,
            "Lost (Non-Overlapping)": lost,
            "New (Non-Overlapping)": new
        })
        
    if stats:
        df_stats = pd.DataFrame(stats).set_index("Transition")
        
        # Plot
        ax = df_stats.plot(kind='bar', stacked=True, figsize=(14, 8), 
                           color=['#2ca02c', '#d62728', '#1f77b4']) # Green=Retain, Red=Lost, Blue=New
        
        plt.title(f"Patient Flow: {series_name}")
        plt.ylabel("Number of Patients")
        plt.xlabel("Year Transition")
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Category")
        plt.tight_layout()
        
        filename = f'sequential_flow_{series_name}.png'
        plt.savefig(filename)
        print(f"✅ Generated '{filename}'")

# Generate Keys for the sequential plots
train_keys = [f"train_{lbl}" for lbl in train_years_labels]
test_keys = [f"test_{yr}" for yr in test_years_labels]

# Run Analysis
plot_sequential_overlap(train_keys, "Train_Series_Full")
plot_sequential_overlap(test_keys, "Test_Series_Full")


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
base_dir = Path(r"D:\HIV Prevention\cohort\Jan_2026")

# Full ranges
train_labels = [
    "2012-2013", "2013-2014", "2014-2015", "2015-2016",
    "2016-2017", "2017-2018", "2018-2019", "2019-2020",
    "2020-2021", 
]
test_years = list(range(2014, 2023))

# ---------------------------------------------------------
# 1. Load Data (IDs only)
# ---------------------------------------------------------
id_sets = {}
print("Loading IDs...")

# Helper to load
def load_ids(fname):
    p = base_dir / fname
    if p.exists():
        return set(pd.read_csv(p, usecols=['ID'])['ID'])
    return set()

# Load Train
for lbl in train_labels:
    id_sets[f"train_{lbl}"] = load_ids(f"train_{lbl}_merged.csv")

# Load Test
for yr in test_years:
    id_sets[f"test_{yr}"] = load_ids(f"test_{yr}_merged.csv")

print(f"Loaded {len(id_sets)} file headers.")

# ---------------------------------------------------------
# 2. Logic: Calculate "Feature Coverage"
# ---------------------------------------------------------
results = []

def check_coverage(target_name, target_year, is_test=False):
    # 1. Identify Target IDs
    target_ids = id_sets.get(target_name, set())
    if not target_ids:
        return
    
    # 2. Identify Source (The Logic)
    lag_year = target_year - 1
    
    # Priority 1: Test File
    source_name = f"test_{lag_year}"
    source_ids = id_sets.get(source_name)
    
    # Priority 2: Train File (Fallback)
    if not source_ids:
        prev_start = lag_year - 1
        source_name = f"train_{prev_start}-{lag_year}"
        source_ids = id_sets.get(source_name, set())
        
    # 3. Calculate Overlap
    # How many of Target IDs exist in Source IDs?
    matches = len(target_ids.intersection(source_ids))
    total = len(target_ids)
    pct = (matches / total) * 100 if total > 0 else 0
    
    results.append({
        "File": target_name,
        "Type": "Test" if is_test else "Train",
        "Total Patients": total,
        "Valid Lag Data": matches,
        "Coverage (%)": pct,
        "Source Used": source_name if source_ids else "None"
    })

# Check Train Files
for lbl in train_labels:
    # train_2013-2014 -> Target Year is 2014
    end_year = int(lbl.split('-')[1])
    check_coverage(f"train_{lbl}", end_year, is_test=False)

# Check Test Files
for yr in test_years:
    check_coverage(f"test_{yr}", yr, is_test=True)

# ---------------------------------------------------------
# 3. Plot: Feature Coverage
# ---------------------------------------------------------
df_res = pd.DataFrame(results)

if not df_res.empty:
    plt.figure(figsize=(14, 8))
    
    # Bar plot
    ax = sns.barplot(data=df_res, x='File', y='Coverage (%)', hue='Type', 
                     palette={'Train': '#2ca02c', 'Test': '#d62728'})
    
    # Formatting
    plt.title("Feasibility Check: % of Patients with Available Prior-Year Data (PDC_Lag1)", fontsize=14)
    plt.ylabel("Coverage (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% Threshold')
    
    # Add text labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'bottom', fontsize=10, fontweight='bold')
        elif height == 0:
             ax.annotate('0%', 
                        (p.get_x() + p.get_width() / 2., 0), 
                        ha = 'center', va = 'bottom', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('feature_coverage_check.png')
    print("\n✅ Generated 'feature_coverage_check.png'")
    print(df_res[['File', 'Coverage (%)', 'Source Used']].to_string())
else:
    print("No data to plot.")


# In[26]:


# ============================================
# Classic ML (sklearn) + LSTM, target = MPR
# ============================================


# ---------------------------
# Paths
# ---------------------------
base_path = r"D:\HIV Prevention\cohort\Jan_2026\Models"

# ---------------------------
# Helpers
# ---------------------------
def normalize_sex_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """If SEX is object, normalize to M=1, F=0 with nullable Int64."""
    if 'SEX' in df.columns and df['SEX'].dtype == object:
        df['SEX'] = (
            df['SEX'].astype(str).str.strip().str.upper()
              .replace({'MALE': 'M', 'FEMALE': 'F'})
              .map({'M': 1, 'F': 0})
              .astype('Int64')
        )
    return df

def prepare_xy(df: pd.DataFrame):
    """Prepare numeric X and binary y=MPR (already 0/1)."""
    df = normalize_sex_if_needed(df)

    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset. Expected binary MPR target.")

    y = df['MPR'].astype(int)
    X = df.drop(columns=[c for c in ['MPR', 'PDC', 'AGE_at_index', 'ID', 'patient_id', 'PDC_lag_1'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

# ---------------------------
# Sklearn models
# ---------------------------
models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier()),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier()),
    "XGBoost": make_pipeline(SimpleImputer(strategy="median"), xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    "LightGBM": make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier())
}

for model_name, model in models.items():
    plt.figure(figsize=(12, 8))
    any_plotted = False

    for test_year in range(2014, 2023):
        train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
        test_file = os.path.join(base_path, f"test_{test_year}_final.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files for {test_year}, skipping: {train_file} | {test_file}")
            continue

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        del train['PDC']
        del test['PDC']
        del train['AGE_at_index']
        del test['AGE_at_index']
        # Prepare data
        X_train, y_train = prepare_xy(train)
        X_test, y_test = prepare_xy(test)

        # Class sanity
        if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            print(f"Skipping {test_year}: not enough label classes.")
            continue

        # Fit
        model.fit(X_train, y_train)

        # Proba or scores
        final_est = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(final_est, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(final_est, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = model.predict(X_test).astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})")
        any_plotted = True

        # Report
        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n{model_name} Classification Report for {test_year-2}-{test_year-1}→{test_year}")
        print(classification_report(y_test, y_pred))

    if any_plotted:
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - Yearly Temporal ROC Curves (Target=MPR)")
        plt.legend()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        roc_path = os.path.join(base_path, f"yearly_temporal_ROC_{model_name.replace(' ', '_')}.png")
        plt.savefig(roc_path)
        print(f"Saved ROC curve for {model_name} at {roc_path}")
    else:
        plt.close()
        print(f"No valid splits for {model_name}; ROC not saved.")

# ---------------------------
# LSTM (sequence_len = 1)
# ---------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).view(-1)

def prepare_xy_numeric(df: pd.DataFrame):
    """Same as prepare_xy but returns X,y arrays; numeric-only features."""
    df = normalize_sex_if_needed(df)
    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset.")
    y = df['MPR'].astype(int).values
    X = df.drop(columns=[c for c in ['MPR', 'PDC', 'AGE_at_index', 'ID', 'patient_id', 'PDC_lag_1'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

metrics_table = []
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10.colors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, test_year in enumerate(range(2014, 2023)):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
    test_file = os.path.join(base_path, f"test_{test_year}_final.csv")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Files missing for {test_year}, skipping.")
        continue

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # X, y
    X_train, y_train = prepare_xy_numeric(train)
    X_test, y_test = prepare_xy_numeric(test)

    # Class check
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(f"Skipping {test_year}: not enough classes.")
        continue

    # Impute (train medians for both)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LSTM expects (N, T, F); we set T = 1
    X_train_seq = np.expand_dims(X_train_scaled, axis=1)
    X_test_seq = np.expand_dims(X_test_scaled, axis=1)

    # DataLoaders
    train_dataset = LSTMDataset(X_train_seq, y_train)
    test_dataset = LSTMDataset(X_test_seq, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model
    model = LSTMModel(input_dim=X_train_seq.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_probs, y_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            y_probs.extend(preds.cpu().numpy())
            y_true.extend(yb.numpy())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)
    y_pred = (y_probs >= 0.5).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    metrics_table.append({
        'Model': 'LSTM',
        'Year': f"{test_year-2}-{test_year-1}→{test_year}",
        'Precision': round(precision, 3),
        'Recall (Sensitivity)': round(sensitivity, 3),
        'Specificity': round(specificity, 3),
        'F1 Score': round(f1, 3),
        'Accuracy': round(accuracy, 3)
    })

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})", color=colors[idx % len(colors)])

# Plot LSTM ROC summary
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM - Yearly Temporal ROC Curves (Target=MPR)")
plt.legend()
plt.grid()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()

roc_path = os.path.join(base_path, "yearly_temporal_ROC_LSTM.png")
plt.savefig(roc_path)
print(f"Saved LSTM ROC curve at {roc_path}")

# Save LSTM metrics table
df_metrics = pd.DataFrame(metrics_table)
print(df_metrics)
df_metrics.to_csv(os.path.join(base_path, "lstm_temporal_metrics_summary.csv"), index=False)


# In[5]:





# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve

# ---------------------------
# Setup Paths & Helpers
# ---------------------------
base_path = r"D:\HIV Prevention\cohort\Jan_2026\Models"

def normalize_sex_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if 'SEX' in df.columns and df['SEX'].dtype == object:
        df['SEX'] = (
            df['SEX'].astype(str).str.strip().str.upper()
              .replace({'MALE': 'M', 'FEMALE': 'F'})
              .map({'M': 1, 'F': 0})
              .astype('Int64')
        )
    return df

def prepare_xy(df: pd.DataFrame):
    df = normalize_sex_if_needed(df)
    # Remove clinical and ID leakage columns
    drop_cols = ['MPR', 'PDC', 'AGE_at_index', 'ID', 'patient_id', 'PDC_lag_1']
    y = df['MPR'].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def calculate_net_benefit(y_true, y_prob, thresh):
    n = len(y_true)
    tp = np.logical_and(y_prob >= thresh, y_true == 1).sum()
    fp = np.logical_and(y_prob >= thresh, y_true == 0).sum()
    if thresh == 1.0: return 0
    return (tp / n) - (fp / n) * (thresh / (1 - thresh))

# ---------------------------
# Main LightGBM Training & Plotting Loop
# ---------------------------
for test_year in range(2014, 2023):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
    test_file = os.path.join(base_path, f"test_{test_year}_final.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        continue

    # Load Data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    # Initialize Pipeline (Consistent with your best results)
    lgbm_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), 
        lgb.LGBMClassifier(random_state=42, verbose=-1)
    )
    
    # Fit Model
    lgbm_pipeline.fit(X_train, y_train)
    y_proba = lgbm_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # 1. GENERATE CALIBRATION PLOT
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='LightGBM')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    plt.title(f"Calibration - {test_year}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.savefig(os.path.join(base_path, f"Calibration_LGBM_{test_year}.png"))
    plt.close()

    # 2. GENERATE DECISION CURVE ANALYSIS (DCA)
    thresholds = np.linspace(0.01, 0.99, 100)
    nb_model = [calculate_net_benefit(y_test, y_proba, t) for t in thresholds]
    nb_all = [calculate_net_benefit(y_test, np.ones(len(y_test)), t) for t in thresholds]
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_model, color='blue', label='LightGBM')
    plt.plot(thresholds, nb_all, color='black', linestyle='--', label='Treat All')
    plt.axhline(y=0, color='red', label='Treat None')
    plt.ylim(-0.05, max(nb_model) + 0.1)
    plt.title(f"DCA - {test_year}")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend()
    plt.savefig(os.path.join(base_path, f"DCA_LGBM_{test_year}.png"))
    plt.close()

 # 3. GENERATE SHAP SUMMARY PLOTS
    # Extract imputed data and the classifier
    X_test_imputed = lgbm_pipeline.named_steps['simpleimputer'].transform(X_test)
    lgbm_clf = lgbm_pipeline.named_steps['lgbmclassifier']
    
    # Initialize the Explainer
    explainer = shap.TreeExplainer(lgbm_clf)
    shap_values = explainer.shap_values(X_test_imputed)

    # FIX: Handle the "list of ndarray" vs "ndarray" output
    # LightGBM binary classification usually returns a list [values_for_0, values_for_1]
    if isinstance(shap_values, list):
        # We want the impact on the positive class (MPR=1)
        shap_to_plot = shap_values[1]
    else:
        # Some versions return a single 3D array or a 2D array directly
        shap_to_plot = shap_values

    plt.figure(figsize=(10, 8))
    
    # Ensure shap_to_plot is a 2D matrix (samples x features)
    if len(shap_to_plot.shape) == 2:
        shap.summary_plot(
            shap_to_plot, 
            X_test_imputed, 
            feature_names=list(X_test.columns), 
            show=False
        )
        plt.title(f"SHAP Impact (MPR=1) - {test_year}")
        plt.savefig(os.path.join(base_path, f"SHAP_LGBM_{test_year}.png"), bbox_inches='tight')
    else:
        print(f"Skipping SHAP for {test_year}: Unexpected SHAP shape {shap_to_plot.shape}")
        
    plt.close()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.calibration import calibration_curve

# ---------------------------
# Helpers
# ---------------------------
def calculate_nb(y_true, y_prob, thresh):
    """
    Net Benefit for a given threshold probability.
    Treat as positive (intervene) if predicted risk >= thresh.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return np.nan

    # Guard against thresh=1 (division by zero)
    if thresh >= 1.0:
        return 0.0

    pred_pos = (y_prob >= thresh)
    tp = np.logical_and(pred_pos, y_true == 1).sum()
    fp = np.logical_and(pred_pos, y_true == 0).sum()

    return (tp / n) - (fp / n) * (thresh / (1 - thresh))


# ---------------------------
# DCA thresholds (avoid 0 and 1)
# ---------------------------
thresholds = np.linspace(0.01, 0.70, 70)

# ---------------------------
# Plot setup
# ---------------------------
colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
          "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]

fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)

# ---------------------------
# Loop Through All Years (2014–2023 inclusive)
# NOTE: range end must be 2024 to include 2023
# ---------------------------
for i, test_year in enumerate(range(2014, 2023)):
    # You currently use 2-year lookback for train file naming
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
    test_file  = os.path.join(base_path, f"test_{test_year}_final.csv")

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        continue

    # Data preparation (your existing helper)
    X_train, y_train = prepare_xy(pd.read_csv(train_file))
    X_test,  y_test  = prepare_xy(pd.read_csv(test_file))

    # Fit model (your existing pipeline)
    lgbm_pipeline.fit(X_train, y_train)
    train_proba = lgbm_pipeline.predict_proba(X_train)[:, 1]
    test_proba  = lgbm_pipeline.predict_proba(X_test)[:, 1]

    year_label = f"{test_year-2}-{test_year-1}→{test_year}"

    # ---------------------------
    # ROW 0: CALIBRATION CURVES
    # ---------------------------
    for idx, (y, prob, set_name) in enumerate([(y_train, train_proba, "Train"),
                                               (y_test,  test_proba,  "Test")]):
        frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="uniform")

        axes[0, idx].plot(
            mean_pred, frac_pos,
            marker='s', markersize=4,
            label=year_label,
            color=colors[i % len(colors)],
            alpha=0.8
        )

        if i == 0:
            axes[0, idx].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Ideal")

        axes[0, idx].set_title(f"Calibration Curves - {set_name} Set", fontsize=14, fontweight='bold')
        axes[0, idx].set_xlabel("Mean Predicted Probability", fontsize=12)
        axes[0, idx].set_ylabel("Fraction of Positives", fontsize=12)
        axes[0, idx].set_xlim([-0.02, 1.02])
        axes[0, idx].set_ylim([-0.05, 1.05])

    # ---------------------------
    # ROW 1: DECISION CURVE ANALYSIS
    # Make Treat None visible (don’t let it disappear into the axis spine)
    # ---------------------------
    for idx, (y, prob, set_name) in enumerate([(y_train, train_proba, "Train"),
                                               (y_test,  test_proba,  "Test")]):

        nb_model = [calculate_nb(y, prob, t) for t in thresholds]
        axes[1, idx].plot(
            thresholds, nb_model,
            label=year_label,
            color=colors[i % len(colors)],
            linewidth=1.5
        )

        if i == 0:
            # Treat All baseline (intervene for everyone)
            nb_all = [calculate_nb(y, np.ones(len(y)), t) for t in thresholds]
            axes[1, idx].plot(thresholds, nb_all, 'k--', label="Treat All", alpha=0.5, linewidth=1.6)

            # Treat None baseline (intervene for nobody) = 0 across thresholds
            # Plot explicitly and keep it above the axis spine
            axes[1, idx].axhline(
                y=0.0,
                color='black',
                linestyle='-',
                linewidth=2.0,
                label="Treat None",
                zorder=5
            )
            # Lighten bottom spine so Treat None is visible even at y=0
            axes[1, idx].spines['bottom'].set_color('#BBBBBB')
            axes[1, idx].spines['bottom'].set_linewidth(1.0)

        axes[1, idx].set_title(f"Decision Curve Analysis - {set_name} Set", fontsize=14, fontweight='bold')
        axes[1, idx].set_xlabel("Threshold Probability", fontsize=12)
        axes[1, idx].set_ylabel("Net Benefit", fontsize=12)
        axes[1, idx].set_xlim([0.0, 0.7])

        # Keep your requested scale but make y slightly negative so Treat None is not hidden
        axes[1, idx].set_ylim([-0.02, 0.5])

# ---------------------------
# Final Styling & Legend
# ---------------------------
for ax in axes.flatten():
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.legend(fontsize='small', ncol=2, frameon=True, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(base_path, "Standardized_Temporal_Validation_with_TreatNone.png"), dpi=300)
plt.show()


# **SHAP**

# In[23]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Nature-style configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# 1. Initialize storage for yearly importance
all_yearly_importance = []

# 2. Loop through all temporal splits
for test_year in range(2014, 2023):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
    test_file = os.path.join(base_path, f"test_{test_year}_final.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        continue

    # Load and prepare data
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    X_train, y_train = prepare_xy(train)
    X_test, y_test = prepare_xy(test)

    # Train LightGBM (Identified as the best model with average AUC 0.637)
    lgbm_pipeline = make_pipeline(
        SimpleImputer(strategy="median"), 
        lgb.LGBMClassifier(random_state=42, verbose=-1)
    )
    lgbm_pipeline.fit(X_train, y_train)
    
    # Calculate SHAP values
    imputer = lgbm_pipeline.named_steps['simpleimputer']
    X_test_imputed = imputer.transform(X_test)
    lgbm_clf = lgbm_pipeline.named_steps['lgbmclassifier']
    
    explainer = shap.TreeExplainer(lgbm_clf)
    shap_values = explainer.shap_values(X_test_imputed)

    # Handle SHAP output format (Index 1 for positive class MPR=1)
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    else:
        shap_to_plot = shap_values

    # Calculate Global Importance: Mean Absolute SHAP per feature
    global_importance = np.abs(shap_to_plot).mean(axis=0)
    
    # Store in a temporary DataFrame
    year_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': global_importance,
        'Year': f"{test_year-2}→{test_year}"
    })
    all_yearly_importance.append(year_df)
    print(f"SHAP processed for year: {test_year}")

# 3. Aggregate into a single Pivot Table
full_importance_df = pd.concat(all_yearly_importance)
pivot_df = full_importance_df.pivot(index='Feature', columns='Year', values='Importance')

# Sort features by average importance across the entire decade
pivot_df['Average'] = pivot_df.mean(axis=1)
pivot_df = pivot_df.sort_values('Average', ascending=False).drop(columns='Average')

# 4. Generate the Heatmap
plt.figure(figsize=(14, 10), dpi=300)
# Nature-style: Muted professional colormap
sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f", 
            linewidths=.5, cbar_kws={'label': 'Mean |SHAP Value|'})

plt.title("Temporal Stability of Predictors for HIV Medication Adherence (2014-2023)", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Temporal Test Split (Train Years → Test Year)", fontsize=12)
plt.ylabel("Clinical and Social Determinance of Health Features", fontsize=12)
plt.xticks(rotation=45)

# Nature formatting: Remove top/right spines on the colorbar if needed
plt.tight_layout()
heatmap_path = os.path.join(base_path, "Aggregate_Temporal_SHAP_Heatmap.png")
plt.savefig(heatmap_path)
plt.show()

print(f"Aggregated SHAP Heatmap saved to: {heatmap_path}")


# In[24]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# ---------------------------
# Aggregate & Filter SHAP Heatmap
# ---------------------------

# 1. Combine all yearly importance results into one DataFrame
# (Assuming all_yearly_importance was populated in the previous loop)
full_importance_df = pd.concat(all_yearly_importance)

# 2. Pivot data: Features as rows, Years as columns
pivot_df = full_importance_df.pivot(index='Feature', columns='Year', values='Importance')

# 3. Sort features by their average importance across all years
pivot_df['Average'] = pivot_df.mean(axis=1)
pivot_df = pivot_df.sort_values('Average', ascending=False)

# 4. Filter the list to stop at "Cocaine_Use_Disorder"
# This removes the trailing features that are consistently zero
cutoff_feature = "Cocaine_Use_Disorder"
if cutoff_feature in pivot_df.index:
    cutoff_idx = pivot_df.index.get_loc(cutoff_feature)
    # Slice the dataframe to keep only from the top down to the cutoff
    filtered_pivot_df = pivot_df.iloc[:cutoff_idx + 1].drop(columns='Average')
else:
    print(f"Warning: {cutoff_feature} not found in the feature list. Plotting full list.")
    filtered_pivot_df = pivot_df.drop(columns='Average')

# 5. Plotting: Nature-Style Aggregated Heatmap
plt.figure(figsize=(14, 12), dpi=300)

# Use the YlGnBu colormap for a professional, academic look
sns.heatmap(filtered_pivot_df, 
            annot=True, 
            cmap="YlGnBu", 
            fmt=".3f", 
            linewidths=.5, 
            cbar_kws={'label': 'Mean |SHAP Value|'})

# Formatting titles and labels for manuscript quality
plt.title("Temporal Stability of Predictors for HIV Medication Adherence (2014-2023)", 
          fontsize=16, fontweight='bold', pad=25)
plt.xlabel("Temporal Test Split (Train Years → Test Year)", fontsize=12, fontweight='bold')
plt.ylabel("Clinical and Social Determinants of Health Features", fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')

# Tight layout to ensure no overlapping of labels
plt.tight_layout()

# Save the final high-resolution figure
heatmap_final_path = os.path.join(base_path, "Final_Aggregated_SHAP_Heatmap.png")
plt.savefig(heatmap_final_path, bbox_inches='tight')
plt.show()

print(f"Successfully generated aggregated SHAP heatmap at: {heatmap_final_path}")


# In[25]:


import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

# 1. Calibration Curve
prob_pos = lgbm_model.predict_proba(X_test)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

plt.figure(figsize=(10, 5))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="LightGBM")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.ylabel("Fraction of positives")
plt.xlabel("Mean predicted probability")
plt.title("Calibration Plot (Reliability Curve)")
plt.legend()
plt.show()

# 2. Decision Curve Analysis (Simplified)
def calculate_net_benefit(thresh, probs, labels):
    y_pred = (probs >= thresh).astype(int)
    tp = ((y_pred == 1) & (labels == 1)).sum()
    fp = ((y_pred == 1) & (labels == 0)).sum()
    n = len(labels)
    if tp == 0 and fp == 0: return 0
    net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
    return net_benefit

thresholds = [i/100 for i in range(1, 99)]
nb_model = [calculate_net_benefit(t, prob_pos, y_test) for t in thresholds]
nb_all = [calculate_net_benefit(t, np.ones(len(y_test)), y_test) for t in thresholds]

plt.plot(thresholds, nb_model, color='blue', label='LightGBM')
plt.plot(thresholds, nb_all, color='black', linestyle='--', label='Treat All')
plt.axhline(y=0, color='red', linestyle='-', label='Treat None')
plt.ylim(-0.05, max(nb_model) + 0.1)
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis')
plt.legend()
plt.show()


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt

# Load prepared data
df = pd.read_excel("yearly_model_classification_metrics.xlsx", sheet_name='positive_class')
df.head()


# In[28]:


# Create LSTM dataframe from user-provided table
lstm_rows = [
    ("LSTM","2012-2013→2014",0.866,0.826,0.845,0.896),
    ("LSTM","2013-2014→2015",0.831,0.711,0.767,0.809),
    ("LSTM","2014-2015→2016",0.692,0.677,0.685,0.758),
    ("LSTM","2015-2016→2017",0.758,0.701,0.729,0.787),
    ("LSTM","2016-2017→2018",0.631,0.612,0.621,0.722),
    ("LSTM","2017-2018→2019",0.595,0.500,0.543,0.752),
    ("LSTM","2018-2019→2020",0.861,0.392,0.539,0.677),
    ("LSTM","2019-2020→2021",0.714,0.395,0.508,0.669),
    ("LSTM","2020-2021→2022",0.705,0.696,0.701,0.811),
    ("LSTM","2021-2022→2023",0.405,0.500,0.448,0.934),
]
df_lstm = pd.DataFrame(lstm_rows, columns=["Model","Year","Precision (class=1)","Recall (class=1)","F1 (class=1)","Accuracy"])
df_lstm["Support (class=1)"] = pd.NA
df_lstm["Support_Total"] = pd.NA

# Combine
df_all = pd.concat([df, df_lstm], ignore_index=True)

# Aggregate across years: mean metrics (positive class emphasis)
summary = (
    df_all.groupby("Model", dropna=False)
      .agg({
          "Recall (class=1)": "mean",
          "F1 (class=1)": "mean",
          "Precision (class=1)": "mean",
          "Accuracy": "mean"
      })
      .reset_index()
)

# Order by mean recall descending (since you want sensitivity higher)
summary = summary.sort_values("Recall (class=1)", ascending=False)

# ---- Plot 1: Mean Recall (Sensitivity) for Class 1 ----
plt.figure()
plt.bar(summary["Model"], summary["Recall (class=1)"])
plt.title("Mean Sensitivity (Recall) for Medication Adherence (Class 1) — incl. LSTM")
plt.ylabel("Recall (Class 1)")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()




# In[29]:


# ---- Plot 2: Mean F1-score for Class 1 ----
plt.figure()
plt.bar(summary["Model"], summary["F1 (class=1)"])
plt.title("Mean F1-score for Medication Adherence (Class 1) — incl. LSTM")
plt.ylabel("F1-score (Class 1)")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



# In[35]:


# ---- Plot 3: Precision vs Recall (Class 1) ----
plt.figure()
plt.scatter(summary["Precision (class=1)"], summary["Recall (class=1)"])
for i, model in enumerate(summary["Model"]):
    plt.text(summary["Precision (class=1)"].iloc[i],
             summary["Recall (class=1)"].iloc[i],
             model)
plt.xlabel("Precision (Class 1)")
plt.ylabel("Recall (Class 1)")
plt.title("Precision vs Sensitivity Trade-off (Class 1)")
plt.tight_layout()
plt.show()



# In[33]:


# ---- Plot 4: Year-by-year Recall for each model (trend stability) ----
# Sort years by target year for plotting lines
def year_key(s):
    tgt = int(str(s).split("→")[1])
    src = str(s).split("→")[0]
    return (tgt, src)

df_all_plot = df_all.copy()
df_all_plot["Year_key"] = df_all_plot["Year"].map(year_key)
df_all_plot = df_all_plot.sort_values("Year_key")

pivot_recall = df_all_plot.pivot_table(index="Year", columns="Model", values="Recall (class=1)", aggfunc="mean")
plt.figure()
for col in pivot_recall.columns:
    plt.plot(pivot_recall.index, pivot_recall[col], marker="o", label=col)
plt.title("Recall (Sensitivity) Over Time for Class 1")
plt.ylabel("Recall (Class 1)")
plt.xlabel("Year window")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.legend()
plt.show()



# In[32]:


# Save combined table for your records
combined_path = "yearly_model_metrics_positive_class_including_LSTM.csv"
df_all.to_csv(combined_path, index=False)

summary, combined_path


# In[34]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data extraction from your logs (Class 1 Metrics: Recall & F1)
data = {
    'Year': ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
    
    # Sensitivity (Recall Class 1)
    'LR_Recall':   [0.44, 0.27, 0.39, 0.25, 0.24, 0.30, 0.10, 0.22, 0.46, 0.20],
    'DT_Recall':   [0.55, 0.52, 0.42, 0.61, 0.46, 0.48, 0.48, 0.53, 0.44, 0.33],
    'RF_Recall':   [0.58, 0.40, 0.58, 0.57, 0.43, 0.25, 0.23, 0.38, 0.68, 0.60],
    'XGB_Recall':  [0.59, 0.51, 0.49, 0.63, 0.39, 0.43, 0.37, 0.49, 0.63, 0.60],
    'LGBM_Recall': [0.55, 0.52, 0.51, 0.67, 0.40, 0.48, 0.32, 0.47, 0.59, 0.67],
    'LSTM_Recall': [0.826, 0.711, 0.677, 0.701, 0.612, 0.500, 0.392, 0.395, 0.696, 0.500],

    # F1-Score (Class 1)
    'LR_F1':       [0.45, 0.34, 0.44, 0.32, 0.32, 0.31, 0.16, 0.32, 0.48, 0.21],
    'DT_F1':       [0.43, 0.51, 0.45, 0.57, 0.42, 0.41, 0.50, 0.50, 0.36, 0.27],
    'RF_F1':       [0.56, 0.48, 0.55, 0.58, 0.46, 0.29, 0.32, 0.47, 0.58, 0.42],
    'XGB_F1':      [0.56, 0.53, 0.50, 0.61, 0.43, 0.39, 0.45, 0.56, 0.57, 0.44],
    'LGBM_F1':     [0.55, 0.54, 0.53, 0.63, 0.44, 0.44, 0.41, 0.55, 0.59, 0.49],
    'LSTM_F1':     [0.845, 0.767, 0.685, 0.729, 0.621, 0.543, 0.539, 0.508, 0.701, 0.448]
}

df = pd.DataFrame(data)

# Melt dataframe for seaborn plotting
df_recall = df.melt(id_vars=['Year'], value_vars=[c for c in df.columns if 'Recall' in c], 
                    var_name='Model', value_name='Sensitivity (Recall)')
df_recall['Model'] = df_recall['Model'].str.replace('_Recall', '')

df_f1 = df.melt(id_vars=['Year'], value_vars=[c for c in df.columns if 'F1' in c], 
                var_name='Model', value_name='F1 Score')
df_f1['Model'] = df_f1['Model'].str.replace('_F1', '')

# --- Plotting ---
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# 1. Sensitivity Trend Line Plot
sns.lineplot(data=df_recall, x='Year', y='Sensitivity (Recall)', hue='Model', marker='o', linewidth=2.5, ax=axes[0])
axes[0].set_title('Model Sensitivity (Recall for Class 1) Over Years', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Sensitivity (Recall)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

# 2. F1 Score Trend Line Plot
sns.lineplot(data=df_f1, x='Year', y='F1 Score', hue='Model', marker='s', linewidth=2.5, ax=axes[1])
axes[1].set_title('Model F1-Score (Class 1) Over Years', fontsize=16, fontweight='bold')
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

# 3. Average Performance Comparison Bar Chart
avg_recall = df_recall.groupby('Model')['Sensitivity (Recall)'].mean().sort_values()
colors = sns.color_palette("viridis", len(avg_recall))
avg_recall.plot(kind='barh', ax=axes[2], color=colors)
axes[2].set_title('Average Sensitivity (Recall Class 1) Across All Years', fontsize=16, fontweight='bold')
axes[2].set_xlabel('Average Sensitivity', fontsize=12)
axes[2].set_xlim(0, 1.0)

for i, v in enumerate(avg_recall):
    axes[2].text(v + 0.01, i, f'{v:.3f}', color='black', fontweight='bold', va='center')

plt.tight_layout()
plt.show()


# In[36]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Data Setup (Same as before) ---
data = {
    'Year': ['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'],
    # Recall (Sensitivity)
    'LR_Recall':   [0.44, 0.27, 0.39, 0.25, 0.24, 0.30, 0.10, 0.22, 0.46, 0.20],
    'DT_Recall':   [0.55, 0.52, 0.42, 0.61, 0.46, 0.48, 0.48, 0.53, 0.44, 0.33],
    'RF_Recall':   [0.58, 0.40, 0.58, 0.57, 0.43, 0.25, 0.23, 0.38, 0.68, 0.60],
    'XGB_Recall':  [0.59, 0.51, 0.49, 0.63, 0.39, 0.43, 0.37, 0.49, 0.63, 0.60],
    'LGBM_Recall': [0.55, 0.52, 0.51, 0.67, 0.40, 0.48, 0.32, 0.47, 0.59, 0.67],
    'LSTM_Recall': [0.826, 0.711, 0.677, 0.701, 0.612, 0.500, 0.392, 0.395, 0.696, 0.500],
    
    # Precision (Extracted from your logs for the Scatter Plot)
    'LR_Precision':   [0.45, 0.47, 0.50, 0.45, 0.47, 0.32, 0.44, 0.55, 0.51, 0.21],
    'DT_Precision':   [0.35, 0.50, 0.48, 0.54, 0.38, 0.36, 0.52, 0.48, 0.30, 0.22],
    'RF_Precision':   [0.53, 0.59, 0.52, 0.58, 0.50, 0.33, 0.56, 0.62, 0.51, 0.32],
    'XGB_Precision':  [0.53, 0.55, 0.51, 0.60, 0.47, 0.36, 0.59, 0.66, 0.52, 0.35],
    'LGBM_Precision': [0.55, 0.57, 0.56, 0.59, 0.48, 0.41, 0.58, 0.67, 0.59, 0.39],
    'LSTM_Precision': [0.866, 0.831, 0.692, 0.758, 0.631, 0.595, 0.861, 0.714, 0.705, 0.405] 
}
df = pd.DataFrame(data)

# Reshape for Heatmap and Boxplot (Focus on Recall)
df_recall = df.melt(id_vars=['Year'], value_vars=[c for c in df.columns if 'Recall' in c], 
                    var_name='Model', value_name='Sensitivity')
df_recall['Model'] = df_recall['Model'].str.replace('_Recall', '')

# Reshape for Scatter Plot (Need Precision AND Recall aligned)
# We create a list of dataframes and concat them
frames = []
models = ['LR', 'DT', 'RF', 'XGB', 'LGBM', 'LSTM']
for m in models:
    temp = df[['Year', f'{m}_Recall', f'{m}_Precision']].copy()
    temp.columns = ['Year', 'Recall', 'Precision']
    temp['Model'] = m
    frames.append(temp)
df_scatter = pd.concat(frames)

# --- Plotting ---
fig = plt.figure(figsize=(18, 12))
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.3)

# 1. Heatmap (Top Left)
ax1 = fig.add_subplot(grid[0, 0])
# Pivot data for heatmap format
heatmap_data = df_recall.pivot(index='Model', columns='Year', values='Sensitivity')
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax1, cbar_kws={'label': 'Sensitivity'})
ax1.set_title('Temporal Stability Heatmap (Sensitivity)', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

# 2. Box Plot (Top Right)
ax2 = fig.add_subplot(grid[0, 1])
sns.boxplot(data=df_recall, x='Model', y='Sensitivity', palette="Set2", ax=ax2)
sns.stripplot(data=df_recall, x='Model', y='Sensitivity', color='black', alpha=0.5, ax=ax2) # Add dots to show actual years
ax2.set_title('Model Reliability & Variance (Box Plot)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Sensitivity Distribution')

# 3. Precision vs Recall Scatter (Bottom - Spanning width)
ax3 = fig.add_subplot(grid[1, :])
sns.scatterplot(data=df_scatter, x='Recall', y='Precision', hue='Model', style='Model', s=150, alpha=0.8, ax=ax3, palette='deep')

# Annotate the 'Best' years for LSTM and LGBM to highlight peaks
best_lstm = df_scatter[df_scatter['Model']=='LSTM'].sort_values('Recall', ascending=False).iloc[0]
ax3.annotate(f"LSTM Peak", (best_lstm['Recall'], best_lstm['Precision']), xytext=(best_lstm['Recall']-0.05, best_lstm['Precision']+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

best_lgbm = df_scatter[df_scatter['Model']=='LGBM'].sort_values('Recall', ascending=False).iloc[0]
ax3.annotate(f"LGBM Peak", (best_lgbm['Recall'], best_lgbm['Precision']), xytext=(best_lgbm['Recall']-0.05, best_lgbm['Precision']-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05))

ax3.set_title('Precision vs. Recall Trade-off (All Years)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Recall (Sensitivity) - Ability to find Adherent Patients', fontsize=12)
ax3.set_ylabel('Precision - Quality of Positive Predictions', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.tight_layout()
plt.show()


# In[17]:


# ---------------------------
# Sklearn models (+ SHAP for LightGBM)
# ---------------------------
import shap
from collections import defaultdict

models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier()),
    "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier()),
    "XGBoost": make_pipeline(SimpleImputer(strategy="median"), xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    "LightGBM": make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier())
}

# Where to write SHAP assets
shap_dir = os.path.join(base_path, "shap_lightgbm")
os.makedirs(shap_dir, exist_ok=True)

# Collect per-year SHAP importances to aggregate later
lightgbm_yearly_importance = []

def _save_shap_fig(save_path):
    # SHAP uses its own plotting; just make sure we save & clear cleanly
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

for model_name, model in models.items():
    plt.figure(figsize=(12, 8))
    any_plotted = False

    for test_year in range(2014, 2024):
        train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_merged.csv")
        test_file = os.path.join(base_path, f"test_{test_year}_merged.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files for {test_year}, skipping: {train_file} | {test_file}")
            continue

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        del train['PDC']
        del test['PDC']
        del train['AGE_at_index']
        del test['AGE_at_index']
        # Prepare data
        X_train, y_train = prepare_xy(train)
        X_test, y_test = prepare_xy(test)

        # Class sanity
        if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            print(f"Skipping {test_year}: not enough label classes.")
            continue

        # Fit
        model.fit(X_train, y_train)

        # Proba or scores
        final_est = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(final_est, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(final_est, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = model.predict(X_test).astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})")
        any_plotted = True

        # Report
        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n{model_name} Classification Report for {test_year-2}-{test_year-1}→{test_year}")
        print(classification_report(y_test, y_pred))

        # ---------------------------
        # SHAP (only for LightGBM)
        # ---------------------------
        if model_name == "LightGBM":
            # Grab fitted steps
            imp = model.named_steps["simpleimputer"]
            lgb_est = model.named_steps["lgbmclassifier"]

            # Transform test features with the same imputer
            X_test_imp = pd.DataFrame(
                imp.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            # For speed, sample up to N rows
            shap_sample_n = min(2000, len(X_test_imp))
            X_shap = X_test_imp.sample(n=shap_sample_n, random_state=42) if shap_sample_n < len(X_test_imp) else X_test_imp

            # Compute SHAP
            explainer = shap.TreeExplainer(lgb_est)
            shap_values = explainer.shap_values(X_shap)

            # Some versions return a list for multiclass; handle binary
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class

            # Per-year importance table
            mean_abs = np.abs(shap_values).mean(axis=0)
            year_imp_df = pd.DataFrame({
                "feature": X_shap.columns,
                "mean_abs_shap": mean_abs,
                "year": f"{test_year-2}-{test_year-1}→{test_year}"
            }).sort_values("mean_abs_shap", ascending=False)
            lightgbm_yearly_importance.append(year_imp_df)

            # Beeswarm
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap, show=False)
            _save_shap_fig(os.path.join(
                shap_dir, f"shap_beeswarm_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))
            print(f"Saved SHAP beeswarm for {test_year-2}-{test_year-1}→{test_year}")

            # Bar plot (mean |SHAP|)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            _save_shap_fig(os.path.join(
                shap_dir, f"shap_bar_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))
            print(f"Saved SHAP bar for {test_year-2}-{test_year-1}→{test_year}")

            # Optional: Dependence plot for top feature of the year
            top_feat = year_imp_df.iloc[0]["feature"]
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(top_feat, shap_values, X_shap, show=False)
            _save_shap_fig(os.path.join(
                shap_dir, f"shap_dependence_{top_feat}_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))
            print(f"Saved SHAP dependence ({top_feat}) for {test_year-2}-{test_year-1}→{test_year}")

    if any_plotted:
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - Yearly Temporal ROC Curves (Target=MPR)")
        plt.legend()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        roc_path = os.path.join(base_path, f"yearly_temporal_ROC_{model_name.replace(' ', '_')}.png")
        plt.savefig(roc_path)
        print(f"Saved ROC curve for {model_name} at {roc_path}")
    else:
        plt.close()
        print(f"No valid splits for {model_name}; ROC not saved.")

# ---------------------------
# Aggregate LightGBM SHAP across years
# ---------------------------
if lightgbm_yearly_importance:
    all_imp = pd.concat(lightgbm_yearly_importance, ignore_index=True)
    all_imp.to_csv(os.path.join(shap_dir, "lightgbm_shap_per_year.csv"), index=False)

    agg = (all_imp.groupby("feature")["mean_abs_shap"]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index())
    agg.to_csv(os.path.join(shap_dir, "lightgbm_shap_all_years_mean_abs.csv"), index=False)

    # Quick global bar (top 20) using SHAP's bar util
    topk = 20
    top_feats = agg.head(topk)["feature"].tolist()
    # Recompute SHAP on the last available year's imputed X for those features (for a pretty bar)
    # If you want a cleaner global figure, you can also rebuild on a pooled sample.
    print("Saved per-year and aggregated LightGBM SHAP importances.")


# In[10]:


# ============================================
# Classic ML (sklearn) + LSTM, target = MPR
# + LightGBM SHAP + Fairness/Drift + Clinical Plots
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

import seaborn as sns
sns.set(style="whitegrid", rc={"figure.dpi": 150})

import xgboost as xgb
import lightgbm as lgb
import shap

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from scipy.stats import ks_2samp

# ---------------------------
# Paths
# ---------------------------
base_path = r"D:\HIV Prevention\cohort\Jan_2026\Models"
os.makedirs(base_path, exist_ok=True)
shap_dir = os.path.join(base_path, "shap_lightgbm")
os.makedirs(shap_dir, exist_ok=True)
fig_dir = os.path.join(base_path, "figs_insights")
os.makedirs(fig_dir, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def normalize_sex_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """If SEX is object, normalize to M=1, F=0 with nullable Int64."""
    if 'SEX' in df.columns and df['SEX'].dtype == object:
        df['SEX'] = (
            df['SEX'].astype(str).str.strip().str.upper()
              .replace({'MALE': 'M', 'FEMALE': 'F'})
              .map({'M': 1, 'F': 0})
              .astype('Int64')
        )
    return df

def prepare_xy(df: pd.DataFrame):
    """Prepare numeric X and binary y=MPR (already 0/1)."""
    df = normalize_sex_if_needed(df)
    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset. Expected binary MPR target.")
    y = df['MPR'].astype(int)
    X = df.drop(columns=[c for c in ['MPR', 'ID', 'patient_id'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def prepare_xy_numeric(df: pd.DataFrame):
    df = normalize_sex_if_needed(df)
    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset.")
    y = df['MPR'].astype(int).values
    X = df.drop(columns=[c for c in ['MPR', 'ID', 'patient_id'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def _save_current_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------------------------
# Sklearn models (+ SHAP for LightGBM)
# ---------------------------
models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000)),
    "Decision Tree":       make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier()),
    "Random Forest":       make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier()),
    "XGBoost":             make_pipeline(SimpleImputer(strategy="median"), xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    "LightGBM":            make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier(random_state=42))
}

# Collect per-year SHAP importances for LightGBM
lightgbm_yearly_importance = []
# Collect per-year predictions/probabilities for LightGBM (for calibration & demographics)
lightgbm_yearly_pred_rows = []

def plot_roc_finalize(model_name, plotted, save_name):
    if plotted:
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - Yearly Temporal ROC Curves (Target=MPR)")
        plt.legend()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        _save_current_fig(save_name)
        print(f"Saved ROC curve for {model_name} at {save_name}")
    else:
        plt.close()
        print(f"No valid splits for {model_name}; ROC not saved.")

for model_name, model in models.items():
    plt.figure(figsize=(12, 8))
    any_plotted = False

    for test_year in range(2014, 2024):
        train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_merged.csv")
        test_file  = os.path.join(base_path, f"test_{test_year}_merged.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files for {test_year}, skipping: {train_file} | {test_file}")
            continue

        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        del train['PDC']
        del test['PDC']
        del train['AGE_at_index']
        del test['AGE_at_index']
        # Keep copies to pull group columns later
        test_raw = test.copy()

        # Prepare data
        X_train, y_train = prepare_xy(train)
        X_test,  y_test  = prepare_xy(test)

        # Class sanity
        if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            print(f"Skipping {test_year}: not enough label classes.")
            continue

        # Fit
        model.fit(X_train, y_train)

        # Proba or scores
        final_est = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(final_est, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(final_est, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = model.predict(X_test).astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})")
        any_plotted = True

        # Report
        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n{model_name} Classification Report for {test_year-2}-{test_year-1}→{test_year}")
        print(classification_report(y_test, y_pred))

        # ---------------------------
        # SHAP + store preds (LightGBM only)
        # ---------------------------
        if model_name == "LightGBM":
            # Pull imputer & model
            imp     = model.named_steps["simpleimputer"]
            lgb_est = model.named_steps["lgbmclassifier"]

            # Imputed test features (keep original column names)
            X_test_imp = pd.DataFrame(
                imp.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            # Save per-year preds for calibration/demographics
            # Try to keep a few group columns if present
            year_label = f"{test_year-2}-{test_year-1}→{test_year}"
            perf_df = pd.DataFrame({
                "year_split": year_label,
                "y_true": y_test.values,
                "y_prob": y_proba
            }, index=X_test_imp.index)

            # Attach known potential group columns if present in test_raw
            for col in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"]:
                if col in test_raw.columns:
                    perf_df[col] = test_raw.loc[perf_df.index, col].values
            lightgbm_yearly_pred_rows.append(perf_df)

            # SHAP (sample for speed)
            shap_sample_n = min(2000, len(X_test_imp))
            X_shap = X_test_imp.sample(n=shap_sample_n, random_state=42) if shap_sample_n < len(X_test_imp) else X_test_imp

            explainer = shap.TreeExplainer(lgb_est)
            shap_values = explainer.shap_values(X_shap)
            if isinstance(shap_values, list):  # binary -> take positive class
                shap_values = shap_values[1]

            # Per-year SHAP importance
            mean_abs = np.abs(shap_values).mean(axis=0)
            year_imp_df = pd.DataFrame({
                "feature": X_shap.columns,
                "mean_abs_shap": mean_abs,
                "year": year_label
            }).sort_values("mean_abs_shap", ascending=False)
            lightgbm_yearly_importance.append(year_imp_df)

            # Beeswarm
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap, show=False)
            _save_current_fig(os.path.join(
                shap_dir, f"shap_beeswarm_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))
            # Bar plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            _save_current_fig(os.path.join(
                shap_dir, f"shap_bar_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))
            # Dependence for top feature
            top_feat = year_imp_df.iloc[0]["feature"]
            plt.figure(figsize=(7, 5))
            shap.dependence_plot(top_feat, shap_values, X_shap, show=False)
            _save_current_fig(os.path.join(
                shap_dir, f"shap_dependence_{top_feat}_{test_year-2}-{test_year-1}_to_{test_year}.png"
            ))

    # finalize ROC for this model
    roc_path = os.path.join(base_path, f"yearly_temporal_ROC_{model_name.replace(' ', '_')}.png")
    plot_roc_finalize(model_name, any_plotted, roc_path)

# ---------------------------
# Aggregate LightGBM SHAP across years
# ---------------------------
if lightgbm_yearly_importance:
    all_imp = pd.concat(lightgbm_yearly_importance, ignore_index=True)
    all_imp.to_csv(os.path.join(shap_dir, "lightgbm_shap_per_year.csv"), index=False)

    agg = (all_imp.groupby("feature")["mean_abs_shap"]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index())
    agg.to_csv(os.path.join(shap_dir, "lightgbm_shap_all_years_mean_abs.csv"), index=False)
    print("Saved per-year and aggregated LightGBM SHAP importances.")

# ---------------------------
# Fairness & Calibration (LightGBM)
# ---------------------------
if lightgbm_yearly_pred_rows:
    preds_all = pd.concat(lightgbm_yearly_pred_rows, ignore_index=True)

    # 1) Calibration plot per year
    years = preds_all["year_split"].unique()
    for yr in years:
        sub = preds_all[preds_all["year_split"] == yr]
        if sub["y_true"].nunique() < 2:
            print(f"[calibration] Skipping {yr}: only one class present.")
            continue
        prob_true, prob_pred = calibration_curve(sub["y_true"], sub["y_prob"], n_bins=10, strategy="quantile")
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label=yr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed probability')
        plt.title(f'Calibration: LightGBM ({yr})')
        plt.legend()
        _save_current_fig(os.path.join(fig_dir, f"calibration_{yr.replace('→','to')}.png"))

    # 2) Demographic performance (AUC & Accuracy) for available group cols
    group_cols = [c for c in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"] if c in preds_all.columns]
    for gcol in group_cols:
        # Plot per year to see stability
        for yr in years:
            sub = preds_all[preds_all["year_split"] == yr].dropna(subset=[gcol])
            # need at least 2 classes total
            if sub["y_true"].nunique() < 2:
                print(f"[demographics] {gcol} {yr}: only one class y_true; skipping.")
                continue
            rows = []
            for grp, df_g in sub.groupby(gcol):
                try:
                    auc_g = roc_auc_score(df_g["y_true"], df_g["y_prob"]) if df_g["y_true"].nunique() == 2 else np.nan
                    acc_g = accuracy_score(df_g["y_true"], (df_g["y_prob"] >= 0.5).astype(int))
                    rows.append({"Group": str(grp), "AUC": auc_g, "Accuracy": acc_g})
                except Exception as e:
                    print(f"[demographics] {gcol}={grp} {yr} error: {e}")
            if not rows:
                continue
            met_df = pd.DataFrame(rows)
            plt.figure(figsize=(8, 5))
            met_df.set_index("Group")[["AUC", "Accuracy"]].plot(kind="bar")
            plt.title(f"LightGBM Performance by {gcol} — {yr}")
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.grid(axis="y")
            _save_current_fig(os.path.join(fig_dir, f"demographic_perf_{gcol}_{yr.replace('→','to')}.png"))

# ---------------------------
# Feature Drift (early vs late)
# ---------------------------
EARLY_TEST_YEARS = [2014, 2015, 2016]   # → train windows: 2012–2015
LATE_TEST_YEARS  = [2021, 2022, 2023]   # → train windows: 2019–2023

def load_concat_train_features(test_years, base_path):
    frames = []
    for ty in test_years:
        train_file = os.path.join(base_path, f"train_{ty-2}-{ty-1}_merged.csv")
        if not os.path.exists(train_file):
            print(f"[drift] Missing train file for {ty}: {train_file}")
            continue
        df = pd.read_csv(train_file)
        X, _ = prepare_xy(df)
        frames.append(X)
    if not frames:
        raise RuntimeError("[drift] No train files found for selected years.")
    return pd.concat(frames, axis=0, ignore_index=True)

try:
    X_early_raw = load_concat_train_features(EARLY_TEST_YEARS, base_path)
    X_late_raw  = load_concat_train_features(LATE_TEST_YEARS,  base_path)

    common_cols = sorted(list(set(X_early_raw.columns) & set(X_late_raw.columns)))
    X_early_raw = X_early_raw[common_cols].copy()
    X_late_raw  = X_late_raw[common_cols].copy()

    early_imputer = SimpleImputer(strategy="median")
    X_train_early = pd.DataFrame(early_imputer.fit_transform(X_early_raw), columns=common_cols)
    X_train_late  = pd.DataFrame(early_imputer.transform(X_late_raw), columns=common_cols)

    # PSI + KS ranking
    def population_stability_index(a, b, bins=10, eps=1e-9):
        a = np.asarray(a).astype(float)
        b = np.asarray(b).astype(float)
        quantiles = np.linspace(0, 1, bins+1)
        edges = np.quantile(a, quantiles)
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0
        a_counts, _ = np.histogram(a, bins=edges)
        b_counts, _ = np.histogram(b, bins=edges)
        a_pct = a_counts / (a_counts.sum() + eps)
        b_pct = b_counts / (b_counts.sum() + eps)
        psi = np.sum((a_pct - b_pct) * np.log((a_pct + eps) / (b_pct + eps)))
        return float(psi)

    drift_rows = []
    for col in common_cols:
        psi = population_stability_index(X_train_early[col].values, X_train_late[col].values, bins=10)
        ks  = ks_2samp(X_train_early[col].values, X_train_late[col].values).statistic
        drift_rows.append({"feature": col, "PSI": psi, "KS": ks})

    drift_df = pd.DataFrame(drift_rows).sort_values(["PSI", "KS"], ascending=False)
    drift_csv = os.path.join(shap_dir, "feature_drift_rank.csv")
    drift_df.to_csv(drift_csv, index=False)
    print(f"[drift] Saved drift ranking to {drift_csv}")

    # Plot top-5 drifting features
    topk = 5
    for feat in drift_df.head(topk)["feature"]:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(X_train_early[feat], label="Early (2014–2016)", fill=True)
        sns.kdeplot(X_train_late[feat],  label="Late (2021–2023)", fill=True)
        plt.title(f"Feature Drift: {feat}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        plt.legend()
        _save_current_fig(os.path.join(fig_dir, f"feature_drift_{feat}.png"))

except Exception as e:
    print(f"[drift] Skipped drift analysis due to: {e}")

# ---------------------------
# Top-10 risk factors (per year) from SHAP CSV
# ---------------------------
shap_per_year_csv = os.path.join(shap_dir, "lightgbm_shap_per_year.csv")
if os.path.exists(shap_per_year_csv):
    shap_df = pd.read_csv(shap_per_year_csv)
    for year, sub in shap_df.groupby("year"):
        top_feats = sub.sort_values("mean_abs_shap", ascending=False).head(10)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=top_feats, x="mean_abs_shap", y="feature", palette="Reds_r")
        plt.title(f"Top-10 Risk Factors — {year}")
        plt.xlabel("Mean |SHAP|")
        _save_current_fig(os.path.join(fig_dir, f"top10_shap_{year.replace('→','to')}.png"))

# ---------------------------
# Patient risk stratification scatter (choose a feature)
# ---------------------------
if lightgbm_yearly_pred_rows:
    preds_all = pd.concat(lightgbm_yearly_pred_rows, ignore_index=True)
    # Choose a likely strong feature; fallback to most important by SHAP-all-years
    feature_for_scatter = None
    candidate_cols = ["NUM_REFILLS", "PRIOR_MPR", "NUM_VISITS", "AGE"]  # edit to your schema
    for c in candidate_cols:
        if c in common_cols:  # from drift common set if available
            feature_for_scatter = c
            break

    # If not found, try from SHAP importance table
    if feature_for_scatter is None and os.path.exists(os.path.join(shap_dir, "lightgbm_shap_all_years_mean_abs.csv")):
        agg_imp = pd.read_csv(os.path.join(shap_dir, "lightgbm_shap_all_years_mean_abs.csv"))
        if not agg_imp.empty:
            feature_for_scatter = agg_imp.iloc[0]["feature"]

    # Make the scatter for the latest year if that feature is present in X_test (need to rebuild last X_test_imp)
    latest_year = max(preds_all["year_split"].unique(), key=lambda s: s.split("→")[-1])
    latest_df = preds_all[preds_all["year_split"] == latest_year].copy()

    if feature_for_scatter and 'y_prob' in latest_df.columns:
        # We need the actual feature values for that year; rebuild from files
        ty = int(latest_year.split("→")[-1])
        test_file = os.path.join(base_path, f"test_{ty}_merged.csv")
        if os.path.exists(test_file):
            test_raw = pd.read_csv(test_file)
            if feature_for_scatter in test_raw.columns:
                latest_df[feature_for_scatter] = test_raw.loc[latest_df.index, feature_for_scatter].values
                plt.figure(figsize=(7, 5))
                sns.scatterplot(data=latest_df, x=feature_for_scatter, y="y_prob", hue="y_true", alpha=0.7)
                plt.axhline(0.5, ls="--", color="gray")
                plt.title(f"Risk Stratification — {feature_for_scatter} ({latest_year})")
                plt.xlabel(feature_for_scatter)
                plt.ylabel("Predicted Probability (MPR=1)")
                _save_current_fig(os.path.join(fig_dir, f"risk_scatter_{feature_for_scatter}_{latest_year.replace('→','to')}.png"))
            else:
                print(f"[risk_scatter] Feature {feature_for_scatter} not in test {ty} dataset.")
    else:
        print("[risk_scatter] No suitable feature found for scatter plot.")

# ---------------------------
# Single-patient explanation (waterfall/force plot)
# ---------------------------
# Build on the latest year LightGBM model for a specific patient
try:
    latest_year = ty if 'ty' in locals() else 2023
    # Refit a LightGBM on that split to get explainer + imputed X for a single case
    train_file = os.path.join(base_path, f"train_{latest_year-2}-{latest_year-1}_merged.csv")
    test_file  = os.path.join(base_path, f"test_{latest_year}_merged.csv")
    if os.path.exists(train_file) and os.path.exists(test_file):
        train = pd.read_csv(train_file); test = pd.read_csv(test_file)
        X_train, y_train = prepare_xy(train)
        X_test,  y_test  = prepare_xy(test)

        pipe = make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier(random_state=42))
        pipe.fit(X_train, y_train)
        imp = pipe.named_steps["simpleimputer"]
        lgb_est = pipe.named_steps["lgbmclassifier"]
        X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

        explainer = shap.TreeExplainer(lgb_est)
        shap_vals = explainer.shap_values(X_test_imp)
        expected_value = explainer.expected_value

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

        # pick a positive or negative case to show
        idx0 = X_test_imp.index[0]
        try:
            # Newer SHAP: waterfall expects Explanation object
            exp = shap.Explanation(values=shap_vals[idx0, :],
                                   base_values=expected_value,
                                   data=X_test_imp.iloc[idx0, :].values,
                                   feature_names=X_test_imp.columns.tolist())
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(exp, show=False, max_display=20)
            _save_current_fig(os.path.join(fig_dir, f"waterfall_patient_{idx0}_{latest_year}.png"))
        except Exception:
            # Fallback to force plot (matplotlib)
            shap.force_plot(expected_value, shap_vals[idx0, :], X_test_imp.iloc[idx0, :], matplotlib=True)
            _save_current_fig(os.path.join(fig_dir, f"force_patient_{idx0}_{latest_year}.png"))
    else:
        print("[waterfall] Latest year train/test not found; skipping.")
except Exception as e:
    print(f"[waterfall] Skipped single-patient plot due to: {e}")

# ---------------------------
# LSTM (sequence_len = 1) — unchanged from your base with saving ROC & metrics
# ---------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).view(-1)

metrics_table = []
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10.colors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, test_year in enumerate(range(2014, 2024)):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_merged.csv")
    test_file  = os.path.join(base_path, f"test_{test_year}_merged.csv")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Files missing for {test_year}, skipping.")
        continue

    train = pd.read_csv(train_file)
    test  = pd.read_csv(test_file)

    # X, y
    X_train, y_train = prepare_xy_numeric(train)
    X_test,  y_test  = prepare_xy_numeric(test)

    # Class check
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(f"Skipping {test_year}: not enough classes.")
        continue

    # Impute (train medians for both)
    train_medians = pd.DataFrame(X_train).median()
    X_train = pd.DataFrame(X_train).fillna(train_medians)
    X_test  = pd.DataFrame(X_test).fillna(train_medians)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # LSTM expects (N, T, F); we set T = 1
    X_train_seq = np.expand_dims(X_train_scaled, axis=1)
    X_test_seq  = np.expand_dims(X_test_scaled,  axis=1)

    # DataLoaders
    train_dataset = LSTMDataset(X_train_seq, y_train)
    test_dataset  = LSTMDataset(X_test_seq,  y_test)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32)

    # Model
    model = LSTMModel(input_dim=X_train_seq.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_probs, y_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            y_probs.extend(preds.cpu().numpy())
            y_true.extend(yb.numpy())

    y_probs = np.array(y_probs)
    y_true  = np.array(y_true)
    y_pred  = (y_probs >= 0.5).astype(int)

    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    metrics_table.append({
        'Model': 'LSTM',
        'Year': f"{test_year-2}-{test_year-1}→{test_year}",
        'Precision': round(precision, 3),
        'Recall (Sensitivity)': round(sensitivity, 3),
        'Specificity': round(specificity, 3),
        'F1 Score': round(f1, 3),
        'Accuracy': round(accuracy, 3)
    })

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})", color=colors[idx % len(colors)])

# Plot LSTM ROC summary
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM - Yearly Temporal ROC Curves (Target=MPR)")
plt.legend()
plt.grid()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()

roc_path = os.path.join(base_path, "yearly_temporal_ROC_LSTM.png")
plt.savefig(roc_path)
print(f"Saved LSTM ROC curve at {roc_path}")

# Save LSTM metrics table
df_metrics = pd.DataFrame(metrics_table)
print(df_metrics)
df_metrics.to_csv(os.path.join(base_path, "lstm_temporal_metrics_summary.csv"), index=False)

print("\nAll done. Key outputs:")
print(f"- ROC PNGs in {base_path}")
print(f"- SHAP plots/CSVs in {shap_dir}")
print(f"- Fairness/Drift/Clinical plots in {fig_dir}")


# **LSTM **

# In[3]:


# ============================================
# Classic ML (sklearn) + LSTM, target = MPR
# + LightGBM SHAP + Fairness/Drift + Clinical Plots
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

import seaborn as sns
sns.set(style="whitegrid", rc={"figure.dpi": 150})

import xgboost as xgb
import lightgbm as lgb
import shap

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from scipy.stats import ks_2samp
# ---------------------------
# Paths
# ---------------------------
base_path = r"D:\HIV Prevention\cohort\Jan_2026\Models"
fig_dir = os.path.join(base_path, "figs_insights")
os.makedirs(fig_dir, exist_ok=True)
shap_dir_lstm = os.path.join(base_path, "shap_lstm")
os.makedirs(shap_dir_lstm, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def normalize_sex_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """If SEX is object, normalize to M=1, F=0 with nullable Int64."""
    if 'SEX' in df.columns and df['SEX'].dtype == object:
        df['SEX'] = (
            df['SEX'].astype(str).str.strip().str.upper()
              .replace({'MALE': 'M', 'FEMALE': 'F'})
              .map({'M': 1, 'F': 0})
              .astype('Int64')
        )
    return df

def prepare_xy(df: pd.DataFrame):
    """Prepare numeric X and binary y=MPR (already 0/1)."""
    df = normalize_sex_if_needed(df)

    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset. Expected binary MPR target.")

    y = df['MPR'].astype(int)
    X = df.drop(columns=[c for c in ['MPR', 'ID', 'patient_id'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def prepare_xy_numeric(df: pd.DataFrame):
    """Same as prepare_xy but returns X,y arrays; numeric-only features."""
    df = normalize_sex_if_needed(df)
    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset.")
    y = df['MPR'].astype(int).values
    X = df.drop(columns=[c for c in ['MPR', 'ID', 'patient_id'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def _save_current_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------------------------
# Sklearn models (unchanged)
# ---------------------------
models = {
    "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=1000)),
    "Decision Tree":       make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier()),
    "Random Forest":       make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier()),
    "XGBoost":             make_pipeline(SimpleImputer(strategy="median"), xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)),
    "LightGBM":            make_pipeline(SimpleImputer(strategy="median"), lgb.LGBMClassifier())
}

for model_name, model in models.items():
    plt.figure(figsize=(12, 8))
    any_plotted = False

    for test_year in range(2014, 2024):
        train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
        test_file  = os.path.join(base_path, f"test_{test_year}_final.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"Missing files for {test_year}, skipping: {train_file} | {test_file}")
            continue

        train = pd.read_csv(train_file)
        test  = pd.read_csv(test_file)
        if 'PDC' in train.columns: del train['PDC']
        if 'PDC' in test.columns:  del test['PDC']
        if 'AGE_at_index' in train.columns: del train['AGE_at_index']
        if 'AGE_at_index' in test.columns:  del test['AGE_at_index']

        X_train, y_train = prepare_xy(train)
        X_test,  y_test  = prepare_xy(test)

        if y_train.nunique(dropna=True) < 2 or y_test.nunique(dropna=True) < 2:
            print(f"Skipping {test_year}: not enough label classes.")
            continue

        model.fit(X_train, y_train)

        final_est = model[-1] if hasattr(model, "__getitem__") else model
        if hasattr(final_est, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(final_est, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = model.predict(X_test).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})")
        any_plotted = True

        y_pred = (y_proba >= 0.5).astype(int)
        print(f"\n{model_name} Classification Report for {test_year-2}-{test_year-1}→{test_year}")
        print(classification_report(y_test, y_pred))

    if any_plotted:
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} - Yearly Temporal ROC Curves (Target=MPR)")
        plt.legend()
        plt.grid()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        _save_current_fig(os.path.join(base_path, f"yearly_temporal_ROC_{model_name.replace(' ', '_')}.png"))
    else:
        plt.close()
        print(f"No valid splits for {model_name}; ROC not saved.")

# ---------------------------
# LSTM (sequence_len = 1) + LSTM SHAP + LSTM Calibration & Fairness
# ---------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).view(-1)   # probabilities in [0,1]

metrics_table = []
lstm_yearly_pred_rows = []   # for calibration & demographics
lstm_yearly_shap_rows = []   # per-year mean |SHAP| feature importances

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10.colors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, test_year in enumerate(range(2014, 2024)):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_merged.csv")
    test_file  = os.path.join(base_path, f"test_{test_year}_merged.csv")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Files missing for {test_year}, skipping.")
        continue

    train = pd.read_csv(train_file)
    test  = pd.read_csv(test_file)
    test_raw = test.copy()  # keep raw for group columns

    # X, y
    X_train_df, y_train = prepare_xy(train)
    X_test_df,  y_test  = prepare_xy(test)

    # Class check
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Skipping {test_year}: not enough classes.")
        continue

    # Impute (train medians for both) and scale
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns)
    X_test  = pd.DataFrame(imputer.transform(X_test_df),  columns=X_test_df.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # LSTM expects (N, T, F); we set T = 1
    X_train_seq = np.expand_dims(X_train_scaled, axis=1)
    X_test_seq  = np.expand_dims(X_test_scaled,  axis=1)

    # DataLoaders
    train_dataset = LSTMDataset(X_train_seq, y_train.values)
    test_dataset  = LSTMDataset(X_test_seq,  y_test.values)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Model
    model = LSTMModel(input_dim=X_train_seq.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_probs, y_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            y_probs.extend(preds.cpu().numpy())
            y_true.extend(yb.numpy())

    y_probs = np.array(y_probs)
    y_true  = np.array(y_true)
    y_pred  = (y_probs >= 0.5).astype(int)

    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0

    metrics_table.append({
        'Model': 'LSTM',
        'Year': f"{test_year-2}-{test_year-1}→{test_year}",
        'Precision': round(precision, 3),
        'Recall (Sensitivity)': round(sensitivity, 3),
        'Specificity': round(specificity, 3),
        'F1 Score': round(f1, 3),
        'Accuracy': round(accuracy, 3)
    })

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})", color=colors[idx % len(colors)])

    # -------- LSTM: store per-year predictions for calibration/fairness
    year_label = f"{test_year-2}-{test_year-1}→{test_year}"
    perf_df = pd.DataFrame({
        "year_split": year_label,
        "y_true": y_true,
        "y_prob": y_probs
    })
    for col in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"]:
        if col in test_raw.columns:
            perf_df[col] = test_raw[col].reset_index(drop=True).iloc[:len(perf_df)].values
    lstm_yearly_pred_rows.append(perf_df)

    # -------- LSTM SHAP (KernelExplainer, model-agnostic)
    # Build a prediction fn from numpy -> numpy probs
    def predict_proba_numpy(x_np):
        x_seq = np.expand_dims(x_np, axis=1)  # add time dimension
        with torch.no_grad():
            tens = torch.tensor(x_seq, dtype=torch.float32, device=device)
            out = model(tens)                 # probabilities
            return out.cpu().numpy()

    # Background: small sample from TRAIN (post-impute/scale)
    bg_n = min(100, X_train.shape[0])
    bg_idx = np.random.RandomState(42).choice(X_train.shape[0], size=bg_n, replace=False)
    background = X_train.values[bg_idx]

    # Sample TEST for SHAP
    shap_n = min(200, X_test.shape[0])
    test_idx = np.random.RandomState(42).choice(X_test.shape[0], size=shap_n, replace=False)
    test_sample = X_test.values[test_idx]

    try:
        explainer = shap.KernelExplainer(predict_proba_numpy, background)
        shap_vals = explainer.shap_values(test_sample, nsamples="auto")
        # shap returns array [N, F]
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3:  # sometimes list-like -> [classes x N x F]; binary -> take positive class (index 0)
            shap_vals = shap_vals[0]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        year_imp_df = pd.DataFrame({
            "feature": X_train.columns,
            "mean_abs_shap": mean_abs,
            "year": year_label
        }).sort_values("mean_abs_shap", ascending=False)
        lstm_yearly_shap_rows.append(year_imp_df)

        # Bar plot top-20
        top20 = year_imp_df.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1])
        plt.xlabel("Mean |SHAP| (KernelExplainer)")
        plt.title(f"LSTM Top-20 Features — {year_label}")
        _save_current_fig(os.path.join(shap_dir_lstm, f"lstm_shap_bar_{test_year-2}-{test_year-1}_to_{test_year}.png"))

        # Save per-year CSV
        year_imp_df.to_csv(os.path.join(shap_dir_lstm, f"lstm_shap_{test_year-2}-{test_year-1}_to_{test_year}.csv"), index=False)

    except Exception as e:
        print(f"[LSTM SHAP] {year_label} skipped due to: {e}")

# ---- Plot LSTM ROC summary
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM - Yearly Temporal ROC Curves (Target=MPR)")
plt.legend()
plt.grid()
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tight_layout()
roc_path = os.path.join(base_path, "yearly_temporal_ROC_LSTM.png")
plt.savefig(roc_path)
print(f"Saved LSTM ROC curve at {roc_path}")

# ---- Save LSTM metrics table
df_metrics = pd.DataFrame(metrics_table)
print(df_metrics)
df_metrics.to_csv(os.path.join(base_path, "lstm_temporal_metrics_summary.csv"), index=False)

# ---- Save aggregated LSTM SHAP importances
if lstm_yearly_shap_rows:
    all_imp_lstm = pd.concat(lstm_yearly_shap_rows, ignore_index=True)
    all_imp_lstm.to_csv(os.path.join(shap_dir_lstm, "lstm_shap_per_year.csv"), index=False)
    agg_lstm = (all_imp_lstm.groupby("feature")["mean_abs_shap"]
                .mean().sort_values(ascending=False).reset_index())
    agg_lstm.to_csv(os.path.join(shap_dir_lstm, "lstm_shap_all_years_mean_abs.csv"), index=False)
    print("[LSTM SHAP] Saved per-year and aggregated attributions.")

# ---------------------------
# LSTM Calibration (per year)
# ---------------------------
if lstm_yearly_pred_rows:
    preds_all_lstm = pd.concat(lstm_yearly_pred_rows, ignore_index=True)
    for yr, sub in preds_all_lstm.groupby("year_split"):
        if pd.Series(sub["y_true"]).nunique() < 2:
            print(f"[LSTM calibration] Skipping {yr}: only one class present.")
            continue
        prob_true, prob_pred = calibration_curve(sub["y_true"], sub["y_prob"], n_bins=10, strategy="quantile")
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='LSTM')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed probability')
        plt.title(f'Calibration: LSTM ({yr})')
        plt.legend()
        _save_current_fig(os.path.join(fig_dir, f"calibration_LSTM_{yr.replace('→','to')}.png"))

# ---------------------------
# LSTM Fairness/Demographics (AUC & Accuracy by group, per year)
# ---------------------------
if lstm_yearly_pred_rows:
    preds_all_lstm = pd.concat(lstm_yearly_pred_rows, ignore_index=True)
    group_cols = [c for c in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"] if c in preds_all_lstm.columns]

    for gcol in group_cols:
        for yr, sub in preds_all_lstm.groupby("year_split"):
            sub = sub.dropna(subset=[gcol])
            if pd.Series(sub["y_true"]).nunique() < 2:
                print(f"[LSTM demographics] {gcol} {yr}: only one class y_true; skipping.")
                continue

            rows = []
            for grp, df_g in sub.groupby(gcol):
                try:
                    auc_g = roc_auc_score(df_g["y_true"], df_g["y_prob"]) if pd.Series(df_g["y_true"]).nunique() == 2 else np.nan
                    acc_g = accuracy_score(df_g["y_true"], (df_g["y_prob"] >= 0.5).astype(int))
                    rows.append((str(grp), auc_g, acc_g))
                except Exception as e:
                    print(f"[LSTM demographics] {gcol}={grp} {yr} error: {e}")

            if not rows:
                continue

            groups, aucs, accs = zip(*rows)
            x = np.arange(len(groups))
            width = 0.35

            plt.figure(figsize=(9, 5))
            plt.bar(x - width/2, aucs, width, label='AUC')
            plt.bar(x + width/2, accs, width, label='Accuracy')
            plt.xticks(x, groups, rotation=30, ha='right')
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(f"LSTM Performance by {gcol} — {yr}")
            plt.legend()
            plt.grid(axis="y")
            _save_current_fig(os.path.join(fig_dir, f"demographic_perf_LSTM_{gcol}_{yr.replace('→','to')}.png"))

print("\nAll done. Key outputs:")
print(f"- LSTM ROC PNG: {roc_path}")
print(f"- LSTM metrics CSV: {os.path.join(base_path, 'lstm_temporal_metrics_summary.csv')}")
print(f"- LSTM SHAP (per-year & aggregated) in: {shap_dir_lstm}")
print(f"- LSTM calibration & demographic plots in: {fig_dir}")


# In[5]:


# ============================================
# LSTM ONLY (Target = MPR)
# Training/Testing across years + LSTM SHAP (beeswarm) + Calibration + Demographics
# ============================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve

import shap
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ---------------------------
# Paths
# ---------------------------
base_path = r"D:\HIV Prevention\cohort\Jan_2026\Models"
os.makedirs(base_path, exist_ok=True)

fig_dir = os.path.join(base_path, "figs_insights")
os.makedirs(fig_dir, exist_ok=True)

shap_dir_lstm = os.path.join(base_path, "shap_lstm")
os.makedirs(shap_dir_lstm, exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def normalize_sex_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """If SEX is object, normalize to M=1, F=0 with nullable Int64."""
    if 'SEX' in df.columns and df['SEX'].dtype == object:
        df['SEX'] = (
            df['SEX'].astype(str).str.strip().str.upper()
              .replace({'MALE': 'M', 'FEMALE': 'F'})
              .map({'M': 1, 'F': 0})
              .astype('Int64')
        )
    return df

def prepare_xy(df: pd.DataFrame):
    """Prepare numeric X and binary y=MPR (already 0/1)."""
    df = normalize_sex_if_needed(df)
    if 'MPR' not in df.columns:
        raise ValueError("No MPR column found in dataset. Expected binary MPR target.")
    y = df['MPR'].astype(int)
    X = df.drop(columns=[c for c in ['MPR', 'ID', 'PDC', 'patient_id'] if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=['number']).copy()
    return X, y

def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ---------------------------
# LSTM (sequence_len = 1)
# ---------------------------
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)          # logits

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])             # shape [N, 1]
        return logits                                # keep [N, 1] for SHAP

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------
# Collectors
# ---------------------------
metrics_table = []
lstm_yearly_pred_rows = []   # for calibration & demographics
lstm_yearly_shap_rows = []   # per-year mean |SHAP| feature importances

# ---------------------------
# Train/Eval across years
# ---------------------------
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10.colors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, test_year in enumerate(range(2014, 2024)):
    train_file = os.path.join(base_path, f"train_{test_year-2}-{test_year-1}_final.csv")
    test_file  = os.path.join(base_path, f"test_{test_year}_final.csv")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Files missing for {test_year}, skipping.")
        continue

    train = pd.read_csv(train_file)
    test  = pd.read_csv(test_file)
    test_raw = test.copy()  # keep raw for group columns

    # Prepare features/labels
    X_train_df, y_train = prepare_xy(train)
    X_test_df,  y_test  = prepare_xy(test)

    # Class check
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print(f"Skipping {test_year}: not enough classes.")
        continue

    # Impute (median from train) & scale
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_df), columns=X_train_df.columns, index=X_train_df.index)
    X_test  = pd.DataFrame(imputer.transform(X_test_df),  columns=X_test_df.columns,  index=X_test_df.index)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # LSTM expects (N, T, F); we set T = 1
    X_train_seq = np.expand_dims(X_train_scaled, axis=1)
    X_test_seq  = np.expand_dims(X_test_scaled,  axis=1)

    # DataLoaders
    train_dataset = LSTMDataset(X_train_seq, y_train.values)
    test_dataset  = LSTMDataset(X_test_seq,  y_test.values)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # Model
    model = LSTMModel(input_dim=X_train_seq.shape[2]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)                  # [N, 1]
            loss = criterion(logits.squeeze(1), yb)  # squeeze to [N] for loss
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    y_logits, y_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))             # [N, 1]
            y_logits.extend(logits.squeeze(1).cpu().numpy())
            y_true.extend(yb.numpy())

    y_logits = np.array(y_logits)                     # [N]
    y_probs  = sigmoid_np(y_logits)                   # probabilities
    y_true   = np.array(y_true)
    y_pred   = (y_probs >= 0.5).astype(int)

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall_pos  = recall_score(y_true, y_pred, zero_division=0)  # same as sensitivity
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    f1          = f1_score(y_true, y_pred, zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)

    metrics_table.append({
        'Model': 'LSTM',
        'Year': f"{test_year-2}-{test_year-1}→{test_year}",
        'Precision': round(precision, 3),
        'Recall (Sensitivity)': round(sensitivity, 3),
        'Specificity': round(specificity, 3),
        'F1 Score': round(f1, 3),
        'Accuracy': round(accuracy, 3)
    })

    # ROC overlay
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{test_year-2}-{test_year-1}→{test_year} (AUC={roc_auc:.2f})",
             color=colors[idx % len(colors)])

    # Store per-year predictions for calibration/demographics
    year_label = f"{test_year-2}-{test_year-1}→{test_year}"
    perf_df = pd.DataFrame({"year_split": year_label, "y_true": y_true, "y_prob": y_probs})
    for col in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"]:
        if col in test_raw.columns:
            perf_df[col] = test_raw[col].reset_index(drop=True).iloc[:len(perf_df)].values
    lstm_yearly_pred_rows.append(perf_df)

    # ---------------------------
    # SHAP via GradientExplainer + beeswarm
    # ---------------------------
    try:
        # Background/sample tensors on the SAME device; keep LSTM sequence shape [N, 1, F]
        bg_n = min(100, X_train_seq.shape[0])
        bg = torch.tensor(X_train_seq[:bg_n], dtype=torch.float32, device=device)  # [B, 1, F]
        explainer = shap.GradientExplainer(model, bg)

        samp_n = min(200, X_test_seq.shape[0])
        test_sample = torch.tensor(X_test_seq[:samp_n], dtype=torch.float32, device=device)  # [S, 1, F]

        shap_vals_list = explainer.shap_values(test_sample)  # single-output -> list of len 1
        shap_vals = shap_vals_list[0] if isinstance(shap_vals_list, list) else shap_vals_list  # [S, 1, F]
        shap_vals_feat = shap_vals.mean(axis=1)  # average over time dim (T=1) -> [S, F]

        # Build a SHAP Explanation for beeswarm using the same S rows from (imputed+scaled) X_test
        S = shap_vals_feat.shape[0]
        explanation = shap.Explanation(
            values=shap_vals_feat,
            data=X_test.iloc[:S].values,
            feature_names=X_test.columns.tolist()
        )

        plt.figure()
        shap.plots.beeswarm(explanation, show=False)
        _save_fig(os.path.join(
            shap_dir_lstm, f"lstm_shap_beeswarm_{test_year-2}-{test_year-1}_to_{test_year}.png"
        ))

        # Save per-year mean |SHAP|
        mean_abs = np.abs(shap_vals_feat).mean(axis=0)
        year_imp_df = pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs,
            "year": year_label
        }).sort_values("mean_abs_shap", ascending=False)
        lstm_yearly_shap_rows.append(year_imp_df)
        # Also save per-year CSV
        year_imp_df.to_csv(os.path.join(
            shap_dir_lstm, f"lstm_shap_{test_year-2}-{test_year-1}_to_{test_year}.csv"
        ), index=False)

    except Exception as e:
        print(f"[LSTM SHAP] {year_label} skipped due to: {e}")

# ---------------------------
# Finalize ROC & Metrics
# ---------------------------
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("LSTM - Yearly Temporal ROC Curves (Target=MPR)")
plt.legend(); plt.grid(); plt.tight_layout()
roc_path = os.path.join(base_path, "yearly_temporal_ROC_LSTM.png")
plt.savefig(roc_path); plt.close()
print(f"Saved LSTM ROC curve at {roc_path}")

df_metrics = pd.DataFrame(metrics_table)
print(df_metrics)
df_metrics.to_csv(os.path.join(base_path, "lstm_temporal_metrics_summary.csv"), index=False)

# ---------------------------
# Save per-year & aggregated LSTM SHAP importances
# ---------------------------
if lstm_yearly_shap_rows:
    all_imp = pd.concat(lstm_yearly_shap_rows, ignore_index=True)
    all_imp.to_csv(os.path.join(shap_dir_lstm, "lstm_shap_per_year.csv"), index=False)

    agg = (all_imp.groupby("feature")["mean_abs_shap"]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index())
    agg.to_csv(os.path.join(shap_dir_lstm, "lstm_shap_all_years_mean_abs.csv"), index=False)
    print("[LSTM SHAP] Saved per-year and aggregated attributions.")

# ---------------------------
# Calibration plots (LSTM ONLY)
# ---------------------------
if lstm_yearly_pred_rows:
    preds_all = pd.concat(lstm_yearly_pred_rows, ignore_index=True)
    years = preds_all["year_split"].unique()

    for yr in years:
        sub = preds_all[preds_all["year_split"] == yr]
        if sub["y_true"].nunique() < 2:
            print(f"[LSTM calibration] Skipping {yr}: only one class present.")
            continue

        prob_true, prob_pred = calibration_curve(
            sub["y_true"], sub["y_prob"], n_bins=10, strategy="quantile"
        )
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='LSTM')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed probability')
        plt.title(f'Calibration: LSTM ({yr})')
        plt.legend()
        _save_fig(os.path.join(fig_dir, f"calibration_LSTM_{yr.replace('→','to')}.png"))

# ---------------------------
# Demographic/Fairness plots (LSTM ONLY)
# ---------------------------
if lstm_yearly_pred_rows:
    preds_all = pd.concat(lstm_yearly_pred_rows, ignore_index=True)
    years = preds_all["year_split"].unique()
    group_cols = [c for c in ["SEX", "AGE_GROUP", "INSURANCE_TYPE", "RACE", "ETHNICITY"]
                  if c in preds_all.columns]

    for gcol in group_cols:
        for yr in years:
            sub = preds_all[preds_all["year_split"] == yr].dropna(subset=[gcol])
            if sub["y_true"].nunique() < 2:
                print(f"[LSTM demographics] {gcol} {yr}: only one class y_true; skipping.")
                continue

            rows = []
            for grp, df_g in sub.groupby(gcol):
                try:
                    auc_g = roc_auc_score(df_g["y_true"], df_g["y_prob"]) \
                            if df_g["y_true"].nunique() == 2 else np.nan
                    acc_g = accuracy_score(df_g["y_true"], (df_g["y_prob"] >= 0.5).astype(int))
                    rows.append((str(grp), auc_g, acc_g))
                except Exception as e:
                    print(f"[LSTM demographics] {gcol}={grp} {yr} error: {e}")

            if not rows:
                continue

            # Bar plot
            groups, aucs, accs = zip(*rows)
            x = np.arange(len(groups))
            width = 0.35
            plt.figure(figsize=(9, 5))
            plt.bar(x - width/2, aucs, width, label='AUC')
            plt.bar(x + width/2, accs, width, label='Accuracy')
            plt.xticks(x, groups, rotation=30, ha='right')
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.title(f"LSTM Performance by {gcol} — {yr}")
            plt.legend()
            plt.grid(axis="y")
            _save_fig(os.path.join(fig_dir, f"demographic_perf_LSTM_{gcol}_{yr.replace('→','to')}.png"))

print("\nAll done. Key outputs:")
print(f"- ROC PNG: {roc_path}")
print(f"- LSTM metrics CSV: {os.path.join(base_path, 'lstm_temporal_metrics_summary.csv')}")
print(f"- LSTM SHAP (per-year & aggregated) in: {shap_dir_lstm}")
print(f"- LSTM calibration & demographic plots in: {fig_dir}")


# In[ ]:




