"""
RoomSense AI — 02: Feature Engineering & Selection Pipeline
Run directly from terminal: python run_02_features.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_utils import engineer_all_features

print("=================================================================")
print("  RoomSense AI: Step 02 — Feature Engineering & Selection")
print("=================================================================")

PROJECT_ROOT = '.' if os.path.exists('data/cleaned_data.csv') else '..'
CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_data.csv')
ENGINEERED_PATH = os.path.join(PROJECT_ROOT, 'data', 'engineered_data.csv')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Cleaned Data
print(f"[1/5] Loading cleaned dataset from {CLEANED_PATH}...")
df = pd.read_csv(CLEANED_PATH)
print(f"      Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Engineer Domain Features
print("[2/5] Engineering 22 domain features using feature_utils.py...")
df_engineered = engineer_all_features(df)
df_engineered['Room_Occupancy_Count'] = df['Room_Occupancy_Count'].values
print(f"      Total features after engineering: {df_engineered.shape[1]-1} (38 modeling features + target)")

# 3. Mutual Information & Random Forest Importance
print("[3/5] Computing feature importance (Mutual Information & RF Gini)...")
drop_cols = ['Date', 'Time', 'Datetime', 'Room_Occupancy_Count']
feature_cols = [c for c in df_engineered.columns if c not in drop_cols]
X = df_engineered[feature_cols].copy()
y = df_engineered['Room_Occupancy_Count'].copy()

# Sample for quick calculation
sample_idx = np.random.RandomState(42).choice(len(X), size=min(4000, len(X)), replace=False)
X_sub = X.iloc[sample_idx]
y_sub = y.iloc[sample_idx]

mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)
mi_df = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X_sub, y_sub)
rf_df = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Plot top 10 MI
plt.figure(figsize=(10, 4.5))
mi_df.head(10).plot(kind='barh', color='#2B5B84', edgecolor='black')
plt.title("Top 10 Features by Mutual Information Ranking", fontweight='bold')
plt.xlabel("Mutual Information Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'mutual_info_ranking.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/mutual_info_ranking.png")

# 4. PCA Projection
print("[4/5] Computing Principal Component Analysis (PCA)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Occupancy Count')
plt.title(f"2D PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)", fontweight='bold')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'pca_2d_occupancy.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/pca_2d_occupancy.png")

# 5. Save Engineered Dataset
print(f"[5/5] Saving engineered dataset to {ENGINEERED_PATH}...")
df_engineered.to_csv(ENGINEERED_PATH, index=False)
print("=================================================================")
print("Step 02 Complete! Engineered data and rankings saved.")
print("=================================================================")
