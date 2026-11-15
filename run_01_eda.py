"""
RoomSense AI — 01: EDA & Preprocessing Pipeline
Run directly from terminal: python run_01_eda.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

print("=================================================================")
print("  RoomSense AI: Step 01 — Exploratory Data Analysis & Preprocessing")
print("=================================================================")

# Setup paths
PROJECT_ROOT = '.' if os.path.exists('data/Occupancy_Estimation.csv') else '..'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Occupancy_Estimation.csv')
CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_data.csv')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
print(f"[1/6] Loading UCI dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Datetime Processing
print("[2/6] Parsing Datetime index and validating missing values...")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)
missing = df.isnull().sum().sum()
duplicates = df.duplicated(subset=['Date', 'Time']).sum()
print(f"      Missing values: {missing}, Duplicates: {duplicates}")

# 3. Class Distribution
print("[3/6] Analyzing Occupancy class distribution...")
counts = df['Room_Occupancy_Count'].value_counts().sort_index()
for cls, cnt in counts.items():
    pct = (cnt / len(df)) * 100
    print(f"      Class {cls}: {cnt:,} samples ({pct:.1f}%)")

plt.figure(figsize=(8, 4.5))
colors = ['#2B5B84', '#3E92CC', '#F1A208', '#D8315B']
bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='black', width=0.55)
plt.title("Room Occupancy Class Distribution (UCI Dataset ID 864)", fontsize=12, fontweight='bold')
plt.xlabel("Occupancy Level (People)")
plt.ylabel("Number of 30-Second Observations")
plt.xticks(counts.index, ['0 (Empty)', '1 Person', '2 People', '3 People'])
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'occupancy_distribution.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/occupancy_distribution.png")

# 4. Outlier & Scaling Visualizations
print("[4/6] Generating Outlier Boxplots & Scaler Comparison...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
sns.boxplot(y=df['S1_Temp'], ax=axes[0], color='#A5B4FC')
axes[0].set_title("Temperature (S1_Temp)")
sns.boxplot(y=df['S1_Light'], ax=axes[1], color='#38BDF8')
axes[1].set_title("Light (S1_Light)")
sns.boxplot(y=df['S1_Sound'], ax=axes[2], color='#34D399')
axes[2].set_title("Sound (S1_Sound)")
sns.boxplot(y=df['S5_CO2'], ax=axes[3], color='#F472B6')
axes[3].set_title("CO2 (S5_CO2)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'outlier_boxplots.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/outlier_boxplots.png")

# Scaler comparison
sample_features = df[['S1_Temp', 'S5_CO2']].copy()
standard_scaled = StandardScaler().fit_transform(sample_features)
minmax_scaled = MinMaxScaler().fit_transform(sample_features)
robust_scaled = RobustScaler().fit_transform(sample_features)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(standard_scaled[:, 1], bins=30, color='#1f77b4', edgecolor='black')
axes[0].set_title("StandardScaler (S5_CO2)")
axes[1].hist(minmax_scaled[:, 1], bins=30, color='#2ca02c', edgecolor='black')
axes[1].set_title("MinMaxScaler (S5_CO2)")
axes[2].hist(robust_scaled[:, 1], bins=30, color='#ff7f0e', edgecolor='black')
axes[2].set_title("RobustScaler (S5_CO2)")
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'scaler_comparison.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/scaler_comparison.png")

# 5. Correlation Heatmap
print("[5/6] Generating Pearson and Spearman Correlation Heatmaps...")
numeric_features = [c for c in df.columns if c.startswith('S')] + ['Room_Occupancy_Count']
plt.figure(figsize=(11, 8))
sns.heatmap(df[numeric_features].corr(method='pearson'), annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0)
plt.title("Pearson Linear Correlation Matrix", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'correlation_heatmap.png'), dpi=300)
plt.close()
print("      Saved: outputs/graphs/correlation_heatmap.png")

# 6. Save Cleaned Data
print(f"[6/6] Saving cleaned dataset to {CLEANED_PATH}...")
df.to_csv(CLEANED_PATH, index=False)
print("=================================================================")
print("Step 01 Complete! Cleaned data and graphs generated successfully.")
print("=================================================================")

