"""
RoomSense AI - Phase 1, 2, 3 Execution Script
Runs EDA, Preprocessing, Outlier Analysis, Statistical Tests, and generates all Phase 1-3 charts.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Output directories
os.makedirs('outputs/graphs', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load Data
df = pd.read_csv('data/Occupancy_Estimation.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Datetime sorting
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)

# 3. Target Distribution Chart
occupancy_counts = df['Room_Occupancy_Count'].value_counts().sort_index()
plt.figure(figsize=(8, 5))
colors = ['#2B5B84', '#3E92CC', '#F1A208', '#D8315B']
bars = plt.bar(occupancy_counts.index, occupancy_counts.values, color=colors, edgecolor='black', width=0.6)
plt.title("Room Occupancy Class Distribution (UCI Dataset)", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("Occupancy Count (Number of People)", fontsize=11)
plt.ylabel("Number of Observations", fontsize=11)
plt.xticks(occupancy_counts.index, ['0 (Empty)', '1 Person', '2 People', '3 People'])
for bar, count in zip(bars, occupancy_counts.values):
    pct = (count / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f"{count:,}\n({pct:.1f}%)", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylim(0, 9500)
plt.tight_layout()
plt.savefig('outputs/graphs/occupancy_distribution.png', dpi=300)
plt.close()
print("Saved outputs/graphs/occupancy_distribution.png")

# 4. Outlier Detection (IQR & Z-score)
sensor_cols = [c for c in df.columns if c.startswith('S') and not c.endswith('PIR')]
outlier_report = []
for col in sensor_cols:
    data_col = df[col]
    q1 = data_col.quantile(0.25)
    q3 = data_col.quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = ((data_col < (q1 - 1.5 * iqr)) | (data_col > (q3 + 1.5 * iqr))).sum()
    z_outliers = (np.abs(stats.zscore(data_col)) > 3.0).sum()
    outlier_report.append({
        'Feature': col,
        'Mean': round(data_col.mean(), 2),
        'Std': round(data_col.std(), 2),
        'IQR_Outliers': iqr_outliers,
        'IQR_Pct': round((iqr_outliers / len(df)) * 100, 2),
        'ZScore_Outliers': z_outliers,
        'ZScore_Pct': round((z_outliers / len(df)) * 100, 2)
    })
outlier_df = pd.DataFrame(outlier_report)
outlier_df.to_csv('outputs/results/outlier_summary.csv', index=False)
print("Saved outputs/results/outlier_summary.csv")

# 5. Isolation Forest Anomaly Analysis
iso_forest = IsolationForest(contamination=0.03, random_state=42)
df['Anomaly_Flag'] = (iso_forest.fit_predict(df[sensor_cols]) == -1).astype(int)

# 6. Scaler comparison chart
sample_features = df[['S1_Temp', 'S5_CO2']].copy()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(StandardScaler().fit_transform(sample_features)[:, 1], bins=30, color='#1f77b4', edgecolor='black')
axes[0].set_title("StandardScaler (S5_CO2)\nMean=0, Std=1")
axes[1].hist(MinMaxScaler().fit_transform(sample_features)[:, 1], bins=30, color='#2ca02c', edgecolor='black')
axes[1].set_title("MinMaxScaler (S5_CO2)\nRange [0, 1]")
axes[2].hist(RobustScaler().fit_transform(sample_features)[:, 1], bins=30, color='#ff7f0e', edgecolor='black')
axes[2].set_title("RobustScaler (S5_CO2)\nMedian=0, IQR Scaled")
plt.tight_layout()
plt.savefig('outputs/graphs/scaler_comparison.png', dpi=300)
plt.close()
print("Saved outputs/graphs/scaler_comparison.png")

# 7. Key Sensor Histograms
key_sensors = ['S1_Temp', 'S1_Light', 'S1_Sound', 'S5_CO2']
units = ['(°C)', '(Lux)', '(Volts)', '(ppm)']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (col, unit) in enumerate(zip(key_sensors, units)):
    sns.histplot(df[col], kde=True, ax=axes[i], color='#2B5B84', edgecolor='black', bins=30)
    axes[i].set_title(f"Distribution of {col} {unit}", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graphs/sensor_histograms.png', dpi=300)
plt.close()
print("Saved outputs/graphs/sensor_histograms.png")

# 8. Bivariate Boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (col, unit) in enumerate(zip(key_sensors, units)):
    sns.boxplot(x='Room_Occupancy_Count', y=col, data=df, ax=axes[i], palette='Blues', hue='Room_Occupancy_Count', legend=False)
    axes[i].set_title(f"{col} by Room Occupancy Level", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graphs/sensor_vs_occupancy_boxplots.png', dpi=300)
plt.close()
print("Saved outputs/graphs/sensor_vs_occupancy_boxplots.png")

# 9. Correlation Heatmaps
numeric_features = [c for c in df.columns if c.startswith('S')] + ['Room_Occupancy_Count']
plt.figure(figsize=(12, 9))
sns.heatmap(df[numeric_features].corr(method='pearson'), annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0)
plt.title("Pearson Linear Correlation Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graphs/correlation_heatmap.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 9))
sns.heatmap(df[numeric_features].corr(method='spearman'), annot=True, fmt='.2f', cmap='viridis', vmin=-0.2, vmax=1.0)
plt.title("Spearman Rank Correlation Matrix", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graphs/spearman_heatmap.png', dpi=300)
plt.close()
print("Saved correlation heatmaps to outputs/graphs/")

# 10. Time-Series Plot
day_sample = df[df['Date'] == '2017/12/22'].copy()
fig, ax1 = plt.subplots(figsize=(12, 4.5))
color = '#1f77b4'
ax1.set_xlabel('Time of Day (2017/12/22)')
ax1.set_ylabel('S5_CO2 (ppm)', color=color)
ax1.plot(pd.to_datetime(day_sample['Time']), day_sample['S5_CO2'], color=color, linewidth=2)
ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Occupancy Count', color=color)
ax2.step(pd.to_datetime(day_sample['Time']), day_sample['Room_Occupancy_Count'], color=color, where='post', linewidth=2)
plt.title("Time-Series: Environmental CO2 vs Occupancy (2017/12/22)", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig('outputs/graphs/timeseries_occupancy_co2.png', dpi=300)
plt.close()
print("Saved outputs/graphs/timeseries_occupancy_co2.png")

# 11. Statistical Hypothesis Testing (ANOVA and Kruskal-Wallis)
hyp_results = []
for feature in ['S5_CO2', 'S1_Light', 'S1_Temp', 'S1_Sound']:
    groups = [df[df['Room_Occupancy_Count'] == k][feature].values for k in [0, 1, 2, 3]]
    f_stat, p_val_anova = stats.f_oneway(*groups)
    h_stat, p_val_kw = stats.kruskal(*groups)
    hyp_results.append({
        'Feature': feature,
        'ANOVA_F_Stat': round(f_stat, 2),
        'ANOVA_p_val': f"{p_val_anova:.3e}",
        'Kruskal_H_Stat': round(h_stat, 2),
        'Kruskal_p_val': f"{p_val_kw:.3e}",
        'Significant (p<0.05)': 'Yes' if p_val_anova < 0.05 else 'No'
    })
hyp_df = pd.DataFrame(hyp_results)
hyp_df.to_csv('outputs/results/hypothesis_tests.csv', index=False)
print("Saved outputs/results/hypothesis_tests.csv")

# 12. Save cleaned dataset
df.to_csv('data/cleaned_data.csv', index=False)
print("Saved data/cleaned_data.csv")
print("Phase 1-3 script completed successfully!")

