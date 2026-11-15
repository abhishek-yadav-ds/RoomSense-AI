"""
RoomSense AI - Phase 4, 5, 6, 7 Execution Script
Feature Engineering, Selection, PCA, Clustering, and Physical Sensor Optimization.
"""

import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import feature_utils

os.makedirs('outputs/graphs', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 1. Load Cleaned Data
df = pd.read_csv('data/cleaned_data.csv')
y = df['Room_Occupancy_Count']

# 2. Compute Engineered Features using feature_utils
print("Engineering features...")
X_engineered = feature_utils.engineer_all_features(df, include_raw=True)
print(f"Engineered feature matrix: {X_engineered.shape[1]} features (from 16 raw features)")

# Save full engineered dataset
full_engineered_df = pd.concat([X_engineered, y], axis=1)
full_engineered_df.to_csv('data/engineered_data.csv', index=False)
print("Saved data/engineered_data.csv")

# 3. Feature Selection: Filter Methods (Mutual Information & ANOVA F-Test)
print("Computing Mutual Information and ANOVA F-tests...")
mi_scores = mutual_info_classif(X_engineered, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X_engineered.columns).sort_values(ascending=False)

f_scores, p_vals = f_classif(X_engineered, y)
anova_series = pd.Series(f_scores, index=X_engineered.columns).sort_values(ascending=False)

# Plot Mutual Information
plt.figure(figsize=(10, 8))
mi_series.head(15).sort_values().plot(kind='barh', color='#2B5B84', edgecolor='black')
plt.title("Top 15 Features by Mutual Information Score", fontsize=13, fontweight='bold', pad=10)
plt.xlabel("Mutual Information Score", fontsize=11)
plt.tight_layout()
plt.savefig('outputs/graphs/mutual_info_ranking.png', dpi=300)
plt.close()
print("Saved outputs/graphs/mutual_info_ranking.png")

# 4. Embedded Method: Random Forest Feature Importance
print("Computing Random Forest Feature Importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_engineered, y)
rf_importance = pd.Series(rf.feature_importances_, index=X_engineered.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
rf_importance.head(15).sort_values().plot(kind='barh', color='#3E92CC', edgecolor='black')
plt.title("Top 15 Features by Random Forest Gini Importance (MDI)", fontsize=13, fontweight='bold', pad=10)
plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11)
plt.tight_layout()
plt.savefig('outputs/graphs/feature_importance_rf.png', dpi=300)
plt.close()
print("Saved outputs/graphs/feature_importance_rf.png")

# 5. Wrapper Method: RFE (Recursive Feature Elimination)
print("Running Recursive Feature Elimination (RFE) for Top 8 features...")
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), n_features_to_select=8, step=2)
rfe.fit(X_engineered, y)
rfe_selected_features = X_engineered.columns[rfe.support_].tolist()
print("RFE Selected Features:", rfe_selected_features)

# 6. Feature Count Experiment (Comparing 5, 8, 10, 15, 20, All features using 5-Fold Stratified CV)
print("Evaluating performance across feature subset sizes...")
feature_counts = [5, 8, 10, 15, 20, len(X_engineered.columns)]
cv_f1_scores = []
cv_acc_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in feature_counts:
    top_k_features = rf_importance.head(k).index.tolist()
    fold_f1 = []
    fold_acc = []
    for train_idx, test_idx in cv.split(X_engineered, y):
        X_tr, X_te = X_engineered.iloc[train_idx][top_k_features], X_engineered.iloc[test_idx][top_k_features]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        
        fold_f1.append(f1_score(y_te, preds, average='macro'))
        fold_acc.append(accuracy_score(y_te, preds))
        
    cv_f1_scores.append(np.mean(fold_f1))
    cv_acc_scores.append(np.mean(fold_acc))
    print(f"Top {k} features: Macro F1 = {np.mean(fold_f1):.4f}, Accuracy = {np.mean(fold_acc):.4f}")

plt.figure(figsize=(8, 5))
plt.plot(feature_counts, cv_f1_scores, marker='o', color='#2B5B84', linewidth=2, label='Macro F1-Score')
plt.plot(feature_counts, cv_acc_scores, marker='s', color='#D8315B', linewidth=2, linestyle='--', label='Accuracy')
plt.title("Model Performance vs Feature Subset Count", fontsize=13, fontweight='bold', pad=10)
plt.xlabel("Number of Selected Features", fontsize=11)
plt.ylabel("Score (5-Fold CV)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig('outputs/graphs/f1_vs_feature_count.png', dpi=300)
plt.close()
print("Saved outputs/graphs/f1_vs_feature_count.png")

# 7. Physical Sensor Group Optimization Study
print("Analyzing physical sensor group configurations...")
sensor_configs = {
    'All Sensors (16 Raw)': feature_utils.ALL_RAW_SENSOR_FEATURES,
    'Temp + CO2 + PIR (7 Sensors)': feature_utils.RAW_TEMP_FEATURES + feature_utils.RAW_CO2_FEATURES + feature_utils.RAW_PIR_FEATURES,
    'CO2 + PIR (3 Sensors - Minimal)': feature_utils.RAW_CO2_FEATURES + feature_utils.RAW_PIR_FEATURES,
    'Light + Sound + PIR (10 Sensors)': feature_utils.RAW_LIGHT_FEATURES + feature_utils.RAW_SOUND_FEATURES + feature_utils.RAW_PIR_FEATURES,
    'Top 8 Engineered Features': rf_importance.head(8).index.tolist()
}

sensor_comparison = []

for name, feature_list in sensor_configs.items():
    if all(col in X_engineered.columns for col in feature_list):
        X_sub = X_engineered[feature_list]
    else:
        X_sub = df[[c for c in feature_list if c in df.columns]]
        
    # Split
    X_train_sub = X_sub.iloc[:8000]
    y_train_sub = y.iloc[:8000]
    X_test_sub = X_sub.iloc[8000:]
    y_test_sub = y.iloc[8000:]
    
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_sub, y_train_sub)
    train_time = round(time.time() - start_time, 3)
    
    start_pred = time.time()
    preds = clf.predict(X_test_sub)
    pred_time_ms = round(((time.time() - start_pred) / len(X_test_sub)) * 1000, 3)
    
    acc = accuracy_score(y_test_sub, preds)
    f1 = f1_score(y_test_sub, preds, average='macro', zero_division=0)
    
    sensor_comparison.append({
        'Configuration': name,
        'Feature_Count': len(feature_list),
        'Accuracy': round(acc, 4),
        'Macro_F1': round(f1, 4),
        'Train_Time_s': train_time,
        'Latency_ms_per_sample': pred_time_ms
    })

sensor_comp_df = pd.DataFrame(sensor_comparison)
sensor_comp_df.to_csv('outputs/results/sensor_optimization_comparison.csv', index=False)
print("Saved outputs/results/sensor_optimization_comparison.csv:")
print(sensor_comp_df)

# Plot Sensor Comparison Bar Chart
plt.figure(figsize=(10, 5))
x_pos = np.arange(len(sensor_comp_df))
width = 0.35
plt.bar(x_pos - width/2, sensor_comp_df['Accuracy'], width, label='Accuracy', color='#2B5B84', edgecolor='black')
plt.bar(x_pos + width/2, sensor_comp_df['Macro_F1'], width, label='Macro F1-Score', color='#F1A208', edgecolor='black')
plt.xticks(x_pos, sensor_comp_df['Configuration'], rotation=20, ha='right', fontsize=9)
plt.ylabel("Score")
plt.title("Physical Sensor Group Optimization Comparison", fontsize=13, fontweight='bold', pad=10)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/graphs/sensor_group_comparison.png', dpi=300)
plt.close()
print("Saved outputs/graphs/sensor_group_comparison.png")

# 8. Principal Component Analysis (PCA)
print("Running Principal Component Analysis (PCA)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_engineered)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# Plot Explained Variance
plt.figure(figsize=(9, 5))
plt.bar(range(1, 16), explained_var[:15], alpha=0.6, color='#2B5B84', label='Individual Variance')
plt.step(range(1, 16), cum_var[:15], where='mid', color='#D8315B', linewidth=2, label='Cumulative Variance')
plt.axhline(y=0.90, color='gray', linestyle='--', label='90% Variance Threshold')
plt.xlabel("Principal Component Index")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance Ratio (First 15 Components)", fontsize=13, fontweight='bold', pad=10)
plt.legend()
plt.tight_layout()
plt.savefig('outputs/graphs/pca_variance.png', dpi=300)
plt.close()
print("Saved outputs/graphs/pca_variance.png")

# Plot 2D PCA Space colored by True Occupancy
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=15, edgecolor='none')
plt.colorbar(scatter, label='Occupancy Count (0 to 3)')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% Variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% Variance)")
plt.title("2D PCA Projection Colored by True Occupancy Count", fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('outputs/graphs/pca_2d_occupancy.png', dpi=300)
plt.close()
print("Saved outputs/graphs/pca_2d_occupancy.png")

# 9. Clustering (K-Means & Agglomerative)
print("Running Unsupervised Clustering (K-Means & Agglomerative)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

sample_sub = np.random.choice(len(X_scaled), size=2000, replace=False)
sil_kmeans = silhouette_score(X_scaled[sample_sub], kmeans_labels[sample_sub])
db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"K-Means (k=4): Silhouette Score = {sil_kmeans:.4f}, Davies-Bouldin Index = {db_kmeans:.4f}")

# Cross-tabulation of K-Means clusters vs True Occupancy
ct = pd.crosstab(y, kmeans_labels, rownames=['True Occupancy'], colnames=['K-Means Cluster'])
ct.to_csv('outputs/results/kmeans_contingency.csv')
print("K-Means vs Occupancy Contingency Matrix:")
print(ct)

# Plot K-Means Clusters in 2D PCA Space
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set2', alpha=0.6, s=15, edgecolor='none')
plt.colorbar(scatter, label='K-Means Cluster (0 to 3)')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Unsupervised Clusters (k=4) in PCA Space", fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('outputs/graphs/kmeans_clusters_2d.png', dpi=300)
plt.close()
print("Saved outputs/graphs/kmeans_clusters_2d.png")

print("Phase 4, 5, 6, 7 script completed successfully!")

