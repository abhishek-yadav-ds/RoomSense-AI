"""
RoomSense AI - Phase 8 to 13 Execution Script
ML & DL Training, Evaluation, SHAP Explainability, Anomaly Experiments, and Model Saving.
"""

import os
os.environ['KERAS_BACKEND'] = 'torch'
import time
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import keras
from keras import layers, callbacks
import shap

import feature_utils

os.makedirs('outputs/graphs', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Libraries imported successfully!")

# 1. Load Engineered Data
data_path = 'data/engineered_data.csv'
if not os.path.exists(data_path):
    print("Generating engineered dataset...")
    raw_df = pd.read_csv('data/cleaned_data.csv')
    X_all = feature_utils.engineer_all_features(raw_df, include_raw=True)
    y_all = raw_df['Room_Occupancy_Count']
    full_df = pd.concat([X_all, y_all], axis=1)
    full_df.to_csv(data_path, index=False)
else:
    full_df = pd.read_csv(data_path)

y = full_df['Room_Occupancy_Count']
X_all = full_df.drop(columns=['Room_Occupancy_Count'])

# Top 8 Selected Features (from Phase 5 optimization)
SELECTED_FEATURES = [
    'S1_Light', 'Light_Mean', 'Light_Max', 'Light_Range',
    'S5_CO2', 'S5_CO2_Slope', 'CO2_Change', 'PIR_Active_Count'
]

# Check if selected features exist in X_all, if not use available subset
valid_selected = [c for c in SELECTED_FEATURES if c in X_all.columns]
if len(valid_selected) < 5:
    valid_selected = X_all.columns[:8].tolist()

print(f"Selected Top Features ({len(valid_selected)}):", valid_selected)

# 2. Train-Test Split (Stratified to maintain class ratios, 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_all[valid_selected], y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# 3. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler and Metadata
joblib.dump(scaler, 'models/scaler.joblib')
metadata = {
    'selected_features': valid_selected,
    'classes': [0, 1, 2, 3],
    'class_names': ['Empty', '1 Person', '2 People', '3 People'],
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("Saved models/scaler.joblib and models/model_metadata.json")

# 4. Phase 8: Train 5 Traditional ML Models
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'K-Nearest Neighbors': (KNeighborsClassifier(n_neighbors=5), True),
    'Decision Tree': (DecisionTreeClassifier(max_depth=8, random_state=42), False),
    'Support Vector Machine': (SVC(kernel='rbf', C=10.0, probability=True, random_state=42), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1), False)
}

results = []
confusion_matrices = {}
trained_models = {}

print("\n--- Training Traditional ML Algorithms ---")
for name, (clf, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test
    
    start_train = time.time()
    clf.fit(X_tr, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    preds = clf.predict(X_te)
    pred_time = ((time.time() - start_pred) / len(X_te)) * 1000  # ms per sample
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, preds)
    confusion_matrices[name] = cm
    trained_models[name] = clf
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Macro_Precision': round(prec, 4),
        'Macro_Recall': round(rec, 4),
        'Macro_F1': round(f1, 4),
        'Weighted_F1': round(f1_weighted, 4),
        'Train_Time_s': round(train_time, 3),
        'Latency_ms': round(pred_time, 4)
    })
    print(f"[{name}] Acc: {acc:.4f} | Macro F1: {f1:.4f} | Latency: {pred_time:.3f}ms")

# 5. Phase 9: Deep Learning Multi-Layer Perceptron (MLP)
print("\n--- Training Keras Deep Learning MLP ---")
num_features = X_train_scaled.shape[1]
mlp_model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax')
])

mlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

start_mlp_train = time.time()
history = mlp_model.fit(
    X_train_scaled, y_train.values,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=0
)
mlp_train_time = time.time() - start_mlp_train

start_mlp_pred = time.time()
mlp_probs = mlp_model.predict(X_test_scaled, verbose=0)
mlp_preds = np.argmax(mlp_probs, axis=1)
mlp_latency = ((time.time() - start_mlp_pred) / len(X_test_scaled)) * 1000

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_prec = precision_score(y_test, mlp_preds, average='macro', zero_division=0)
mlp_rec = recall_score(y_test, mlp_preds, average='macro', zero_division=0)
mlp_f1 = f1_score(y_test, mlp_preds, average='macro', zero_division=0)
mlp_f1_weighted = f1_score(y_test, mlp_preds, average='weighted', zero_division=0)

results.append({
    'Model': 'Deep Learning (MLP)',
    'Accuracy': round(mlp_acc, 4),
    'Macro_Precision': round(mlp_prec, 4),
    'Macro_Recall': round(mlp_rec, 4),
    'Macro_F1': round(mlp_f1, 4),
    'Weighted_F1': round(mlp_f1_weighted, 4),
    'Train_Time_s': round(mlp_train_time, 3),
    'Latency_ms': round(mlp_latency, 4)
})
confusion_matrices['Deep Learning (MLP)'] = confusion_matrix(y_test, mlp_preds)
print(f"[Deep Learning (MLP)] Acc: {mlp_acc:.4f} | Macro F1: {mlp_f1:.4f} | Latency: {mlp_latency:.3f}ms")

# Save MLP Training Curves Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2B5B84', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#F1A208', linewidth=2, linestyle='--')
axes[0].set_title("MLP Model Accuracy", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss', color='#D8315B', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='#3E92CC', linewidth=2, linestyle='--')
axes[1].set_title("MLP Model Loss", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/graphs/mlp_training_history.png', dpi=300)
plt.close()
print("Saved outputs/graphs/mlp_training_history.png")

# Save Keras Model
mlp_model.save('models/mlp_model.keras')
print("Saved models/mlp_model.keras")

# 6. Save Overall Model Comparison Results
df_results = pd.DataFrame(results)
df_results.to_csv('outputs/results/model_comparison.csv', index=False)
print("\n=== Model Comparison Table ===")
print(df_results.to_string(index=False))

# Plot Model Comparison Bar Chart
plt.figure(figsize=(11, 5))
x_idx = np.arange(len(df_results))
w = 0.35
plt.bar(x_idx - w/2, df_results['Accuracy'], w, label='Accuracy', color='#2B5B84', edgecolor='black')
plt.bar(x_idx + w/2, df_results['Macro_F1'], w, label='Macro F1-Score', color='#3E92CC', edgecolor='black')
plt.xticks(x_idx, df_results['Model'], rotation=15, ha='right', fontsize=10)
plt.ylabel("Score")
plt.ylim(0.85, 1.02)
plt.title("Comparative Performance Across Machine Learning & Deep Learning Algorithms", fontsize=12, fontweight='bold', pad=10)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/graphs/model_comparison_bar.png', dpi=300)
plt.close()
print("Saved outputs/graphs/model_comparison_bar.png")

# Plot Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(m_name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Occupancy")
    axes[idx].set_ylabel("Actual Occupancy")
plt.tight_layout()
plt.savefig('outputs/graphs/all_confusion_matrices.png', dpi=300)
plt.close()
print("Saved outputs/graphs/all_confusion_matrices.png")

# 7. Phase 12: Explainable AI with SHAP
print("\n--- Computing SHAP Interpretability ---")
rf_best = trained_models['Random Forest']
explainer = shap.TreeExplainer(rf_best)
# Use a sample of 200 test points for fast and accurate SHAP analysis
sample_X = X_test.iloc[:200]
shap_values = explainer.shap_values(sample_X)

# Handle shap values format (list of arrays for multiclass or single 3D array)
if isinstance(shap_values, list):
    # Mean absolute SHAP across classes
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
else:
    mean_shap = np.mean(np.abs(shap_values), axis=-1)

shap_importance = pd.Series(np.mean(mean_shap, axis=0), index=valid_selected).sort_values(ascending=False)

plt.figure(figsize=(9, 6))
shap_importance.sort_values().plot(kind='barh', color='#2B5B84', edgecolor='black')
plt.title("SHAP Global Feature Importance (|SHAP| Across Occupancy Classes)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
plt.tight_layout()
plt.savefig('outputs/graphs/shap_summary_bar.png', dpi=300)
plt.close()
print("Saved outputs/graphs/shap_summary_bar.png")

# 8. Phase 13: Save Best Model (Random Forest & Pipeline)
print("\n--- Saving Best Model Artifacts ---")
joblib.dump(rf_best, 'models/best_random_forest.joblib')
print("Saved models/best_random_forest.joblib")

# Test reload and sample prediction
reloaded_rf = joblib.load('models/best_random_forest.joblib')
sample_pred = reloaded_rf.predict(X_test.iloc[:5])
print("Verified reloaded model sample predictions:", sample_pred.tolist())
print("Actual ground truth:", y_test.iloc[:5].tolist())

print("\nPhase 8 to 13 execution finished successfully!")

