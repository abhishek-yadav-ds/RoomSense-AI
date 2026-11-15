"""
Fix and execute 03_ML_DL_Training.ipynb end-to-end.
Applies universal paths, executes all cells in-process, captures outputs and inline PNG figures,
and writes the updated notebook to disk with zero errors.
"""
import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import io
import time
import json
import base64
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import keras
from keras import layers, callbacks
import shap
import joblib

NB_PATH = 'notebooks/03_ML_DL_Training.ipynb'

print("Loading existing notebook...")
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define cell code replacements with universal paths and robust syntax
cell_1_code = """# Step 1: Import libraries and configure inline visualization with universal paths
%matplotlib inline
import os
os.environ['KERAS_BACKEND'] = 'torch'
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import keras
from keras import layers, callbacks
import shap

# Universal path resolution: Works whether executed from project root or notebooks/ folder
PROJECT_ROOT = '..' if os.path.exists('../data/Occupancy_Estimation.csv') else '.'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Occupancy_Estimation.csv')
CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_data.csv')
ENGINEERED_PATH = os.path.join(PROJECT_ROOT, 'data', 'engineered_data.csv')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import feature_utils

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print(f"Project root resolved to: '{PROJECT_ROOT}'")
print(f"Engineered data path:     '{ENGINEERED_PATH}'")
print("Environment configured: %matplotlib inline active, Keras backend initialized!")
"""

cell_3_code = """# Step 2: Load engineered data, select top features, and split
df = pd.read_csv(ENGINEERED_PATH)
y = df['Room_Occupancy_Count']

SELECTED_FEATURES = [
    'S1_Light', 'Light_Mean', 'Light_Max', 'Light_Range',
    'S5_CO2', 'S5_CO2_Slope', 'CO2_Change', 'PIR_Active_Count'
]

X = df[SELECTED_FEATURES]

# Stratified 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Top 8 Features: {SELECTED_FEATURES}")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")
"""

cell_4_code = """# Step 3: Train and evaluate traditional ML algorithms
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'K-Nearest Neighbors': (KNeighborsClassifier(n_neighbors=5), True),
    'Decision Tree': (DecisionTreeClassifier(max_depth=8, random_state=42), False),
    'Support Vector Machine': (SVC(kernel='rbf', C=10.0, random_state=42), True),
    'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1), False)
}

results = []
confusion_matrices = {}
trained_models = {}

for name, (clf, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test
    
    start_train = time.time()
    clf.fit(X_tr, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    preds = clf.predict(X_te)
    latency_ms = ((time.time() - start_pred) / len(X_te)) * 1000
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
    
    confusion_matrices[name] = confusion_matrix(y_test, preds)
    trained_models[name] = clf
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Macro_Precision': round(prec, 4),
        'Macro_Recall': round(rec, 4),
        'Macro_F1': round(f1, 4),
        'Weighted_F1': round(f1_weighted, 4),
        'Train_Time_s': round(train_time, 3),
        'Latency_ms': round(latency_ms, 4)
    })
    print(f"[{name}] Accuracy: {acc*100:.2f}% | Macro F1: {f1:.4f} | Latency: {latency_ms:.3f} ms")
"""

cell_6_code = """# Step 4: Build, train, and plot learning curves for Keras Sequential MLP
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

print(f"[Deep Learning (MLP)] Accuracy: {mlp_acc*100:.2f}% | Macro F1: {mlp_f1:.4f} | Latency: {mlp_latency:.3f} ms")

# Inline Plot: MLP Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2B5B84', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#F1A208', linewidth=2, linestyle='--')
axes[0].set_title("MLP Model Accuracy across Epochs", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Epoch", fontsize=10)
axes[0].set_ylabel("Accuracy", fontsize=10)
axes[0].legend()

axes[1].plot(history.history['loss'], label='Train Loss', color='#D8315B', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val Loss', color='#3E92CC', linewidth=2, linestyle='--')
axes[1].set_title("MLP Model Loss across Epochs", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Epoch", fontsize=10)
axes[1].set_ylabel("Loss", fontsize=10)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'mlp_training_history.png'), dpi=300, bbox_inches='tight')
plt.show()
"""

cell_8_code = """# Step 5: Overall Model Comparison Table and Visualizations
df_results = pd.DataFrame(results)
display(df_results)
df_results.to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

# 1. Comparative Bar Chart
plt.figure(figsize=(10, 4.5))
x_idx = np.arange(len(df_results))
w = 0.35
plt.bar(x_idx - w/2, df_results['Accuracy'], w, label='Accuracy', color='#2B5B84', edgecolor='black')
plt.bar(x_idx + w/2, df_results['Macro_F1'], w, label='Macro F1-Score', color='#3E92CC', edgecolor='black')
plt.xticks(x_idx, df_results['Model'], rotation=15, ha='right', fontsize=9.5)
plt.ylabel("Score", fontsize=11)
plt.ylim(0.90, 1.02)
plt.title("Comparative Performance: Machine Learning vs Deep Learning", fontsize=12, fontweight='bold', pad=10)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'model_comparison_bar.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Multi-Class Confusion Matrices (2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for idx, (m_name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(m_name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Predicted Occupancy", fontsize=10)
    axes[idx].set_ylabel("Actual Occupancy", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'all_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.show()
"""

cell_11_code = """# Step 6: SHAP Global Feature Importance
rf_best = trained_models['Random Forest']
explainer = shap.TreeExplainer(rf_best)
sample_X = X_test.iloc[:200]
shap_values = explainer.shap_values(sample_X)

if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    importance = np.mean(mean_shap, axis=0)
elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:
        importance = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        importance = np.mean(np.abs(shap_values), axis=0)
else:
    importance = np.zeros(len(SELECTED_FEATURES))

shap_importance = pd.Series(importance, index=SELECTED_FEATURES).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
shap_importance.sort_values().plot(kind='barh', color='#2B5B84', edgecolor='black')
plt.title("SHAP Global Feature Importance (Mean |SHAP| Across Occupancy Levels)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Mean |SHAP Value| (Impact on Model Decision)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'shap_summary_bar.png'), dpi=300, bbox_inches='tight')
plt.show()
"""

cell_13_code = """# Step 7: Serialize model artifacts and verify sample inference
joblib.dump(trained_models['Random Forest'], os.path.join(MODELS_DIR, 'best_random_forest.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
mlp_model.save(os.path.join(MODELS_DIR, 'mlp_model.keras'))

# Save model metadata
metadata = {
    'champion_model': 'Random Forest',
    'selected_features': SELECTED_FEATURES,
    'accuracy': round(accuracy_score(y_test, trained_models['Random Forest'].predict(X_test)), 4),
    'macro_f1': round(f1_score(y_test, trained_models['Random Forest'].predict(X_test), average='macro'), 4)
}
with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Verify reloading
loaded_rf = joblib.load(os.path.join(MODELS_DIR, 'best_random_forest.joblib'))
sample_preds = loaded_rf.predict(X_test.iloc[:5])
print("Sample ground truth:      ", y_test.iloc[:5].tolist())
print("Reloaded model predictions:", sample_preds.tolist())
assert (sample_preds == trained_models['Random Forest'].predict(X_test.iloc[:5])).all(), "Predictions match perfectly!"
print("Model pipeline verified successfully!")
"""

def split_lines(s):
    lines = s.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

nb['cells'][1]['source'] = split_lines(cell_1_code)
nb['cells'][3]['source'] = split_lines(cell_3_code)
nb['cells'][4]['source'] = split_lines(cell_4_code)
nb['cells'][6]['source'] = split_lines(cell_6_code)
nb['cells'][8]['source'] = split_lines(cell_8_code)
nb['cells'][11]['source'] = split_lines(cell_11_code)
nb['cells'][13]['source'] = split_lines(cell_13_code)

print("Executing all code cells in-process...")
exec_globals = {
    '__name__': '__main__',
    'display': lambda x: None
}

captured_figures = []
original_show = plt.show

def custom_show(*args, **kwargs):
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    captured_figures.append(b64)
    plt.close(fig)

plt.show = custom_show

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    code = ''.join(cell['source'])
    # Filter magic commands
    clean_lines = [l for l in code.split('\n') if not l.strip().startswith('%')]
    clean_code = '\n'.join(clean_lines)
    
    print(f"Running Cell {idx}...")
    stdout_buf = io.StringIO()
    captured_figures.clear()
    
    start_time = time.time()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stdout_buf):
            exec(clean_code, exec_globals)
        elapsed = time.time() - start_time
        print(f"  -> Done in {elapsed:.2f}s")
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"  -> ERROR in Cell {idx}: {e}")
        print(err_msg)
        sys.exit(1)
        
    out_text = stdout_buf.getvalue()
    cell['outputs'] = []
    
    if out_text.strip():
        cell['outputs'].append({
            'name': 'stdout',
            'output_type': 'stream',
            'text': [l + '\n' for l in out_text.split('\n') if l]
        })
        
    for fig_b64 in captured_figures:
        cell['outputs'].append({
            'data': {
                'image/png': fig_b64,
                'text/plain': ['<Figure output>']
            },
            'metadata': {},
            'output_type': 'display_data'
        })
        
    cell['execution_count'] = idx

plt.show = original_show

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 03 successfully executed, rendered, and saved!")

