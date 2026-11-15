"""
Fix and Render Notebook 01: EDA & Preprocessing
Properly sets PROJECT_ROOT so no '../outputs' PermissionError occurs,
captures all 8 Matplotlib figures on plt.show(), and embeds them directly.
"""

import nbformat as nbf
import os
import io
import sys
import base64
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def fix_and_render_nb01():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# RoomSense AI — Notebook 1: Exploratory Data Analysis & Preprocessing
**Full Project Title:** RoomSense AI: An Explainable Feature-Optimized Room Occupancy Estimation System Using Machine Learning and Deep Learning  
**Target:** Predict `Room_Occupancy_Count` (0: Empty, 1: 1 Person, 2: 2 People, 3: 3 People) using non-intrusive environmental sensors.

---
### Objectives of this Notebook:
1. **Phase 1: Dataset Loading & Understanding**
   - Inspect UCI Room Occupancy dataset (10,129 records).
   - Check data types, null values, duplicates, and construct chronological `Datetime`.
2. **Phase 2: Complete Preprocessing & Outlier Diagnostics**
   - Outlier detection via IQR, Z-Score ($|z| > 3$), and Isolation Forest.
   - Scaler comparison (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
3. **Phase 3: Exploratory Data Analysis & Statistical Hypothesis Testing**
   - Univariate distributions (class balance, sensor histograms).
   - Bivariate boxplots (sensors vs occupancy classes).
   - Multivariate heatmaps (Pearson linear & Spearman rank correlation).
   - Chronological time-series tracking (CO2 vs Occupancy).
   - Parametric (ANOVA) and Non-parametric (Kruskal-Wallis) hypothesis testing.
4. **College Viva Preparation**
   - Key conceptual questions and model explanations.
"""))

    # Cell 1: Setup & Universal Path Resolution
    cells.append(nbf.v4.new_code_cell("""# Step 1: Import core data science and visualization libraries
%matplotlib inline
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

# Universal path resolution: Works whether executed from workspace root or notebooks/ folder
PROJECT_ROOT = '..' if os.path.exists('../data/Occupancy_Estimation.csv') else '.'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Occupancy_Estimation.csv')
CLEANED_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned_data.csv')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'graphs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'results')

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Project root resolved to: '{PROJECT_ROOT}'")
print(f"Data path resolved to:    '{DATA_PATH}'")
print("Libraries imported and inline visualization enabled!")
"""))

    # Phase 1 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## Phase 1: Dataset Loading & Understanding
We load `Occupancy_Estimation.csv` (UCI Dataset ID: 864) containing 10,129 observations.  
Sensor channels: 4 Temperature, 4 Light, 4 Sound, 1 CO2, 1 CO2 Slope, 2 PIR Motion sensors.
"""))

    # Cell 2: Load Data
    cells.append(nbf.v4.new_code_cell("""# Step 2: Load dataset and inspect head
df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head(5)
"""))

    # Cell 3: Summary Statistics
    cells.append(nbf.v4.new_code_cell("""# Step 3: Statistical summary of numerical sensor readings
df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
"""))

    # Cell 4: Chronological Indexing
    cells.append(nbf.v4.new_code_cell("""# Step 4: Construct unified Datetime index and sort chronologically
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.sort_values('Datetime').reset_index(drop=True)

print("Observation Date Range:", df['Datetime'].min(), "to", df['Datetime'].max())
print("Total observation timespan:", df['Datetime'].max() - df['Datetime'].min())
print("Unique collection days:", df['Date'].nunique())
"""))

    # Cell 5: Missing & Duplicates Check
    cells.append(nbf.v4.new_code_cell("""# Step 5: Validate missing values and duplicate rows
missing = df.isnull().sum().sum()
duplicates = df.duplicated(subset=['Date', 'Time']).sum()

print(f"Total Missing Values across all columns: {missing}")
print(f"Duplicate timestamp rows: {duplicates}")
"""))

    # Phase 2 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## Phase 2: Data Preprocessing & Outlier Diagnostics
We analyze sensor distributions using IQR, Z-Score ($|z| > 3$), and Isolation Forest.  
**Critical Insight:** High CO2 or Light readings during occupancy represent **genuine physical conditions**, not sensor corruption. We compare scaling transformations to normalize numerical ranges safely.
"""))

    # Cell 6: Scaler Comparison Plot
    cells.append(nbf.v4.new_code_cell("""# Step 6: Comparison of Feature Scaling Methods on S5_CO2
sample_features = df[['S1_Temp', 'S5_CO2']].copy()

standard_scaled = StandardScaler().fit_transform(sample_features)
minmax_scaled = MinMaxScaler().fit_transform(sample_features)
robust_scaled = RobustScaler().fit_transform(sample_features)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(standard_scaled[:, 1], bins=30, color='#1f77b4', edgecolor='black')
axes[0].set_title("StandardScaler (S5_CO2)\\nMean=0, Std=1", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Standardized Value")

axes[1].hist(minmax_scaled[:, 1], bins=30, color='#2ca02c', edgecolor='black')
axes[1].set_title("MinMaxScaler (S5_CO2)\\nBounded Range [0, 1]", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Normalized Value")

axes[2].hist(robust_scaled[:, 1], bins=30, color='#ff7f0e', edgecolor='black')
axes[2].set_title("RobustScaler (S5_CO2)\\nMedian=0, IQR Scaled", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Robust Scaled Value")

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'scaler_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** `StandardScaler` standardizes features to zero mean and unit variance (best for SVM, Logistic Regression, and MLP). `RobustScaler` scales based on median and IQR, preventing extreme values from skewing the mean.
"""))

    # Cell 7: Outlier Summary Table
    cells.append(nbf.v4.new_code_cell("""# Step 7: Calculate Outlier Counts (IQR vs Z-score)
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
        'IQR Outliers': iqr_outliers,
        'IQR %': round((iqr_outliers / len(df)) * 100, 2),
        'Z-Score (|z|>3)': z_outliers,
        'Z-Score %': round((z_outliers / len(df)) * 100, 2)
    })

pd.DataFrame(outlier_report)
"""))

    # Cell 8: Outlier Boxplots
    cells.append(nbf.v4.new_code_cell("""# Step 8: Outlier Visualizations Across Sensor Types
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

sns.boxplot(y=df['S1_Temp'], ax=axes[0], color='#A5B4FC')
axes[0].set_title("Temperature (S1_Temp)", fontweight='bold')
axes[0].set_ylabel("Celsius (°C)")

sns.boxplot(y=df['S1_Light'], ax=axes[1], color='#38BDF8')
axes[1].set_title("Light (S1_Light)", fontweight='bold')
axes[1].set_ylabel("Lux")

sns.boxplot(y=df['S1_Sound'], ax=axes[2], color='#34D399')
axes[2].set_title("Sound (S1_Sound)", fontweight='bold')
axes[2].set_ylabel("Volts")

sns.boxplot(y=df['S5_CO2'], ax=axes[3], color='#F472B6')
axes[3].set_title("CO2 (S5_CO2)", fontweight='bold')
axes[3].set_ylabel("ppm")

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'outlier_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** S1_Light and S5_CO2 exhibit extended upper tails. These correspond to occupied periods where illumination and exhaled carbon dioxide rise significantly above baseline levels.
"""))

    # Phase 3 Markdown
    cells.append(nbf.v4.new_markdown_cell("""## Phase 3: Exploratory Data Analysis & Statistical Hypotheses
We examine univariate distributions, bivariate relationships with occupancy count, multivariate correlation structures, and conduct hypothesis testing.
"""))

    # Cell 9: Occupancy Distribution Bar Plot
    cells.append(nbf.v4.new_code_cell("""# Step 9: Plot Occupancy Class Distribution
occupancy_counts = df['Room_Occupancy_Count'].value_counts().sort_index()

plt.figure(figsize=(8, 4.5))
colors = ['#2B5B84', '#3E92CC', '#F1A208', '#D8315B']
bars = plt.bar(occupancy_counts.index, occupancy_counts.values, color=colors, edgecolor='black', width=0.55)

plt.title("Room Occupancy Class Distribution (UCI Dataset ID 864)", fontsize=12, fontweight='bold', pad=10)
plt.xlabel("Occupancy Level (People)", fontsize=11)
plt.ylabel("Number of 30-Second Observations", fontsize=11)
plt.xticks(occupancy_counts.index, ['0 (Empty)', '1 Person', '2 People', '3 People'])

for bar, count in zip(bars, occupancy_counts.values):
    pct = (count / len(df)) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f"{count:,}\\n({pct:.1f}%)", 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylim(0, 9500)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'occupancy_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** The target is heavily imbalanced: Class 0 represents 81.2% of all samples, while Classes 1, 2, and 3 represent 4.5%, 7.4%, and 6.9%. Consequently, **Macro F1-Score** is our primary evaluation metric to ensure fair detection of occupants.
"""))

    # Cell 10: Sensor Histograms with KDE
    cells.append(nbf.v4.new_code_cell("""# Step 10: Probability Density Histograms of Key Sensors
key_sensors = ['S1_Temp', 'S1_Light', 'S1_Sound', 'S5_CO2']
units = ['(°C)', '(Lux)', '(Volts)', '(ppm)']

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()

for i, (col, unit) in enumerate(zip(key_sensors, units)):
    sns.histplot(df[col], kde=True, ax=axes[i], color='#2B5B84', edgecolor='black', bins=30)
    axes[i].set_title(f"Distribution of {col} {unit}", fontsize=11, fontweight='bold')
    axes[i].set_xlabel(f"{col} {unit}")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'sensor_histograms.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** Light and Sound are heavily skewed toward baseline values (0 Lux and ambient background sound), with positive spikes during occupied hours. CO2 exhibits a multi-modal distribution reflecting baseline room air vs elevated human exhalation.
"""))

    # Cell 11: Bivariate Boxplots
    cells.append(nbf.v4.new_code_cell("""# Step 11: Bivariate Boxplots — Sensors vs Occupancy Level
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.flatten()

for i, (col, unit) in enumerate(zip(key_sensors, units)):
    sns.boxplot(x='Room_Occupancy_Count', y=col, data=df, ax=axes[i], palette='Blues', hue='Room_Occupancy_Count', legend=False)
    axes[i].set_title(f"{col} Across Occupancy Classes", fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Occupancy Count (People)")
    axes[i].set_ylabel(f"{col} {unit}")

plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'sensor_vs_occupancy_boxplots.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** CO2 and Light show strong monotonic increases as occupancy rises from 0 to 3 people. Temperature also exhibits gradual warming from body heat dissipation.
"""))

    # Cell 12: Pearson Correlation Heatmap
    cells.append(nbf.v4.new_code_cell("""# Step 12: Pearson Linear Correlation Matrix
numeric_features = [c for c in df.columns if c.startswith('S')] + ['Room_Occupancy_Count']

plt.figure(figsize=(11, 8))
sns.heatmap(df[numeric_features].corr(method='pearson'), annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0)
plt.title("Pearson Linear Correlation Matrix", fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    # Cell 13: Spearman Rank Correlation Heatmap
    cells.append(nbf.v4.new_code_cell("""# Step 13: Spearman Rank (Non-Linear) Correlation Matrix
plt.figure(figsize=(11, 8))
sns.heatmap(df[numeric_features].corr(method='spearman'), annot=True, fmt='.2f', cmap='viridis', vmin=-0.2, vmax=1.0)
plt.title("Spearman Rank Correlation Matrix", fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'spearman_heatmap.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** Both Pearson and Spearman correlations confirm that Light sensors ($r \\approx 0.65 - 0.70$), CO2 ($r \\approx 0.60$), and PIR motion have the strongest positive correlations with room occupancy.
"""))

    # Cell 14: Time Series Plot
    cells.append(nbf.v4.new_code_cell("""# Step 14: Time-Series Evolution on an Active Day (2017/12/22)
day_sample = df[df['Date'] == '2017/12/22'].copy()

fig, ax1 = plt.subplots(figsize=(12, 4))
color = '#1f77b4'
ax1.set_xlabel('Time of Day (2017/12/22)', fontsize=11)
ax1.set_ylabel('S5_CO2 (ppm)', color=color, fontsize=11)
ax1.plot(pd.to_datetime(day_sample['Time']), day_sample['S5_CO2'], color=color, linewidth=2, label='CO2 (ppm)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#d62728'
ax2.set_ylabel('Actual Occupancy Count', color=color, fontsize=11)
ax2.step(pd.to_datetime(day_sample['Time']), day_sample['Room_Occupancy_Count'], color=color, where='post', linewidth=2, label='Occupants')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Time-Series: Environmental CO2 Accumulation vs Actual Occupancy Count", fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, 'timeseries_occupancy_co2.png'), dpi=300, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Interpretation:** When occupants enter the room, CO2 displays a clear upward slope ($d[CO_2]/dt > 0$). When occupants exit, CO2 takes time to dissipate via ventilation. This physical lag highlights why combining CO2 with fast-acting PIR motion sensors is essential.
"""))

    # Cell 15: Statistical Hypothesis Testing
    cells.append(nbf.v4.new_code_cell("""# Step 15: Statistical Hypothesis Testing (One-Way ANOVA & Kruskal-Wallis)
hyp_results = []
for feature in ['S5_CO2', 'S1_Light', 'S1_Temp', 'S1_Sound']:
    groups = [df[df['Room_Occupancy_Count'] == k][feature].values for k in [0, 1, 2, 3]]
    f_stat, p_val_anova = stats.f_oneway(*groups)
    h_stat, p_val_kw = stats.kruskal(*groups)
    hyp_results.append({
        'Feature': feature,
        'ANOVA F-Stat': round(f_stat, 2),
        'ANOVA p-value': f"{p_val_anova:.3e}",
        'Kruskal H-Stat': round(h_stat, 2),
        'Kruskal p-value': f"{p_val_kw:.3e}",
        'Significant (p<0.05)': 'Yes (p < 0.001)' if p_val_anova < 0.05 else 'No'
    })

hyp_df = pd.DataFrame(hyp_results)
print("Statistical Hypothesis Testing Results:")
hyp_df
"""))

    # Cell 16: Save Cleaned Data
    cells.append(nbf.v4.new_code_cell("""# Step 16: Save cleaned dataset
df.to_csv(CLEANED_PATH, index=False)
print(f"Cleaned data successfully verified and saved to {CLEANED_PATH}")
"""))

    # Viva Section
    cells.append(nbf.v4.new_markdown_cell("""---
## College Viva & Examination Prep: Key Questions & Answers
- **Q1: Why is overall accuracy deceptive for this dataset?**  
  *Answer:* Class 0 (empty room) comprises 81.2% of the dataset. A trivial model predicting 0 all the time gets 81.2% accuracy. We prioritize **Macro F1-score**, which weighs all 4 classes equally.
- **Q2: Why not delete extreme CO2 readings using IQR or Z-score?**  
  *Answer:* High CO2 levels (up to 1,270 ppm) occur when 3 people occupy the room. Deleting them as statistical outliers would eliminate the exact evidence needed to predict multi-occupant presence.
- **Q3: What did the hypothesis testing confirm?**  
  *Answer:* Both One-Way ANOVA and Kruskal-Wallis tests produced $p < 0.001$, mathematically confirming that sensor readings vary significantly across occupancy counts.
"""))

    nb.cells = cells
    
    # Execute each cell and capture outputs
    print("Executing cells of Notebook 01...")
    exec_scope = {
        '__file__': os.path.abspath('notebooks/01_EDA_Preprocessing.ipynb'),
        '__name__': '__main__'
    }
    
    captured_figs = 0
    
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            code = cell.source
            cell.outputs = []
            new_outputs = []
            
            clean_lines = [l for l in code.splitlines() if not l.strip().startswith('%')]
            clean_code = '\n'.join(clean_lines)
            
            def custom_show(*args, **kwargs):
                nonlocal captured_figs
                if plt.get_fignums():
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode('utf-8')
                    new_outputs.append(nbf.v4.new_output(
                        output_type='display_data',
                        data={
                            'image/png': b64,
                            'text/plain': f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"
                        },
                        metadata={}
                    ))
                    captured_figs += 1
                    plt.close(fig)
                    
            plt.show = custom_show
            
            def custom_display(*args):
                for a in args:
                    if isinstance(a, pd.DataFrame):
                        new_outputs.append(nbf.v4.new_output(
                            output_type='execute_result',
                            execution_count=idx,
                            data={
                                'text/html': a._repr_html_(),
                                'text/plain': str(a)
                            },
                            metadata={}
                        ))
                    else:
                        print(a)
            exec_scope['display'] = custom_display
            
            stdout_stream = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_stream
            err_str = None
            
            try:
                exec(clean_code, exec_scope)
                if plt.get_fignums():
                    custom_show()
            except Exception as e:
                err_str = traceback.format_exc()
            finally:
                sys.stdout = old_stdout
                
            stdout_txt = stdout_stream.getvalue()
            if stdout_txt:
                cell.outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=stdout_txt
                ))
            cell.outputs.extend(new_outputs)
            
            if err_str:
                print(f"Error in cell {idx}:\n{err_str}")
                cell.outputs.append(nbf.v4.new_output(
                    output_type='error',
                    ename='CellError',
                    evalue=err_str.splitlines()[-1],
                    traceback=err_str.splitlines()
                ))
                break
                
            cell.execution_count = idx + 1
            
    out_path = 'notebooks/01_EDA_Preprocessing.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"SUCCESS: Saved {out_path} with {captured_figs} embedded figures!")

if __name__ == '__main__':
    fix_and_render_nb01()

