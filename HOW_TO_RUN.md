# RoomSense AI — Step-by-Step Execution Guide & Reference

This guide details how to execute every individual script, notebook, and tool in the **RoomSense AI** project.

---

## 📌 Recommended Execution Pipeline

To run the complete data science pipeline from scratch, execute the scripts in the following chronological sequence:

```text
Step 1: Data Cleaning & Exploratory Data Analysis
   ↓  (outputs data/cleaned_data.csv and EDA graphs)
Step 2: Feature Engineering & Sensor Optimization
   ↓  (outputs data/engineered_data.csv and selection plots)
Step 3: Machine Learning & Deep Learning Training
   ↓  (outputs models/best_random_forest.joblib, models/mlp_model.keras)
Step 4: Interactive Web Dashboard
      (launches Streamlit application on http://localhost:8501)
```

---

## 💻 1. Executing Python Scripts via Command Line

Run each command from the root project directory (`roomsences ai`):

### **Step 1: Data Preprocessing & Outlier Diagnostics**
```bash
python run_01_eda.py
```
*Outputs: Cleans UCI data, creates `data/cleaned_data.csv`, and generates 8 EDA visualization graphs in `outputs/graphs/`.*

### **Step 2: Feature Engineering & Selection**
```bash
python run_02_features.py
```
*Outputs: Calculates 22 domain features via `feature_utils.py`, creates `data/engineered_data.csv`, and generates Mutual Information and PCA plots.*

### **Step 3: Machine Learning & Deep Learning Training**
```bash
python run_03_training.py
```
*Outputs: Benchmarks 5 ML algorithms + Keras Sequential MLP, saves `models/best_random_forest.joblib`, `models/mlp_model.keras`, and generates comparative bar chart and confusion matrices.*
```
- **What it does:**
  - Ingests `data/Occupancy_Estimation.csv` (10,129 rows).
  - Performs outlier diagnostics using IQR, Z-score ($|z| > 3$), and Isolation Forest.
  - Executes One-Way ANOVA and Kruskal-Wallis statistical hypothesis tests.
  - Exports 8 high-resolution charts to `outputs/graphs/`.
  - Saves the verified preprocessed data to `data/cleaned_data.csv`.

---

### **Step 2: Feature Engineering & Physical Sensor Optimization**
```bash
python run_phase2_features.py
```
- **What it does:**
  - Expands 16 raw features into 38 domain-aware spatial and temporal features.
  - Runs Mutual Information score ranking and Random Forest Gini importance.
  - Benchmarks 5 physical hardware configurations (All 16 vs 7 vs 3 vs 10 sensors).
  - Performs PCA dimensionality reduction and K-Means clustering ($k=4$).
  - Saves the full feature matrix to `data/engineered_data.csv`.

---

### **Step 3: ML & Deep Learning Model Training**
```bash
python run_phase3_ml_dl.py
```
- **What it does:**
  - Splits data into stratified 80% train / 20% test sets.
  - Trains 5 traditional ML models: Logistic Regression, KNN, Decision Tree, SVM, and Random Forest.
  - Builds and trains a Keras Sequential Deep Learning MLP with Dropout regularization.
  - Computes game-theoretic SHAP feature attributions using `shap.TreeExplainer`.
  - Serializes the champion model to `models/best_random_forest.joblib` and `models/mlp_model.keras`.
  - Generates multi-class confusion matrices and model comparison tables.

---

### **Step 4: Launching the Interactive Web Dashboard**
```bash
streamlit run app.py
```
- **What it does:**
  - Starts the local Streamlit web server on port 8501.
  - Automatically opens the modern dark purple SaaS UI in your browser at `http://localhost:8501`.
  - *Windows Shortcut:* Double-click `run_dashboard.bat` to launch without opening terminal.

---

## 📓 2. Executing Jupyter Notebooks (.ipynb)

All notebooks are pre-executed with all cell outputs and 19 inline PNG figures embedded. To re-run them:

### **Option A: Inside VS Code / Cursor / Antigravity IDE**
1. Open the `notebooks/` folder in your file explorer.
2. Click any notebook:
   - `notebooks/01_EDA_Preprocessing.ipynb`
   - `notebooks/02_Feature_Engineering_Selection.ipynb`
   - `notebooks/03_ML_DL_Training.ipynb`
3. Click **"Run All"** at the top toolbar, or press **Shift + Enter** on individual cells.
4. Plots and tables will render immediately below each code cell.

### **Option B: Inside Browser via Jupyter Notebook**
```bash
jupyter notebook
```
Navigate to `notebooks/` and select **Kernel -> Restart & Run All**.

---

## 🛠️ 3. Utility & Helper Scripts

### **A. Automated Test Suite (20 Health Checks)**
```bash
python test_project.py
```
Validates dataset integrity, model serialization, notebook cell outputs, and directory structures.

### **B. Instant Terminal Prediction Demo**
```bash
python predict_cli.py
```
Runs instant occupancy predictions for 4 pre-packaged room scenarios directly from terminal without a browser.

### **C. College Viva Practice Simulator**
```bash
python viva_quiz.py
```
An interactive terminal quiz that tests you on key academic questions (Macro F1, Outliers, SHAP, Sensor Optimization) with detailed model explanations.

### **D. Generate Clean Project ZIP for College Submission**
```bash
python export_project_zip.py
```
Creates `RoomSense_AI_Submission.zip` (7.28 MB) excluding bytecode and temporary files.

---

## ❓ Troubleshooting & FAQs

1. **Port 8501 is already in use:**
   Run on a different port:
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Keras/TensorFlow backend warning:**
   The project sets `os.environ['KERAS_BACKEND'] = 'torch'`, ensuring native CPU compatibility on Python 3.11, 3.12, and 3.13 without CUDA errors.

3. **Matplotlib GUI blocking:**
   All execution scripts use `matplotlib.use('Agg')` or `%matplotlib inline`, preventing GUI window freezes.

