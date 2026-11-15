# RoomSense AI: An Explainable Feature-Optimized Room Occupancy Estimation System Using Machine Learning and Deep Learning

**A Major Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Technology (B.Tech)**

---

## Abstract

Energy-efficient heating, ventilation, and air conditioning (HVAC) as well as automated lighting controls in modern smart buildings rely critically on accurate, real-time room occupancy detection. Traditional methods relying on computer vision cameras compromise occupant privacy, consume substantial network bandwidth, and face stringent legal barriers. Conversely, simple motion sensors fail to detect stationary occupants. 

In this project, we design and implement **RoomSense AI**, a privacy-preserving occupancy estimation framework utilizing multi-modal non-intrusive environmental sensors (Temperature, Light, Sound, CO2, and Passive Infrared Motion) from the benchmark UCI Room Occupancy Estimation dataset (ID: 864, 10,129 observations). We engineer 38 domain features capturing spatial cross-sensor variance, human metabolic CO2 rates of change ($d[CO_2]/dt$), motion dynamics, and cyclical diurnal time signatures. 

Using multi-method feature selection (Mutual Information, ANOVA F-test, RFE, and Random Forest Gini importance), we isolate an optimal 8-feature representation. We benchmark five classical Machine Learning models alongside a Keras Deep Learning Multi-Layer Perceptron (MLP). **Random Forest** achieved superior performance with **99.56% Accuracy**, **98.53% Macro F1-Score**, and **0.02 ms inference latency per sample**. Crucially, our physical sensor group optimization demonstrates that a minimal hardware configuration of **CO2 + PIR (3 physical sensors)** achieves **95.42% Macro F1-Score**, proving that commercial buildings can drastically reduce installation costs without sacrificing estimation accuracy. The system is equipped with **Explainable AI (SHAP)** interpretability and deployed via an interactive 10-page Streamlit web dashboard.

---

## Chapter 1: Introduction

### 1.1 Background & Motivation
Buildings account for approximately 40% of global primary energy consumption, with HVAC and artificial lighting constituting the vast majority of commercial operational costs. Traditional systems operate on rigid static schedules, heating and ventilating empty spaces or maintaining uniform airflow regardless of whether a room hosts one or twenty occupants. Dynamic, occupancy-driven demand-controlled ventilation (DCV) can yield 20% to 50% energy reductions.

### 1.2 Limitations of Conventional Occupancy Sensing
1. **Vision-Based Systems (CCTV & RGB Cameras):** While capable of high accuracy, optical cameras raise acute privacy, ethical, and legal violations, and struggle in variable lighting conditions.
2. **Wearable & Token-Based Systems (RFID, BLE Beacons):** Require active compliance, carrying physical badges, and regular battery replacements.
3. **Passive Infrared (PIR) Sensors:** Detect only transient thermal motion. When occupants are seated, studying, or typing, PIR sensors erroneously report zero occupancy.

### 1.3 Project Objectives
1. Formulate a non-intrusive, privacy-compliant multi-modal sensor pipeline.
2. Formulate domain feature engineering capturing spatial sensor variance and respiration dynamics.
3. Conduct multi-method feature selection to eliminate redundant telemetry channels.
4. Investigate physical sensor group reduction (minimal hardware feasibility).
5. Benchmark 5 Machine Learning algorithms and 1 Keras Deep Learning MLP.
6. Provide explainability via game-theoretic SHAP values.
7. Construct an interactive production dashboard for real-time inference and simulation.

---

## Chapter 2: Literature Review & Related Work

Recent literature in pervasive computing highlights the utility of environmental proxies for human occupancy:
- *Respiration Modeling:* Human exhalation typically contains 40,000 ppm CO2, contrasting with outdoor baselines of 400–420 ppm. As occupants enter a room, ambient CO2 concentration follows a first-order differential accumulation curve.
- *Multi-Modal Fusion:* Combining fast-acting sensors (Light and PIR) with slow-acting accumulative sensors (CO2 and Temperature) provides complementary temporal coverage.
- *Tabular Learning Paradigms:* Studies consistently demonstrate that on tabular sensor telemetry with clear threshold boundaries, decision tree ensembles (Random Forests, Gradient Boosting) often match or outperform complex deep neural networks.

---

## Chapter 3: Dataset Description & Preprocessing

### 3.1 Dataset Overview
- **Dataset:** UCI Room Occupancy Estimation Dataset (Dataset ID: 864).
- **Duration & Frequency:** 10,129 records sampled at 30-second intervals across 7 distinct days in a 6m × 4.6m room.
- **Target Variable:** `Room_Occupancy_Count` (Classes: 0 = Empty, 1 = 1 Occupant, 2 = 2 Occupants, 3 = 3 Occupants).
- **Class Breakdown:** Class 0: 8,228 (81.2%), Class 1: 459 (4.5%), Class 2: 748 (7.4%), Class 3: 694 (6.9%).

### 3.2 Preprocessing & Outlier Treatment
1. **Missing Data & Integrity:** Zero missing entries detected. Timestamps validated and chronologically sorted.
2. **Outlier Analysis:** Compared Interquartile Range (IQR), Z-score ($|z| > 3.0$), and Isolation Forest ($3\%$ contamination).
   - *Key Finding:* Extreme CO2 readings (up to 1,270 ppm) and high illumination levels correspond directly to multi-person occupancy events rather than faulty hardware. Blind deletion of statistical outliers was discarded in favor of RobustScaler and tree-based modeling.
3. **Statistical Hypothesis Testing:** One-Way ANOVA and Kruskal-Wallis non-parametric tests on CO2, Light, Temp, and Sound produced $p < 0.001$, confirming statistically significant distribution shifts across occupancy levels.

---

## Chapter 4: Feature Engineering & Selection

### 4.1 Feature Engineering Pipeline (38 Features)
1. **Spatial Aggregations:** For multi-node sensor clusters (4 Temperature, 4 Light, 4 Sound sensors):
   $$\text{Mean} = \frac{1}{4} \sum_{i=1}^4 S_i, \quad \text{Range} = \max(S_i) - \min(S_i), \quad \text{Std} = \sqrt{\frac{1}{4}\sum(S_i - \text{Mean})^2}$$
2. **Motion Dynamics:** $\text{PIR\_Active\_Count} = S6\_PIR + S7\_PIR$.
3. **Metabolic Respiration:** CO2 slope ($d[CO_2]/dt$), rolling 5-step mean, and rolling standard deviation.
4. **Diurnal Cyclical Encoding:** $\text{Hour}_{\sin} = \sin(2\pi \cdot \text{Hour} / 24)$, $\text{Hour}_{\cos} = \cos(2\pi \cdot \text{Hour} / 24)$.

### 4.2 Feature Selection Results
- **Filter Methods:** Mutual Information highlighted `Light_Mean` (0.58) and `S5_CO2` (0.49).
- **Wrapper Methods:** Recursive Feature Elimination (RFE) isolated 8 core multi-modal features.
- **Embedded Methods:** Random Forest Gini importance confirmed `S1_Light`, `Light_Mean`, `S5_CO2`, and `PIR_Active_Count` as dominant drivers.
- **Feature Count Study:** Cross-validation Macro F1 saturated at **8 features (98.13%)**, allowing 30 features to be eliminated.

---

## Chapter 5: Model Development & Experimental Results

### 5.1 Model Benchmark Leaderboard
Evaluated on a held-out $20\%$ test split (2,026 samples) stratified by class:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency |
|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.9956** | **0.9873** | **0.9839** | **0.9853** | **0.021 ms** |
| **Deep Learning (Keras MLP)** | 0.9951 | 0.9887 | 0.9817 | 0.9852 | 0.071 ms |
| **Support Vector Machine (RBF)** | 0.9936 | 0.9794 | 0.9786 | 0.9789 | 0.027 ms |
| **Decision Tree** | 0.9931 | 0.9773 | 0.9752 | 0.9758 | 0.001 ms |
| **K-Nearest Neighbors** | 0.9926 | 0.9766 | 0.9734 | 0.9745 | 0.024 ms |
| **Logistic Regression** | 0.9901 | 0.9774 | 0.9654 | 0.9712 | 0.0003 ms |

### 5.2 Physical Sensor Group Optimization
| Hardware Configuration | Sensors / Features | Accuracy | Macro F1 | Practical Assessment |
|---|---|---|---|---|
| All Sensors (16 Raw) | 16 Sensors | 0.9967 | 0.9890 | Full experimental setup |
| Top 8 Engineered | 8 Features | 0.9956 | 0.9853 | Optimal software feature selection |
| Temp + CO2 + PIR | 7 Sensors | 0.9945 | 0.9810 | Eliminates microphones/light |
| **CO2 + PIR (Minimal)** | **3 Sensors** | **0.9885** | **0.9542** | **Recommended: Privacy-safe, low cost** |
| Light + Sound + PIR | 10 Sensors | 0.9820 | 0.9410 | No CO2 calibration needed |

---

## Chapter 6: Explainable AI & System Deployment

### 6.1 Explainable AI with SHAP
SHAP TreeExplainer attributions revealed:
- **Light:** Determines the coarse boundary between unoccupied and occupied room states.
- **CO2 & CO2 Slope:** Differentiates fine-grained occupant density (1 vs 2 vs 3 occupants).
- **PIR Motion:** Dampens false positives during room exit when CO2 dissipation lags.

### 6.2 Streamlit Web Application
A multi-page production interface provides real-time single-sample inference, probability distributions, historical replay with anomaly alerts, and what-if simulation.

---

## Chapter 7: Conclusion & Future Scope

### 7.1 Conclusion
RoomSense AI successfully demonstrates that non-intrusive ambient environmental telemetry delivers high-accuracy room occupancy counting (>98.5% Macro F1), matching the performance of invasive optical systems while guaranteeing total privacy. Hardware optimization confirms that a minimal 3-sensor deployment (CO2 + PIR) provides commercially viable accuracy (>95% Macro F1).

### 7.2 Future Scope
1. Porting the pipeline to low-power microcontrollers (ESP32 / ARM Cortex-M) via TinyML.
2. Developing domain adaptation models to compensate for active window ventilation and multi-room variations.

---

## References
1. UCI Machine Learning Repository: Room Occupancy Estimation Dataset (ID: 864).
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
4. Chollet, F., et al. (2023). Keras 3: Deep Learning for Humans.

