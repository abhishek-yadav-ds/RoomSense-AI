# RoomSense AI — College Presentation & Seminar Slides Outline

Use this slide deck structure to prepare your PowerPoint presentation for your final year college evaluation.

---

### Slide 1: Title Slide
- **Project Title:** RoomSense AI: An Explainable Feature-Optimized Room Occupancy Estimation System Using Machine Learning and Deep Learning
- **Domain:** Internet of Things (IoT), Machine Learning, Deep Learning, Explainable AI (XAI)
- **Student Name:** Abhishek
- **Degree:** B.Tech Computer Science & Engineering / Data Science / AI
- **Dataset:** UCI Machine Learning Repository (ID: 864)

---

### Slide 2: Problem Statement & Motivation
- **The Problem:** Smart buildings need to regulate HVAC (heating/cooling/air quality) and lighting to reduce energy waste.
- **Flaws with Existing Systems:**
  - *CCTV / Cameras:* Extreme privacy violations, high compute and network bandwidth.
  - *RFID / BLE Tags:* Uncomfortable, require battery maintenance, low user compliance.
  - *Basic PIR Motion Detectors:* Fail when people sit stationary (reading, typing, meditating).
- **Our Vision:** Estimate discrete room headcounts (0, 1, 2, 3 people) non-intrusively using ambient environmental physics (CO2, Light, Sound, Temperature, PIR).

---

### Slide 3: Research Questions
1. Can domain feature engineering capture occupancy signals without camera or microphone eavesdropping?
2. How does classical Machine Learning compare against Deep Learning (MLP) on ambient sensor data?
3. **Core Research Question:** *Can we minimize the physical sensor count (e.g. from 16 down to 3 sensors) while maintaining >95% prediction reliability?*

---

### Slide 4: Dataset & Sensor Modalities
- **Source:** UCI Room Occupancy Estimation Dataset (10,129 observations at 30-second intervals across 7 distinct days).
- **Physical Sensors:**
  - 4 Temperature Sensors (Celsius)
  - 4 Light Sensors (Lux)
  - 4 Sound Sensors (Volts)
  - 1 CO2 Sensor (ppm) + CO2 Slope (ppm/s)
  - 2 PIR Motion Sensors
- **Target Variable:** `Room_Occupancy_Count` (Classes: 0, 1, 2, 3 occupants).
- **Class Imbalance:** 81.2% Empty room, 18.8% Occupied.

---

### Slide 5: Exploratory Data Analysis & Statistics
- **Outlier Analysis:** Compared IQR, Z-Score ($|z| > 3$), and Isolation Forest. Highlighted why extreme CO2 readings are genuine physical events, not sensor failures.
- **Statistical Hypothesis Testing:**
  - Conducted One-Way ANOVA and Kruskal-Wallis tests across all 4 occupancy classes.
  - Both tests yielded $p < 0.001$, statistically confirming significant environmental variance driven by human presence.

---

### Slide 6: Domain Feature Engineering
Expanded 16 raw channels into 38 domain features:
- **Spatial Aggregation:** Mean, Range, Max, and Std across multi-node temperature, light, and sound sensors.
- **Motion Dynamics:** Combined PIR activation count (`PIR_Active_Count`).
- **Respiration Dynamics:** $d[CO_2]/dt$ slope and rolling moving averages.
- **Cyclical Temporal Encoding:** $\sin(2\pi \cdot \text{Hour} / 24)$ and $\cos(2\pi \cdot \text{Hour} / 24)$ to represent time of day.

---

### Slide 7: Multi-Method Feature Selection
- Benchmarked 3 feature selection paradigms:
  1. *Filter Methods:* Mutual Information Score & ANOVA F-Test.
  2. *Wrapper Methods:* Recursive Feature Elimination (RFE).
  3. *Embedded Methods:* Random Forest Gini Importance (MDI).
- **Finding:** Macro F1-score saturates at **8 features** (Macro F1 = 98.13%), proving that 30 redundant features can be dropped.

---

### Slide 8: Physical Hardware Sensor Optimization
*Evaluating real-world hardware deployment feasibility:*
- **All 16 Sensors:** 99.67% Accuracy | 98.90% Macro F1
- **Top 8 Features:** 99.56% Accuracy | 98.53% Macro F1
- **Temp + CO2 + PIR (7 Sensors):** 99.45% Accuracy | 98.10% Macro F1
- **CO2 + PIR (3 Sensors — Minimal):** **98.85% Accuracy | 95.42% Macro F1**
- **Conclusion:** A 3-sensor deployment is sufficient for commercial building HVAC automation!

---

### Slide 9: Machine Learning & Deep Learning Leaderboard
- **Evaluation Set:** 2,026 samples ($20\%$ held-out test split, stratified).
- **Random Forest (Champion):** 99.56% Accuracy | 98.53% Macro F1 | 0.02 ms latency
- **Deep Learning MLP:** 99.51% Accuracy | 98.52% Macro F1 | 0.07 ms latency
- **SVM (RBF):** 99.36% Accuracy | 97.89% Macro F1
- **Decision Tree:** 99.31% Accuracy | 97.58% Macro F1
- **K-Nearest Neighbors:** 99.26% Accuracy | 97.45% Macro F1
- **Logistic Regression:** 99.01% Accuracy | 97.12% Macro F1

---

### Slide 10: Explainable AI (XAI) with SHAP
- Used **Shapley Additive Explanations (TreeExplainer)** to unpack black-box decisions.
- **Light:** Immediately distinguishes between vacant vs occupied room states.
- **CO2 & CO2 Slope:** Differentiates between 1, 2, and 3 occupants based on metabolic respiration volume.
- **PIR Motion:** Prevents false alarms when residual CO2 slowly clears after occupants depart.

---

### Slide 11: System Demonstration (Streamlit Web Dashboard)
- 10 functional modules:
  1. Real-time sensor input sliders & instant headcount prediction.
  2. Class probability distribution bars.
  3. Historical 30-minute test replay with anomaly alerts.
  4. What-If scenario simulations.
  5. Exportable experiment CSV reports.

---

### Slide 12: Conclusion & Future Scope
- **Conclusion:** RoomSense AI successfully proves that non-intrusive ambient environmental sensing can reliably predict exact room headcounts with >98.5% Macro F1, while reducing physical hardware down to 3 sensors.
- **Future Scope:**
  - Porting the trained model to an ESP32 microcontroller via TinyML.
  - Multi-room transfer learning and open-window draft compensation.

---

### Slide 13: Thank You & Viva Defense Q&A
- Questions & Discussions.

