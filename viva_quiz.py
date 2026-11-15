"""
RoomSense AI - Interactive College Viva Voce Practice Simulator
===============================================================
A self-study quiz and preparation script for final year B.Tech viva defense.
Tests the student on core ML/DL concepts, data leakage, SHAP, and sensor optimization.
"""

import sys
import time

VIVA_QUESTIONS = [
    {
        "id": 1,
        "topic": "Evaluation Metrics & Class Imbalance",
        "question": "Why is Macro F1-Score preferred over standard Accuracy for RoomSense AI?",
        "options": [
            "A) Because Macro F1 is easier to calculate than Accuracy.",
            "B) Because Class 0 (empty room) comprises 81.2% of data; a trivial model predicting '0' gets 81% accuracy without detecting occupants.",
            "C) Because Accuracy cannot be used on multi-class classification problems.",
            "D) Because Macro F1 only evaluates the majority class."
        ],
        "correct": "B",
        "explanation": "With 81.2% Class 0 observations, accuracy is deceptive. Macro F1 unweights class proportions and averages F1 across all 4 classes equally, penalizing false negatives on occupied states."
    },
    {
        "id": 2,
        "topic": "Outlier Detection vs Data Cleaning",
        "question": "Why shouldn't extreme CO2 readings (e.g. 1,270 ppm) be deleted using the IQR or Z-score rule?",
        "options": [
            "A) Because IQR is only meant for normally distributed data.",
            "B) Because high CO2 readings represent genuine physical multi-person occupancy events rather than hardware defects.",
            "C) Because Scikit-Learn crashes if rows are deleted.",
            "D) Because CO2 has no statistical variance."
        ],
        "correct": "B",
        "explanation": "When 3 people occupy a room, their respiration raises CO2 to 1,000-1,270 ppm. Deleting these statistical outliers removes the exact signal needed to predict multi-occupant presence."
    },
    {
        "id": 3,
        "topic": "Feature Engineering & Respiration Dynamics",
        "question": "Why did we engineer both CO2_Slope (d[CO2]/dt) and PIR_Active_Count?",
        "options": [
            "A) Because CO2 has an accumulation lag when people leave, so PIR motion confirms immediate physical presence.",
            "B) Because PIR sensors can measure temperature.",
            "C) Because CO2 slope is always negative.",
            "D) Because machine learning cannot train on raw numbers."
        ],
        "correct": "A",
        "explanation": "CO2 accumulates and dissipates slowly (thermal and fluid lag). PIR detects instantaneous movement. Combining both prevents false alarms when residual CO2 slowly clears after people leave."
    },
    {
        "id": 4,
        "topic": "Physical Sensor Group Optimization",
        "question": "What was the main finding of the Physical Sensor Optimization experiment?",
        "options": [
            "A) All 16 sensors are strictly required to exceed 70% accuracy.",
            "B) Using only 3 sensors (CO2 + PIR) achieves over 95% Macro F1-score, cutting hardware costs and preserving privacy.",
            "C) Temperature sensors are the only sensor needed.",
            "D) Microphones and CCTV cameras must be installed for reliable sensing."
        ],
        "correct": "B",
        "explanation": "Our benchmark showed that a minimal CO2 + PIR configuration achieves 95.42% Macro F1, proving smart buildings can dispense with expensive light meters and invasive audio microphones."
    },
    {
        "id": 5,
        "topic": "Machine Learning vs Deep Learning",
        "question": "Why did Random Forest (99.56% Acc) outperform or match Deep Learning MLP (99.51% Acc)?",
        "options": [
            "A) Because Deep Learning cannot be used for classification.",
            "B) Because environmental sensor data is tabular with orthogonal decision thresholds that tree ensembles segment naturally.",
            "C) Because Keras does not support softmax activation.",
            "D) Because neural networks always overfit on tabular datasets."
        ],
        "correct": "B",
        "explanation": "Tabular sensor data has sharp orthogonal boundaries (e.g. CO2 > 600 ppm AND Light > 50 Lux). Decision tree ensembles naturally segment orthogonal partitions without requiring massive sample sizes."
    },
    {
        "id": 6,
        "topic": "Explainable AI (XAI) with SHAP",
        "question": "What is the primary advantage of SHAP values over standard Random Forest Gini importance?",
        "options": [
            "A) SHAP values run 100 times faster than Gini importance.",
            "B) SHAP is based on cooperative game theory and provides mathematically fair local and global feature attributions without cardinality bias.",
            "C) SHAP replaces the need to train a model.",
            "D) Gini importance can only be calculated on linear models."
        ],
        "correct": "B",
        "explanation": "Gini importance (MDI) is biased toward continuous high-cardinality features. SHAP allocates fair additive credit using cooperative game theory, providing both global rankings and local sample explanations."
    }
]

def run_viva_simulator():
    print("=" * 70)
    print("        RoomSense AI — Interactive Viva Defense Simulator")
    print("=" * 70)
    print("Test your conceptual mastery before presenting to professors & examiners.\n")
    
    score = 0
    total = len(VIVA_QUESTIONS)
    
    for q in VIVA_QUESTIONS:
        print(f"\n[Question {q['id']}/{total}] — Domain: {q['topic']}")
        print(f"Q: {q['question']}")
        for opt in q['options']:
            print(f"   {opt}")
            
        ans = input("\nYour Answer (A/B/C/D) or press Enter to reveal model answer: ").strip().upper()
        
        if ans == q['correct']:
            print(f"Correct! Well done.")
            score += 1
        elif ans in ["A", "B", "C", "D"]:
            print(f"Incorrect. The correct answer is ({q['correct']}).")
        else:
            print(f"Answer Key: ({q['correct']})")
            
        print(f"Academic Explanation: {q['explanation']}")
        print("-" * 70)
        time.sleep(0.5)
        
    print("\n" + "=" * 70)
    print(f"VIVA SIMULATION COMPLETE: Score {score}/{total} ({(score/total)*100:.1f}%)")
    print("=" * 70)
    if score == total:
        print("Outstanding! You are 100% prepared to defend RoomSense AI in your viva!")
    else:
        print("Good effort! Review the questions above and in 'outputs/reports/final_project_report.md'.")

if __name__ == '__main__':
    run_viva_simulator()

