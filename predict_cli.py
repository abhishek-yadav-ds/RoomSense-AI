"""
RoomSense AI - CLI Prediction Utility
=====================================
Quick terminal utility to test the trained model on sample sensor readings
without needing a web browser. Ideal for fast viva demonstrations.
"""

import os
import json
import joblib
import pandas as pd

def run_cli_prediction():
    print("=" * 60)
    print("      RoomSense AI — CLI Inference Engine")
    print("=" * 60)
    
    model_path = 'models/best_random_forest.joblib'
    meta_path = 'models/model_metadata.json'
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        print("Error: Trained model artifacts not found in models/ directory.")
        return

    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    features = meta['selected_features']
    class_names = {0: "0 (Empty Room)", 1: "1 Person", 2: "2 People", 3: "3 People"}
    
    print(f"Loaded Champion Model: Random Forest (100 Trees)")
    print(f"Required Features ({len(features)}): {features}\n")
    
    # Pre-packaged test scenarios
    scenarios = [
        {
            "name": "Scenario 1: Vacant Dark Room (Midnight)",
            "values": {'S1_Light': 0.0, 'Light_Mean': 0.0, 'Light_Max': 0.0, 'Light_Range': 0.0,
                       'S5_CO2': 355.0, 'S5_CO2_Slope': 0.0, 'CO2_Change': 0.0, 'PIR_Active_Count': 0}
        },
        {
            "name": "Scenario 2: Single Worker at Desk (Afternoon)",
            "values": {'S1_Light': 65.0, 'Light_Mean': 45.0, 'Light_Max': 70.0, 'Light_Range': 25.0,
                       'S5_CO2': 490.0, 'S5_CO2_Slope': 0.8, 'CO2_Change': 1.5, 'PIR_Active_Count': 1}
        },
        {
            "name": "Scenario 3: Two Occupants Working Collaboratively",
            "values": {'S1_Light': 110.0, 'Light_Mean': 95.0, 'Light_Max': 130.0, 'Light_Range': 45.0,
                       'S5_CO2': 680.0, 'S5_CO2_Slope': 1.9, 'CO2_Change': 3.2, 'PIR_Active_Count': 1}
        },
        {
            "name": "Scenario 4: Fully Occupied Meeting Room (3 People)",
            "values": {'S1_Light': 160.0, 'Light_Mean': 140.0, 'Light_Max': 195.0, 'Light_Range': 60.0,
                       'S5_CO2': 980.0, 'S5_CO2_Slope': 3.8, 'CO2_Change': 5.5, 'PIR_Active_Count': 2}
        }
    ]
    
    for sc in scenarios:
        df_sample = pd.DataFrame([sc['values']])[features]
        pred = model.predict(df_sample)[0]
        probs = model.predict_proba(df_sample)[0]
        
        print(f"▶ {sc['name']}")
        print(f"  Inputs: CO2={sc['values']['S5_CO2']} ppm, Light={sc['values']['S1_Light']} Lux, PIR={sc['values']['PIR_Active_Count']}")
        print(f"  --> Predicted Occupancy: {class_names[pred]}")
        print(f"  --> Confidence: {probs[pred]*100:.1f}%  (Class Probabilities: {[round(p, 3) for p in probs]})\n")

    print("=" * 60)
    print("CLI demonstration complete! Run 'streamlit run app.py' for interactive UI.")
    print("=" * 60)

if __name__ == '__main__':
    run_cli_prediction()

