"""
RoomSense AI - Feature Engineering & Pipeline Utilities
======================================================
This module provides clean, reusable functions for sensor feature engineering,
sensor group definitions, and prediction preprocessing.
Designed to be simple, readable, and easy to explain during a college viva.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SENSOR GROUP DEFINITIONS
# ---------------------------------------------------------------------------
RAW_TEMP_FEATURES = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']
RAW_LIGHT_FEATURES = ['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']
RAW_SOUND_FEATURES = ['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']
RAW_CO2_FEATURES = ['S5_CO2', 'S5_CO2_Slope']
RAW_PIR_FEATURES = ['S6_PIR', 'S7_PIR']

ALL_RAW_SENSOR_FEATURES = (
    RAW_TEMP_FEATURES +
    RAW_LIGHT_FEATURES +
    RAW_SOUND_FEATURES +
    RAW_CO2_FEATURES +
    RAW_PIR_FEATURES
)

# Physical Sensor Groups for Optimization Studies
SENSOR_GROUPS = {
    'All Sensors (Baseline)': ALL_RAW_SENSOR_FEATURES,
    'Temp + CO2 + PIR': RAW_TEMP_FEATURES + RAW_CO2_FEATURES + RAW_PIR_FEATURES,
    'CO2 + PIR (Minimal)': RAW_CO2_FEATURES + RAW_PIR_FEATURES,
    'Light + Sound + PIR': RAW_LIGHT_FEATURES + RAW_SOUND_FEATURES + RAW_PIR_FEATURES,
}


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING FUNCTIONS
# ---------------------------------------------------------------------------

def compute_temperature_features(df):
    """Compute aggregate temperature statistics across all 4 temperature sensors."""
    temp_df = df[RAW_TEMP_FEATURES]
    features = pd.DataFrame(index=df.index)
    features['Temp_Mean'] = temp_df.mean(axis=1)
    features['Temp_Std'] = temp_df.std(axis=1).fillna(0.0)
    features['Temp_Min'] = temp_df.min(axis=1)
    features['Temp_Max'] = temp_df.max(axis=1)
    features['Temp_Range'] = features['Temp_Max'] - features['Temp_Min']
    return features


def compute_light_features(df):
    """Compute aggregate light statistics across all 4 light sensors."""
    light_df = df[RAW_LIGHT_FEATURES]
    features = pd.DataFrame(index=df.index)
    features['Light_Mean'] = light_df.mean(axis=1)
    features['Light_Std'] = light_df.std(axis=1).fillna(0.0)
    features['Light_Max'] = light_df.max(axis=1)
    features['Light_Range'] = light_df.max(axis=1) - light_df.min(axis=1)
    return features


def compute_sound_features(df):
    """Compute aggregate sound statistics across all 4 sound sensors."""
    sound_df = df[RAW_SOUND_FEATURES]
    features = pd.DataFrame(index=df.index)
    features['Sound_Mean'] = sound_df.mean(axis=1)
    features['Sound_Std'] = sound_df.std(axis=1).fillna(0.0)
    features['Sound_Max'] = sound_df.max(axis=1)
    features['Sound_Range'] = sound_df.max(axis=1) - sound_df.min(axis=1)
    return features


def compute_motion_features(df):
    """Compute PIR motion indicators from the 2 PIR sensors."""
    pir_df = df[RAW_PIR_FEATURES]
    features = pd.DataFrame(index=df.index)
    features['PIR_Active_Count'] = pir_df.sum(axis=1)
    features['PIR_Any_Active'] = (features['PIR_Active_Count'] > 0).astype(int)
    return features


def compute_co2_features(df, rolling_window=5):
    """
    Compute CO2 trend features.
    If multiple rows exist chronologically, rolling stats are computed.
    For a single row (inference), default change and std to 0.0.
    """
    features = pd.DataFrame(index=df.index)
    co2_series = df['S5_CO2']
    
    if len(df) > 1:
        features['CO2_Change'] = co2_series.diff().fillna(0.0)
        features['CO2_Rolling_Mean'] = co2_series.rolling(window=rolling_window, min_periods=1).mean()
        features['CO2_Rolling_Std'] = co2_series.rolling(window=rolling_window, min_periods=1).std().fillna(0.0)
    else:
        # Fallback for single-sample inference
        features['CO2_Change'] = df['S5_CO2_Slope'] if 'S5_CO2_Slope' in df else 0.0
        features['CO2_Rolling_Mean'] = co2_series
        features['CO2_Rolling_Std'] = 0.0
        
    return features


def compute_temporal_features(df):
    """
    Extract cyclical time features from datetime or Time column.
    Uses sine and cosine encoding so that 23:59 and 00:00 are close in feature space.
    """
    features = pd.DataFrame(index=df.index)
    
    # Check if a combined Datetime column exists
    if 'Datetime' in df.columns:
        dt_series = pd.to_datetime(df['Datetime'])
        hour = dt_series.dt.hour
        minute = dt_series.dt.minute
    elif 'Time' in df.columns:
        # Parse Time string HH:MM:SS
        time_parts = df['Time'].astype(str).str.split(':', expand=True)
        hour = time_parts[0].astype(int)
        minute = time_parts[1].astype(int)
    else:
        # Default fallback (e.g. middle of workday)
        hour = pd.Series(12, index=df.index)
        minute = pd.Series(0, index=df.index)
        
    features['Hour'] = hour
    features['Minute'] = minute
    
    # Cyclical transformations (24 hours per day)
    features['Hour_Sin'] = np.sin(2 * np.pi * hour / 24.0)
    features['Hour_Cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    return features


def engineer_all_features(df, include_raw=True):
    """
    Full feature engineering pipeline combining all sensor domains.
    Takes raw dataframe and returns engineered dataframe.
    """
    engineered_parts = []
    
    if include_raw:
        # Keep original raw sensor columns that are present
        raw_cols = [c for c in ALL_RAW_SENSOR_FEATURES if c in df.columns]
        engineered_parts.append(df[raw_cols].copy())
        
    engineered_parts.append(compute_temperature_features(df))
    engineered_parts.append(compute_light_features(df))
    engineered_parts.append(compute_sound_features(df))
    engineered_parts.append(compute_motion_features(df))
    engineered_parts.append(compute_co2_features(df))
    engineered_parts.append(compute_temporal_features(df))
    
    full_engineered = pd.concat(engineered_parts, axis=1)
    # Ensure no NaN or infinite values
    full_engineered = full_engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return full_engineered


# Target class labels dictionary for display
OCCUPANCY_LABELS = {
    0: '0 - Empty Room',
    1: '1 - One Occupant',
    2: '2 - Two Occupants',
    3: '3 - Three Occupants'
}

