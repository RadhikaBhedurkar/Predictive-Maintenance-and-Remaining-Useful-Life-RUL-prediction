import numpy as np
import pandas as pd

BASE_FEATURES = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "sensor_8", "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]

ROLLING_BASE = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_11", "sensor_12", "sensor_15"
]

def add_rul(df):
    out = df.copy()
    max_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["RUL"] = max_cycle - out["cycle"]
    return out

def add_rolling_features(df):
    out = df.copy()
    # shift(1) prevents the current observation from being used to calculate
    # its own historical rolling statistics.
    for col in ROLLING_BASE:
        grouped = out.groupby("engine_id")[col]
        out[f"{col}_rolling_mean"] = grouped.transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        out[f"{col}_rolling_std"] = grouped.transform(
            lambda s: s.shift(1).rolling(5, min_periods=2).std()
        )
    return out

def _feature_columns(df):
    rolling = [c for c in df.columns if c.endswith("_rolling_mean") or c.endswith("_rolling_std")]
    return BASE_FEATURES + rolling

def prepare_train(df):
    out = add_rolling_features(add_rul(df))
    features = _feature_columns(out)
    X = out[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = out["RUL"].astype(float)
    return X, y, features

def prepare_test(df, rul):
    out = add_rolling_features(df)
    final_rows = out.groupby("engine_id", as_index=False).tail(1).reset_index(drop=True)

    if len(final_rows) != len(rul):
        raise ValueError(
            f"Test engines ({len(final_rows)}) and RUL rows ({len(rul)}) do not match."
        )

    final_rows["actual_RUL"] = rul["RUL"].values
    features = _feature_columns(out)
    X = final_rows[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = final_rows["actual_RUL"].astype(float)
    return X, y, final_rows, features
