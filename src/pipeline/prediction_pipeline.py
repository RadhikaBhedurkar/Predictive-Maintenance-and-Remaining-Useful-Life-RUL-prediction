import joblib
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load("artifacts/model.pkl")
        self.features = joblib.load("artifacts/feature_columns.pkl")

    def predict(self, values: dict) -> float:
        row = {}
        for feature in self.features:
            if feature in values:
                row[feature] = values[feature]
            elif feature.endswith("_rolling_mean"):
                base = feature.replace("_rolling_mean", "")
                row[feature] = values.get(base, 0.0)
            elif feature.endswith("_rolling_std"):
                row[feature] = 0.0
            else:
                row[feature] = 0.0

        X = pd.DataFrame([row], columns=self.features)
        prediction = float(self.model.predict(X)[0])
        return max(0.0, prediction)
