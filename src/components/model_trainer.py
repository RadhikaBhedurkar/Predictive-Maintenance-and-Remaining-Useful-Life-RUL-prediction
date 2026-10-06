from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class ModelTrainer:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def metrics(model, X, y):
        pred = model.predict(X)
        return {
            "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(mean_squared_error(y, pred) ** 0.5),
            "R2": float(r2_score(y, pred))
        }, pred

    def train(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=250, random_state=42, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:squarederror", random_state=42, n_jobs=4
            )
        }

        rows = []
        best = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            m, _ = self.metrics(model, X_valid, y_valid)
            rows.append({"Model": name, **m})

            if best is None or m["MAE"] < best["metrics"]["MAE"]:
                best = {"name": name, "model": model, "metrics": m}

        results = pd.DataFrame(rows).sort_values("MAE")
        results.to_csv(self.artifact_dir / "model_comparison.csv", index=False)

        joblib.dump(best["model"], self.artifact_dir / "model.pkl")
        return best, results
