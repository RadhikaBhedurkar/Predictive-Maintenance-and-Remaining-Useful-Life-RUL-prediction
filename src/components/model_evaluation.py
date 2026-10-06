from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save(model, X, y, metadata, artifact_dir="artifacts"):
    artifact_dir = Path(artifact_dir)
    pred = model.predict(X)

    metrics = {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred))
    }

    output = metadata[["engine_id", "cycle", "actual_RUL"]].copy()
    output["predicted_RUL"] = pred.round(2)
    output["absolute_error"] = (output["actual_RUL"] - output["predicted_RUL"]).abs().round(2)
    output.to_csv(artifact_dir / "test_predictions.csv", index=False)

    pd.DataFrame([metrics]).to_csv(artifact_dir / "test_metrics.csv", index=False)
    return metrics
