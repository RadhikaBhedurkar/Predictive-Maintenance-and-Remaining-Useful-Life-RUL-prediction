import pandas as pd

class DataValidation:
    REQUIRED_COLUMNS = {"engine_id", "cycle"}

    def validate(self, df: pd.DataFrame):
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if df.empty:
            raise ValueError("Dataset is empty.")

        if df["engine_id"].isna().any() or df["cycle"].isna().any():
            raise ValueError("engine_id/cycle contains missing values.")

        if (df["cycle"] <= 0).any():
            raise ValueError("Cycle values must be positive.")

        return True
