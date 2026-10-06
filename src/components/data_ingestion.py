from pathlib import Path
import pandas as pd
from src.logger import logger

COLUMNS = [
    "engine_id", "cycle", "setting_1", "setting_2", "setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

class DataIngestion:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)

    def _read(self, filename):
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Put the NASA FD001 file in data/raw/."
            )
        logger.info("Reading %s", path)
        return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)

    def load_train(self):
        return self._read("train_FD001.txt")

    def load_test(self):
        return self._read("test_FD001.txt")

    def load_rul(self):
        path = self.data_dir / "RUL_FD001.txt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        return pd.read_csv(path, header=None, names=["RUL"])
