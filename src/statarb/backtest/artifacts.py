import os
from pathlib import Path
import pandas as pd
import pickle


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: str, name: str = None) -> None:
    if name:
        path = os.path.join(path, f"{name}.csv")
    _ensure_dir(os.path.dirname(path))
    df.to_csv(path)


def save_artifact(df: pd.DataFrame, artifact_dir: str, sector_id, rebal_date, name: str) -> str:
    dir_path = os.path.join(artifact_dir, str(sector_id), pd.to_datetime(rebal_date).strftime("%Y%m%d"))
    _ensure_dir(dir_path)
    file_path = os.path.join(dir_path, f"{name}.csv")
    df.to_csv(file_path)
    return file_path


def save_pickle(obj: object, path: str, name: str = None) -> None:
    if name:
        path = os.path.join(path, f"{name}.pkl")
    _ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
