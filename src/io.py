from pathlib import Path
import pandas as pd

def ensure_dirs(*paths: str):
    for p in paths:
        # parents=True дозволяє створювати "батьківські" папки рекурсивно
        Path(p).mkdir(parents=True, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: str):
    # parent відсікає назву файлу і залишає лише шлях до папки
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
