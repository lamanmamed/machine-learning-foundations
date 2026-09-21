"""Dataset loaders matching the experiments in the submitted projects."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


DIABETES_URL = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"


def load_raw_diabetes(url: str = DIABETES_URL) -> tuple[np.ndarray, np.ndarray]:
    """Load the raw diabetes table used in the regression experiment."""
    with urlopen(url) as response:
        text = response.read().decode("utf-8")
    frame = pd.read_csv(StringIO(text), sep="	")
    return frame.drop(columns=["Y"]).to_numpy(dtype=np.float32), frame["Y"].to_numpy(dtype=np.float32)


def load_iris_arrays() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    return iris.data.astype(np.float32), iris.target.astype(np.int64)


def load_peterson_barney(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load F1/F2 features and phoneme IDs from a local PB_data.npy file."""
    data = np.load(Path(path), allow_pickle=True).tolist()
    x = np.column_stack([data["f1"], data["f2"]]).astype(np.float64)
    y = np.asarray(data["phoneme_id"])
    return x, y
