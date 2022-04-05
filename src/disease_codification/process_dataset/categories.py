from pathlib import Path
import pandas as pd


def process_categories(corpuses_path: Path):
    return pd.read_csv(corpuses_path / "cie-10.csv", sep=",", on_bad_lines="skip", header=True)
