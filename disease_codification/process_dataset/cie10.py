from pathlib import Path
from typing import List

import pandas as pd


def get_cie10_df(corpuses_path: Path):
    df = pd.read_csv(corpuses_path / "cie-10.csv", sep=",")
    return df


def from_cie10_descriptions(self):
    all_labels = self.mappings_label_to_cluster.keys()
    cie10_df = get_cie10_df(self.corpuses_path)
    sentences = []
    labels = []
    for _, row in cie10_df.iterrows():
        code = row["code"]
        if len(code.split("-")) == 2:
            start = code.split("-")[0]
            end = code.split("-")[1]
            ls = self._determine_labels_in_range(start, end, all_labels)
        else:
            ls = self._determine_labels_startswith(code, all_labels)
        if ls:
            sentences.append(row["description"])
            labels.append(ls)
    return sentences, labels


def determine_labels_startswith(code: str, all_labels: List[str]):
    return [label for label in all_labels if label.startswith(code.lower())]


def determine_labels_in_range(start: str, end: str, all_labels: List[str]):
    start_cat = start[0].lower()
    end_cat = end[0].lower()
    starts_int = start[1:3] if not start[1:3] == "01" else "00"
    ends_int = end[1:3]
    labels_in_range = []
    for label in all_labels:
        label_int = label[1:3]
        label_in_range = (
            (start_cat == end_cat and label.startswith(start_cat) and starts_int <= label_int <= ends_int)
            or (label.startswith(start_cat) and label_int >= starts_int)
            or (label.startswith(end_cat) and label_int <= ends_int)
        )
        if label_in_range:
            labels_in_range.append(label)
    return labels_in_range
