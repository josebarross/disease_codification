import os
from pathlib import Path
import pickle
from typing import List


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_as_pickle(object, path: Path):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def write_fasttext_file(sentences: List[str], labels: List[List[str]], filepath: Path):
    sentences_without_lines = [text.replace("\n", " ") for text in sentences]
    lines = []
    for text, list_of_labels in zip(sentences_without_lines, labels):
        if not list_of_labels:
            continue
        if not text:
            continue
        lines.append(f"{' '.join(f'__label__<{label}>' for label in set(list_of_labels))} <{text}>")
    with open(filepath, "w") as file:
        file.write("\n".join(lines))


def create_dir_if_dont_exist(path: Path):
    if not path.exists():
        os.makedirs(f"{path}/")
