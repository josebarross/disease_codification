import itertools
import re
from pathlib import Path
from typing import List

import pandas as pd


def read_livingner_training_subtask_csv(corpuses_path: Path, split_type: str):
    maps = {"train": "training", "test": "valid"}
    maps2 = {"train": "training", "test": "validation"}
    corpus_path = corpuses_path / "livingner"
    return pd.read_csv(
        corpus_path / maps[split_type] / "subtask3-Clinical-Impact" / f"{maps2[split_type]}_subtask3.tsv", sep="\t"
    )


def read_training_enriched(corpuses_path: Path, split_type: str):
    maps = {"train": "training", "test": "valid"}
    maps2 = {"train": "training", "test": "validation"}
    corpus_path = corpuses_path / "livingner"
    return pd.read_csv(
        corpus_path
        / maps[split_type]
        / "subtask3-Clinical-Impact"
        / f"{maps2[split_type]}_enriched_dataset_subtask3.tsv",
        sep="\t",
    )


def process_labels(corpuses_path: Path, for_augmentation=False) -> pd.DataFrame:
    df_files = process_files_df(corpuses_path)
    df = pd.DataFrame()
    sentences_labels = list(set(itertools.chain.from_iterable([label for label in df_files["labels"].values])))
    df["labels"] = sentences_labels
    return df.reset_index()


def process_sentence(corpuses_path: Path) -> pd.DataFrame:
    corpus_path = corpuses_path / "livingner"
    df = process_files_df(corpuses_path)
    sentences = []
    for _, row in df.iterrows():
        sentences.append(read_sentence(corpus_path, row.split_type, row.filename))
    df["sentence"] = sentences
    return df.reset_index()


def process_files_df(corpuses_path: Path):
    df = pd.DataFrame()
    for split_type in ["train", "test"]:
        df_raw = read_livingner_training_subtask_csv(corpuses_path, split_type)
        df_split_type = pd.DataFrame(columns=["filename", "labels"])
        mappings_col = {
            "isPet": "PetIDs",
            "isAnimalInjury": "AnimalInjuryIDs",
            "isFood": "FoodIDs",
            "isNosocomial": "NosocomialIDs",
        }
        for i, colname in enumerate(mappings_col):
            df_colname = df_raw[df_raw[colname] == "Yes"].copy()
            df_colname["labels"] = (
                df_colname[mappings_col[colname]]
                .apply(lambda x: re.split("\\||\\+", x))
                .apply(lambda x: [f"{i}.{s}" for s in x])
            )
            df_split_type = pd.concat([df_split_type, df_colname])
        nothing = (
            (df_raw["isPet"] == "No")
            & (df_raw["isAnimalInjury"] == "No")
            & (df_raw["isFood"] == "No")
            & (df_raw["isNosocomial"] == "No")
        )
        df_nothing = df_raw[nothing].copy()
        df_nothing["labels"] = ["-"] * df_nothing.shape[0]
        df_split_type = pd.concat([df_split_type, df_nothing])
        df_split_type = (
            df_split_type.groupby("filename")["labels"].apply(itertools.chain.from_iterable).apply(list).reset_index()
        )
        df_split_type["split_type"] = split_type
        df = pd.concat([df, df_split_type])
    return df


def read_sentence(corpus_path: Path, split_type: str, filename: str) -> str:
    maps = {"train": "training", "test": "valid"}
    filename = corpus_path / maps[split_type] / "text-files" / f"{filename}.txt"
    with open(filename, "r") as f:
        return f.read()


def cluster_assigner(corpuses_path: Path, labels: List[str]):
    mappings = {"0": "isPet", "1": "isAnimalInjury", "2": "isFood", "3": "isNosocomial", "-": "-"}
    clusters_assigned = [mappings[label.split(".")[0]] for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned


def process_ner_mentions(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    df = pd.DataFrame()
    for split_type in ["train"]:
        df_raw = read_training_enriched(corpuses_path, split_type)
        mappings_col = {
            "isPet": "0",
            "isAnimalInjury": "1",
            "isFood": "2",
            "isNosocomial": "3",
        }
        sentences = []
        labels = []
        for _, row in df_raw.iterrows():
            categories_bool = {category: row[category] for category in mappings_col.keys()}
            prefixes = [mappings_col[category] for category, bool_category in categories_bool.items() if bool_category]
            codes = row["code"].split("|")
            ls = [f"{prefix}.{code}" for prefix, code in itertools.product(prefixes, codes)]
            labels.append(ls if ls else None)
            sentences.append(row["span"])
        df_raw["sentence"] = sentences
        df_raw["labels"] = labels
        df = pd.concat([df, df_raw])
    df = df.dropna(subset=["labels"])
    return df.reset_index()


def process_ner_sentences(corpuses_path: Path) -> pd.DataFrame:
    sentences_df = pd.DataFrame()
    documents_df = process_sentence(corpuses_path)
    ner_df = process_ner_mentions(corpuses_path, for_augmentation=False)
    sentences = []
    labels = []
    for _, row_documents in documents_df.iterrows():
        if row_documents["split_type"] == "test":
            continue
        document_sentences = [s.strip() for s in re.split("\.|\n", row_documents["sentence"]) if s]
        document_mentions = ner_df[ner_df["filename"] == row_documents["filename"]]
        end_char = 0
        for sentence in document_sentences:
            start_char = end_char
            end_char += len(sentence) + 1
            labels_sentence = []
            for _, row_mention in document_mentions.iterrows():
                if start_char <= int(row_mention["off0"]) and end_char >= int(row_mention["off1"]):
                    labels_sentence += row_mention["labels"]
            if labels_sentence:
                sentences.append(sentence)
                labels.append(labels_sentence)
    sentences_df["sentence"] = sentences
    sentences_df["labels"] = labels
    return sentences_df


def process_ner_stripped(corpuses_path: Path) -> pd.DataFrame:
    df = process_ner_mentions(corpuses_path)
    group = df.groupby("filename")
    labels = group["labels"].sum().apply(set)
    sentences = group["sentence"].apply(set).apply(" ".join)
    df = labels.to_frame("labels").join(sentences.to_frame("sentence"))
    return df.reset_index()
