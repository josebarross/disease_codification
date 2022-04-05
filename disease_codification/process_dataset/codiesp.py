import itertools
from pathlib import Path
import pandas as pd
import random


def preprocess_codiesp(corpuses_path: Path):
    corpus_path_labels = corpuses_path / "codiesp_codes"
    corpus_path_sentences = corpuses_path / "codiesp"
    df_sentences = preprocess_sentences_df(corpus_path_sentences)
    df_labels = preprocess_labels_df(corpus_path_labels, df_sentences)
    return df_sentences, df_labels


def preprocess_labels_df(corpus_path: Path, df_sentences):
    df = pd.read_csv(
        corpus_path / "codiesp-D_codes.tsv",
        sep="\t",
        on_bad_lines="skip",
        header=None,
        names=["label", "spanish_description", "english_description"],
    )
    df["label"] = df["label"].str.lower()
    sentences_labels = set(itertools.chain.from_iterable([label for label in df_sentences["labels"].values]))
    for label in sentences_labels:
        if label not in df["label"].values:
            df.loc[len(df.index)] = [label, "Sin descripcion", "Sin descripcion"]
    df["cluster"] = df["label"].str.slice(0, 1)
    df["split_type"] = get_split_types(df)
    return df


def preprocess_sentences_df(corpus_path: Path):
    sentences = []
    df = pd.DataFrame()
    for split_type in ["train", "test", "dev"]:
        df_split_type = pd.read_csv(
            corpus_path / split_type / f"{split_type}D.tsv", sep="\t", header=None, names=["filename", "labels"]
        )
        df_split_type = df_split_type.groupby("filename")["labels"].apply(list).reset_index()
        df_split_type["split_type"] = split_type
        sentences = []
        for _, row in df_split_type.iterrows():
            sentences.append(read_sentence(corpus_path, split_type, row.filename))
        df_split_type["sentence"] = sentences
        df = pd.concat([df, df_split_type])
    return df.reset_index()


def get_split_types(labels_df):
    len_list = len(labels_df)
    my_list = get_split("train", len_list) + get_split("dev", len_list)
    random.shuffle(my_list)
    return my_list[:len_list]


def get_split(split_type, len_list):
    percentage = {"train": 0.8, "dev": 0.2}
    return [split_type] * round(len_list * percentage[split_type] + 0.5)


def read_sentence(corpus_path: Path, split_type: str, filename: str):
    filename = corpus_path / split_type / "text_files" / f"{filename}.txt"
    with open(filename, "r") as f:
        return f.read()
