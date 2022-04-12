import itertools
import re
from pathlib import Path

import pandas as pd


def process_codiesp_labels(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    corpus_path = corpuses_path / "codiesp_codes"
    df_sentences = process_codiesp_sentence(corpuses_path)
    df = pd.read_csv(
        corpus_path / "codiesp-D_codes.tsv",
        sep="\t",
        on_bad_lines="skip",
        header=None,
        names=["labels", "sentence", "english_description"],
    )
    df["labels"] = df["labels"].str.lower()
    sentences_labels = set(itertools.chain.from_iterable([label for label in df_sentences["labels"].values]))
    for label in sentences_labels:
        if label not in df["labels"].values:
            df.loc[len(df.index)] = [label, "Sin descripcion", "Sin descripcion"]
    df = df[df["labels"].isin(sentences_labels)]
    if for_augmentation:
        df = df[df["sentence"] != "Sin descripcion"]
        df = df.drop_duplicates(["sentence", "labels"])
        df["labels"] = df["labels"].apply(lambda x: [x])
    return df.reset_index()


def process_codiesp_sentence(corpuses_path: Path) -> pd.DataFrame:
    corpus_path = corpuses_path / "codiesp"
    df = pd.DataFrame()
    for split_type in ["train", "test", "dev"]:
        df_split_type = pd.read_csv(
            corpus_path / split_type / f"{split_type}D.tsv",
            sep="\t",
            header=None,
            names=["filename", "labels"],
        )
        df_split_type = df_split_type.groupby("filename")["labels"].apply(list).reset_index()
        df_split_type["split_type"] = split_type
        sentences = []
        for _, row in df_split_type.iterrows():
            sentences.append(read_sentence(corpus_path, split_type, row.filename))
        df_split_type["sentence"] = sentences
        df = pd.concat([df, df_split_type])
    return df.reset_index()


def preprocess_ner_mentions(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    df = pd.DataFrame()
    for split_type in ["train", "dev"]:
        df_split_type = pd.read_csv(
            corpuses_path / "codiesp" / split_type / f"{split_type}X.tsv",
            sep="\t",
            on_bad_lines="skip",
            header=None,
            names=["document", "type", "labels", "sentence", "position"],
        )
        df_split_type = df_split_type[df_split_type["type"] == "DIAGNOSTICO"]
        df = pd.concat([df, df_split_type])
    if for_augmentation:
        df = df.drop_duplicates(["sentence", "labels"])
        df["labels"] = df["labels"].apply(lambda x: [x])
    return df.reset_index()


def preprocess_ner_sentences(corpuses_path: Path) -> pd.DataFrame:
    sentences_df = pd.DataFrame()
    documents_df = process_codiesp_sentence(corpuses_path)
    ner_df = preprocess_ner_mentions(corpuses_path, for_augmentation=False)
    sentences = []
    labels = []
    for _, row_documents in documents_df.iterrows():
        document_sentences = [s.strip() for s in re.split("\.|\n", row_documents["sentence"]) if s]
        document_labels = ner_df[ner_df["document"] == row_documents["filename"]]
        end_char = 0
        for sentence in document_sentences:
            start_char = end_char
            end_char += len(sentence) + 1
            labels_sentence = []
            for _, row_label in document_labels.iterrows():
                positions = row_label["position"].split(";")
                if len(positions) == 1 and any(
                    start_char <= int(position.split(" ")[0]) and end_char >= int(position.split(" ")[1])
                    for position in positions
                ):
                    labels_sentence.append(row_label["labels"])
            sentences.append(sentence)
            labels.append(labels_sentence)
    sentences_df["sentence"] = sentences
    sentences_df["labels"] = labels
    return sentences_df


def preprocess_ner_stripped(corpuses_path: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    for split_type in ["train", "dev"]:
        df_split_type = pd.read_csv(
            corpuses_path / "codiesp" / split_type / f"{split_type}X.tsv",
            sep="\t",
            on_bad_lines="skip",
            header=None,
            names=["document", "type", "labels", "sentence", "location"],
        )
        df_split_type = df_split_type[df_split_type["type"] == "DIAGNOSTICO"]
        group = df_split_type.groupby("document")
        labels = group["labels"].apply(list)
        sentences = group["sentence"].apply(" ".join)
        df_split_type = labels.to_frame("labels").join(sentences.to_frame("sentence"))
        df = pd.concat([df, df_split_type])
    return df.reset_index()


def read_sentence(corpus_path: Path, split_type: str, filename: str) -> str:
    filename = corpus_path / split_type / "text_files" / f"{filename}.txt"
    with open(filename, "r") as f:
        return f.read()
