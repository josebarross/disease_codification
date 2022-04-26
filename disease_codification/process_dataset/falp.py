import itertools
import os
import re
import random
from pathlib import Path
import statistics
from typing import List

import pandas as pd

from disease_codification.custom_io import load_pickle, save_as_pickle


def process_sentence(corpuses_path: Path) -> pd.DataFrame:
    corpus_path = corpuses_path / "falp"
    filenames = [filename.split(".")[0] for filename in os.listdir(corpus_path / "text")]
    df_sentence = pd.DataFrame(columns=["file", "sentence", "labels"])
    for filename in filenames:
        text = read_txt_file(corpus_path, filename)
        ann_df = read_ann_file(corpus_path, filename)
        class_df = read_class_file(corpus_path, filename)
        labels_df = ann_df.set_index("term").join(class_df.set_index("term"))
        labels_df = labels_df[labels_df["type"] == "Morfologia"]
        labels = [l for l in labels_df["labels"].to_list()]
        df_sentence.loc[len(df_sentence.index)] = [filename, text, labels]
    split_types_dict = get_split_types(corpuses_path, df_sentence)
    split_types_df = pd.DataFrame(data=split_types_dict.items(), columns=["file", "split_type"])
    df_sentence = df_sentence.set_index("file").join(split_types_df.set_index("file"))
    return df_sentence.reset_index()


def get_split_types(corpuses_path: Path, df_sentence=None):
    path = corpuses_path / "falp" / "split_type.pickle"
    if path.exists():
        return load_pickle(path)
    else:
        print("Generating split_types")
        split_types_dict = {}
        split_types = [
            "test" if n >= 90 else "dev" if n >= 80 else "train"
            for n in random.choices(range(100), k=df_sentence.shape[0])
        ]
        for split_type, (i, row) in zip(split_types, df_sentence.iterrows()):
            split_types_dict[row["file"]] = split_type
        save_as_pickle(split_types_dict, path)
        return split_types_dict


def read_txt_file(corpus_path: Path, filename):
    with open(corpus_path / "text" / f"{filename}.txt", "r") as f:
        return f.read()


def read_ann_file(corpus_path: Path, filename: str):
    annotations_df = pd.read_csv(
        corpus_path / "cie-o" / f"{filename}-cie-code.ann", sep="\t", names=["term", "info", "sentence"]
    )
    annotations_df[["labels", "off0", "off1"]] = annotations_df["info"].str.split(" ", 2, expand=True)
    annotations_df = annotations_df[~annotations_df["labels"].str.startswith("C")]
    annotations_df["file"] = [filename] * annotations_df.shape[0]
    return annotations_df


def read_class_file(corpus_path: Path, filename: str):
    class_df = pd.read_csv(
        corpus_path / "class" / f"{filename}-class.ann", sep="\t", names=["term", "info", "sentence"]
    )
    class_df[["type", "off0", "off1"]] = class_df["info"].str.split(" ", 2, expand=True)
    return class_df[["type", "term"]]


def process_labels(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    df_sentence = process_sentence(corpuses_path)
    sentences_labels = list(set(itertools.chain.from_iterable([label for label in df_sentence["labels"].values])))

    df = pd.read_excel(corpuses_path / "cie-o-3-codes.xlsx", sheet_name=1)
    df = pd.melt(df, id_vars=["codigo"], value_vars=["Descriptor Completo", "Descriptor Abreviado"])
    df["labels"] = df["codigo"]
    df["sentence"] = df["value"]
    df = df.loc[:, ["labels", "sentence"]]

    for label in sentences_labels:
        if label not in df["labels"].values:
            df.loc[len(df.index)] = [label, "Sin descripcion"]

    df = df[df["labels"].isin(sentences_labels)]

    if for_augmentation:
        df = df[df["sentence"] != "Sin descripcion"]
        df = df.drop_duplicates(["sentence", "labels"])
        df["labels"] = df["labels"].apply(lambda x: [x])
    return df.reset_index()


def process_ner_mentions(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    corpus_path = corpuses_path / "falp"
    split_types = get_split_types(corpuses_path)
    df = pd.DataFrame()
    filenames = [
        filename.split(".")[0]
        for filename in os.listdir(corpus_path / "text")
        if split_types[filename.split(".")[0]] in ["train", "dev"]
    ]
    df = pd.DataFrame()
    for filename in filenames:
        ann_df = read_ann_file(corpus_path, filename)
        class_df = read_class_file(corpus_path, filename)
        labels_df = ann_df.set_index("term").join(class_df.set_index("term"))
        labels_df = labels_df[labels_df["type"] == "Morfologia"]
        df = pd.concat([df, labels_df])
    if for_augmentation:
        df = df.drop_duplicates(subset=["sentence", "labels"])
    df["labels"] = df["labels"].apply(lambda x: [x])
    return df.reset_index()


def process_ner_sentences(corpuses_path: Path) -> pd.DataFrame:
    sentences_df = pd.DataFrame()
    documents_df = process_sentence(corpuses_path)
    ner_df = process_ner_mentions(corpuses_path, for_augmentation=False)
    sentences = []
    labels = []
    error_ner_sentences = 0
    for _, row_documents in documents_df.iterrows():
        if row_documents["split_type"] == "test":
            continue
        complete_document = row_documents["sentence"]
        document_sentences = [s for s in re.split("\.|\n", complete_document)]
        document_labels = ner_df[ner_df["file"] == row_documents["file"]]
        end_char = 0
        for sentence in document_sentences:
            start_char = end_char
            end_char += len(sentence) + 1
            labels_sentence = []
            for _, row_label in document_labels.iterrows():
                if start_char <= int(row_label["off0"]) and end_char >= int(row_label["off1"]):
                    try:
                        assert row_label["sentence"].replace(".", "").replace(" ", "") in sentence.replace(
                            ".", ""
                        ).replace(" ", "")
                        labels_sentence.append(row_label["labels"][0])
                    except AssertionError:
                        print(row_label, sentence)
                        error_ner_sentences += 1
            if sentence:
                sentences.append(sentence.strip())
                labels.append(labels_sentence)
    print(f"Error parsing {error_ner_sentences} mentions")
    sentences_df["sentence"] = sentences
    sentences_df["labels"] = labels
    return sentences_df


def process_ner_stripped(corpuses_path: Path) -> pd.DataFrame:
    df = process_ner_mentions(corpuses_path)
    group = df.groupby("file")
    labels = group["labels"].sum().apply(set)
    sentences = group["sentence"].apply(list).apply(" ".join)
    df = labels.to_frame("labels").join(sentences.to_frame("sentence"))
    return df.reset_index()


clusters = {
    "800-800": "Neoplasias, SAI",
    "801-804": "NEOPLASIAS EPITELIALES, SAI",
    "805-808": "NEOPLASIAS EPIDERMOIDES",
    "809-811": "NEOPLASIAS BASOCELULARES",
    "812-813": "PAPILOMAS Y CARCINOMAS DE CELULAS TRANSICIONALES",
    "814-838": "ADENOMAS Y ADENOCARCINOMAS",
    "839-842": "NEOPLASIAS DE LOS ANEXOS CUTANEOS",
    "843-843": "NEOPLASIAS MUCOEPIDERMOIDES",
    "844-849": "NEOPLASIAS QUISTICAS, MUCINOSAS Y SEROSAS",
    "850-854": "NEOPLASIAS DUCTALES, LOBULILLARES Y MEDULARES",
    "855-855": "NEOPLASIAS DE CELULAS ACINOSAS",
    "856-857": "NEOPLASIAS EPITELIALES COMPLEJAS",
    "858-858": "NEOPLASIAS EPITELIALES TIMOMAS",
    "859-867": "NEOPLASIAS DEL ESTROMA ESPECIALIZADO DE LAS GONADAS",
    "868-871": "PARAGANGLIOMAS Y TUMORES GLOMICOS",
    "872-879": "NEVOS Y MELANOMAS",
    "880-880": "TUMORES Y SARCOMAS DE TEJIDOS BLANDOS, SAI",
    "881-883": "NEOPLASIAS FIBROMATOSAS",
    "884-884": "NEOPLASIAS MIXOMATOSAS",
    "885-888": "NEOPLASIAS LIPOMATOSAS",
    "889-892": "NEOPLASIAS MIOMATOSAS",
    "893-899": "NEOPLASIAS COMPLEJAS MIXTAS Y DEL ESTROMA",
    "900-903": "NEOPLASIAS FIBROEPITELIALES",
    "904-904": "NEOPLASIAS SINOVIALES",
    "905-905": "NEOPLASIAS MESOTELIALES",
    "906-909": "NEOPLASIAS DE CELULAS GERMINALES",
    "910-910": "NEOPLASIAS TROFOBLASTICAS",
    "911-911": "MESONEFROMAS",
    "953-953": "MENINGIOMAS",
    "912-916": "TUMORES DE LOS VASOS SANGUINEOS",
    "917-917": "TUMORES DE LOS VASOS LINFATICOS",
    "918-924": "NEOPLASIAS OSEAS Y CONDROMATOSAS",
    "925-925": "TUMORES DE CELULAS GIGANTES",
    "926-926": "OTROS TUMORES OSEOS",
    "927-934": "TUMORES ODONTOGENICOS",
    "935-937": "OTROS TUMORES",
    "938-948": "GLIOMAS",
    "949-952": "NEOPLASIAS NEUROEPITELIOMATOSAS",
    "954-957": "TUMORES DE LAS VAINAS NERVIOSAS",
    "958-958": "TUMORES DE CELULAS GRANULARES Y SARCOMA ALVEOLAR DE PARTES BLANDAS",
    "959-972": "LINFOMAS HODGKIN Y NO-HODGKIN",
    "973-973": "TUMORES DE CELULAS PLASMATICAS",
    "974-974": "TUMORES DE LOS MASTOCITOS",
    "975-975": "NEOPLASIAS DE HISTIOCITOS Y DE CÉLULAS LINFOIDES ACCESORIAS",
    "976-976": "ENFERMEDADES INMUNOPROLIFERATIVAS",
    "980-994": "LEUCEMIAS",
    "995-996": "OTROS SINDROMES MIELOPROLIFERATIVOS",
    "997-997": "OTROS DESORDENES HEMATOLOGICOS",
    "998-999": "SINDROMES MIELODISPLASICOS",
}


def cluster_assigner(corpuses_path: Path, labels: List[str]):
    clusters_assigned = [assign_label(label) for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned


def assign_label(label):
    category = label[:3]
    for cluster in clusters:
        if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
            return cluster
