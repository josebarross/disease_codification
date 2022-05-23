from enum import Enum
import itertools
import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from disease_codification.dataset.base import DACCorpus
from disease_codification import logger
from disease_codification import corpora_downloader as cd


class CodiespSubtask(Enum):
    diagnostics = "diagnostics"
    procedures = "procedures"


class CodiespCorpus(DACCorpus):
    def __init__(self, data_path: Path, subtask: CodiespSubtask):
        self.subtask = subtask
        super().__init__(
            f"codiesp_{self.subtask}", data_path, first_n_digits_f1=4 if subtask == CodiespSubtask.procedures else 3
        )

    def process_corpus(self):
        def read_sentence(corpus_path: Path, split_type: str, filename: str) -> str:
            filename = corpus_path / split_type / "text_files" / f"{filename}.txt"
            with open(filename, "r") as f:
                return f.read()

        corpus_path = self.corpuses_path / "codiesp"
        df = pd.DataFrame()
        for split_type in ["train", "test", "dev"]:
            name = f"{split_type}D.tsv" if self.subtask == CodiespSubtask.diagnostics else f"{split_type}P.tsv"
            df_split_type = pd.read_csv(
                corpus_path / split_type / name,
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
        return df[["sentence", "labels", "split_type", "filename"]].reset_index(drop=True)

    def assign_clusters_to_label(self, label: str) -> List[str]:
        if self.subtask == CodiespSubtask.diagnostics:
            category = label[:3]
            for cluster in CLUSTERS:
                if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
                    return [cluster]
        else:
            return [label[0]]

    def download_corpus(self):
        logger.info("Downloading codiesp corpus")
        url = "https://zenodo.org/record/3837305/files/codiesp.zip?download=1"
        cd.download_corpus(self.corpuses_path, "codiesp", url, old_name="final_dataset_v4_to_publish")
        logger.info("Downloading codiesp labels")
        url = "https://zenodo.org/record/3632523/files/codiesp_codes.zip?download=1"
        cd.download_corpus(self.corpuses_path, "codiesp_codes", url, create_containing_folder=True)

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        df = pd.DataFrame()
        for split_type in ["train", "dev"]:
            df_split_type = pd.read_csv(
                self.corpuses_path / "codiesp" / split_type / f"{split_type}X.tsv",
                sep="\t",
                on_bad_lines="skip",
                header=None,
                names=["filename", "type", "label", "mention_text", "position"],
            )
            type_of_mention = "DIAGNOSTICO" if self.subtask == CodiespSubtask.diagnostics else "PROCEDIMIENTO"
            df_split_type = df_split_type[df_split_type["type"] == type_of_mention]
            df_split_type["off0"] = df_split_type["position"].str.split(" ").apply(lambda x: x[0])
            df_split_type["off1"] = df_split_type["position"].str.split(" ").apply(lambda x: x[-1])
            df = pd.concat([df, df_split_type])
        return df[["filename", "label", "mention_text", "off0", "off1"]].reset_index(drop=True)

    def process_augmentation_descriptions(self) -> pd.DataFrame:
        corpus_path = self.corpuses_path / "codiesp_codes"
        name = "codiesp-D_codes.tsv" if self.subtask == CodiespSubtask.diagnostics else "codiesp-P_codes.tsv"
        df = pd.read_csv(
            corpus_path / name,
            sep="\t",
            on_bad_lines="skip",
            header=None,
            names=["label", "description", "english_description"],
        )
        df["label"] = df["label"].str.lower()
        return df[["label", "description"]].reset_index(drop=True)


CLUSTERS = {
    "a00-b99": "Ciertas enfermedades infecciosas y parasitarias",
    "c00-d49": "Tumores [neoplasias]",
    "d50-d89": "Enfermedades de la sangre y de los órganos hematopoyéticos, y ciertos trastornos que afectan el mecanismo de la inmunidad",
    "e00-e89": "Enfermedades endocrinas, nutricionales y metabolicas",
    "f00-f99": "Trastornos mentales y del comportamiento",
    "g00-g99": "Enfermedades del sistema nervioso",
    "h00-h59": "Enfermedades del ojo y sus anexos",
    "h60-h95": "Enfermedades del oído y de la apófisis mastoides",
    "i00-i99": "Enfermedades del sistema circulatorio",
    "j00-j99": "Enfermedades del sistema respiratorio",
    "k00-k95": "Enfermedades del sistema digestivo",
    "l00-l99": "Enfermedades de la piel y del tejido subcutáneo",
    "m00-m99": "Enfermedades del sistema osteomuscular y del tejido conjuntivo",
    "n00-n99": "Enfermedades del sistema genitourinario",
    "o00-o9a": "Embarazo, parto y puerperio",
    "p00-p96": "Ciertas afecciones originadas en el período perinatal",
    "q00-q99": "Malformaciones congénitas, deformidades y anomalías cromosómicas",
    "r00-r99": " Síntomas, signos y hallazgos anormales clínicos y de laboratorio, no clasificados en otra parte",
    "s00-t88": "Traumatismos, envenenamientos y algunas otras consecuencias de causas externas",
    "v00-y99": "Causas externas de morbilidad y de mortalidad",
    "z00-z99": "Factores que influyen en el estado de salud y contacto con los servicios de salud",
}
