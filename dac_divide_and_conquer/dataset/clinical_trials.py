from enum import Enum
import itertools
import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from dac_divide_and_conquer.dataset.base import DACCorpus
from dac_divide_and_conquer import logger
from dac_divide_and_conquer import corpora_downloader as cd


class ClinicalTrialsSubtask(Enum):
    abstracts = "abstracts"
    eudract = "eudract"


MAPPINGS_ICD_10 = {}


def set_mappings_icd10(corpuses_path):
    filename = corpuses_path / "mappings-icd10-umls.csv"
    df = pd.read_csv(filename, sep=",", on_bad_lines="skip")
    df_bas = df[df["SAB"] == "ICD10CM"]
    for i, row in df_bas.iterrows():
        MAPPINGS_ICD_10[row["CUI"]] = row["CODE"]
    df_bas = df[df["SAB"] == "ICD10"]
    for i, row in df_bas.iterrows():
        if row["CUI"] not in MAPPINGS_ICD_10:
            MAPPINGS_ICD_10[row["CUI"]] = row["CODE"]


def read_annotation(file: Path) -> str:
    with open(f"{file}.ann", "r") as f:
        lines = f.readlines()
        annotations = {}
        for line in lines:
            if line.startswith("T"):
                contents = line.split("\t")
                detail = contents[1].split(" ")
                text = contents[2]
                annotations[contents[0]] = {"type": detail[0], "off0": detail[1], "off1": detail[2], "sentence": text}
            elif line.startswith("#"):
                annotation_number = line.split("\t")[1].split(" ")[1]
                codes = [c[:8].strip("\n").strip() for c in line.split("\t")[2].split(", ")]
                annotations[annotation_number]["codes"] = [c for c in codes if c in MAPPINGS_ICD_10]
        return annotations


class ClinicalTrialsCorpus(DACCorpus):
    def __init__(self, data_path: Path, subtask: ClinicalTrialsSubtask):
        self.data_path = data_path
        self.subtask = subtask
        super().__init__(f"clinical_trials_{self.subtask}", data_path)

    def process_corpus(self):
        def read_sentence(file: Path) -> str:
            with open(f"{file}.txt", "r") as f:
                return f.read()

        set_mappings_icd10(self.corpuses_path)

        corpus_path = self.corpuses_path / "CT-EBM-SP"
        df_dict = {"sentence": [], "labels": [], "split_type": [], "filename": []}
        for split_type in ["train", "test", "dev"]:
            path: Path = corpus_path / split_type / self.subtask.value
            filenames = {p.name.split(".")[0] for p in path.iterdir()}
            for filename in filenames:
                file_path = path / filename
                txt = read_sentence(file_path)
                annotations = read_annotation(file_path)
                labels = [l for l in itertools.chain(*[ann.get("codes", []) for ann in annotations.values()])]
                df_dict["sentence"].append(txt)
                df_dict["filename"].append(filename)
                df_dict["split_type"].append(split_type)
                df_dict["labels"].append(labels)
        return pd.DataFrame(data=df_dict)

    def assign_clusters_to_label(self, label: str) -> List[str]:
        label = MAPPINGS_ICD_10[label]
        category = label[:3].lower()
        for cluster in CLUSTERS_CM:
            if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
                return [cluster]

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        df_dict = {"filename": [], "label": [], "mention_text": [], "off0": [], "off1": []}
        corpus_path = self.corpuses_path / "CT-EBM-SP"
        for split_type in ["train", "test", "dev"]:
            path: Path = corpus_path / split_type / self.subtask.value
            filenames = {p.name.split(".")[0] for p in path.iterdir()}
            for filename in filenames:
                file_path = path / filename
                annotations = read_annotation(file_path)
                for ann in annotations.values():
                    for c in ann.get("codes", []):
                        df_dict["filename"].append(filename)
                        df_dict["label"].append(c)
                        df_dict["mention_text"].append(ann["sentence"])
                        df_dict["off0"].append(ann["off0"])
                        df_dict["off1"].append(ann["off1"])
        return pd.DataFrame(data=df_dict)


CLUSTERS_CM = {
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
    "u00-y99": "Causas externas de morbilidad y de mortalidad",
    "z00-z99": "Factores que influyen en el estado de salud y contacto con los servicios de salud",
}
