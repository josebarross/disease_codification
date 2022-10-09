import itertools
import random
from pathlib import Path
from typing import List

import pandas as pd
from dac_divide_and_conquer.custom_io import load_pickle, save_as_pickle

from dac_divide_and_conquer.dataset.base import DACCorpus
from dac_divide_and_conquer import logger

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


class WaitingList(DACCorpus):
    def __init__(self, data_path: Path):
        super().__init__("waiting_list", data_path)

    def process_corpus(self):
        set_mappings_icd10(self.corpuses_path)
        corpus_path = self.corpuses_path / "waiting_list"
        df_sentence = pd.read_csv(corpus_path / "cwl_umls_coded_23082022.csv", sep=",", header="infer")
        df_sentence = df_sentence.groupby(["filename", "text"])["manual_code"].apply(set).reset_index()
        labels = [
            list(itertools.chain(*[[s for s in c.split(", ") if s in MAPPINGS_ICD_10] for c in codes]))
            for codes in df_sentence["manual_code"]
        ]
        df_sentence["sentence"] = df_sentence["text"]
        df_sentence["labels"] = pd.Series(labels)
        split_types_dict = self._get_split_types_(df_sentence)
        split_types_df = pd.DataFrame(data=split_types_dict.items(), columns=["filename", "split_type"])
        df_sentence = df_sentence.set_index("filename").join(split_types_df.set_index("filename")).reset_index()
        df_sentence = df_sentence[["sentence", "labels", "split_type", "filename"]].reset_index(drop=True)
        return df_sentence

    def assign_clusters_to_label(self, label: str) -> List[str]:
        label = MAPPINGS_ICD_10[label]
        category = label[:3].lower()
        for cluster in CLUSTERS_CM:
            if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
                return [cluster]

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        corpus_path = self.corpuses_path / "waiting_list"
        df = pd.read_csv(corpus_path / "cwl_umls_coded_23082022.csv", sep=",", header="infer")
        df = df[df["manual_code"].isin(MAPPINGS_ICD_10.keys())]
        df["label"] = df["manual_code"]
        df["off0"] = df["Start"]
        df["off1"] = df["End"]
        mention_texts = []
        for text, off0, off1 in zip(df.text, df.off0, df.off1):
            try:
                mention_texts.append(text[int(off0) : int(off1)])
            except ValueError:
                mention_texts.append(None)
        df["mention_text"] = mention_texts
        return df[["filename", "label", "mention_text", "off0", "off1"]].reset_index(drop=True).dropna()

    def _get_split_types_(self, df_sentence=None):
        path = self.corpuses_path / "waiting_list" / "split_type.pickle"
        if path.exists():
            split_types_dict = load_pickle(path)
            return split_types_dict
        else:
            logger.info("Generating split_types")
            split_types_dict = {}
            split_types = [
                "test" if n >= 85 else "dev" if n >= 70 else "train"
                for n in random.choices(range(100), k=df_sentence.shape[0])
            ]
            for split_type, (_, row) in zip(split_types, df_sentence.iterrows()):
                split_types_dict[row["filename"]] = split_type
            print(split_types_dict)
            save_as_pickle(split_types_dict, path)
            return split_types_dict


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
    "s00-t99": "Traumatismos, envenenamientos y algunas otras consecuencias de causas externas",
    "u00-y99": "Causas externas de morbilidad y de mortalidad",
    "z00-z99": "Factores que influyen en el estado de salud y contacto con los servicios de salud",
}
