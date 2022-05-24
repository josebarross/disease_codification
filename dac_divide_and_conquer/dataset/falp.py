import itertools
import os
import re
import random
from pathlib import Path
import statistics
from typing import List

import pandas as pd

from dac_divide_and_conquer.custom_io import load_pickle, save_as_pickle
from dac_divide_and_conquer import logger
from dac_divide_and_conquer.dataset.base import DACCorpus


class FALPCorpus(DACCorpus):
    def __init__(self, data_path: Path):
        super().__init__("falp", data_path)

    def process_corpus(self):
        corpus_path = self.corpuses_path / self.corpus

        def read_txt_file(corpus_path: Path, filename):
            with open(corpus_path / "text" / f"{filename}.txt", "r") as f:
                return f.read()

        filenames = [filename.split(".")[0] for filename in os.listdir(corpus_path / "text")]
        df_sentence = pd.DataFrame(columns=["filename", "sentence", "labels"])
        for filename in filenames:
            text = read_txt_file(corpus_path, filename)
            ann_df = self._read_ann_file_(filename)
            class_df = self._read_class_file_(filename)
            labels_df = ann_df.set_index("term").join(class_df.set_index("term"))
            labels_df = labels_df[labels_df["type"] == "Morfologia"]
            labels = [l for l in labels_df["label"].to_list()]
            df_sentence.loc[len(df_sentence.index)] = [filename, text, labels]
        split_types_dict = self._get_split_types_(df_sentence)
        split_types_df = pd.DataFrame(data=split_types_dict.items(), columns=["filename", "split_type"])
        df_sentence = df_sentence.set_index("filename").join(split_types_df.set_index("filename")).reset_index()
        return df_sentence[["sentence", "labels", "split_type", "filename"]].reset_index(drop=True)

    def assign_clusters_to_label(self, label: str) -> List[str]:
        category = label[:3]
        for cluster in CLUSTERS:
            if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
                return [cluster]

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        corpus_path = self.corpuses_path / "falp"
        split_types = self._get_split_types_()
        df = pd.DataFrame()
        filenames = [
            filename.split(".")[0]
            for filename in os.listdir(corpus_path / "text")
            if split_types[filename.split(".")[0]] in ["train", "dev"]
        ]
        df = pd.DataFrame()
        for filename in filenames:
            ann_df = self._read_ann_file_(filename)
            class_df = self._read_class_file_(filename)
            labels_df = ann_df.set_index("term").join(class_df.set_index("term"))
            labels_df = labels_df[labels_df["type"] == "Morfologia"]
            df = pd.concat([df, labels_df])
        return df[["filename", "label", "mention_text", "off0", "off1"]].reset_index(drop=True)

    def process_augmentation_descriptions(self) -> pd.DataFrame:
        df = pd.read_excel(self.corpuses_path / "cie-o-3-codes.xlsx", sheet_name=1)
        df = pd.melt(df, id_vars=["codigo"], value_vars=["Descriptor Completo", "Descriptor Abreviado"])
        df["label"] = df["codigo"]
        df["description"] = df["value"]
        return df[["label", "description"]].reset_index(drop=True)

    def _read_ann_file_(self, filename: str):
        corpus_path = self.corpuses_path / self.corpus
        annotations_df = pd.read_csv(
            corpus_path / "cie-o" / f"{filename}-cie-code.ann", sep="\t", names=["term", "info", "mention_text"]
        )
        annotations_df[["label", "off0", "off1"]] = annotations_df["info"].str.split(" ", 2, expand=True)
        annotations_df = annotations_df[~annotations_df["label"].str.startswith("C")]
        annotations_df["filename"] = [filename] * annotations_df.shape[0]
        return annotations_df

    def _read_class_file_(self, filename: str):
        class_df = pd.read_csv(
            self.corpuses_path / self.corpus / "class" / f"{filename}-class.ann",
            sep="\t",
            names=["term", "info", "mention_text"],
        )
        class_df[["type", "off0", "off1"]] = class_df["info"].str.split(" ", 2, expand=True)
        return class_df[["type", "term"]]

    def _get_split_types_(self, df_sentence=None):
        path = self.corpuses_path / "falp" / "split_type.pickle"
        if path.exists():
            return load_pickle(path)
        else:
            logger.info("Generating split_types")
            split_types_dict = {}
            split_types = [
                "test" if n >= 90 else "dev" if n >= 80 else "train"
                for n in random.choices(range(100), k=df_sentence.shape[0])
            ]
            for split_type, (_, row) in zip(split_types, df_sentence.iterrows()):
                split_types_dict[row["filename"]] = split_type
            save_as_pickle(split_types_dict, path)
            return split_types_dict


CLUSTERS = {
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
