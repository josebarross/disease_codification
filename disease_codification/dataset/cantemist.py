import os
from pathlib import Path
import statistics
from typing import List

import pandas as pd
from disease_codification import logger
from disease_codification.corpora_downloader import download_corpus
from disease_codification.dataset.base import DACCorpus


class CantemistCorpus(DACCorpus):
    def __init__(self, data_path: Path):
        super().__init__("cantemist", data_path)

    def process_corpus(self):
        def read_sentence(corpus_path: Path, split_type: str, filename: str, n: int) -> str:
            if split_type == "test":
                filename = (
                    corpus_path
                    / (f"{split_type}-set" if not n else f"{split_type}-set{n}")
                    / "cantemist-ner"
                    / f"{filename}.txt"
                )
            else:
                filename = (
                    corpus_path
                    / (f"{split_type}-set" if not n else f"{split_type}-set{n}")
                    / "cantemist-coding"
                    / "txt"
                    / f"{filename}.txt"
                )
            with open(filename, "r") as f:
                return f.read()

        corpus_path = self.corpuses_path / self.corpus
        df = pd.DataFrame()
        for split_type in ["train", "test", "dev1", "dev2"]:
            n = 0
            if split_type.startswith("dev"):
                n = int(split_type[-1])
                df_split_type = pd.read_csv(
                    corpus_path / f"dev-set{n}" / "cantemist-coding" / f"dev{n}-coding.tsv", sep="\t"
                )
                split_type = "dev"
            else:
                df_split_type = pd.read_csv(
                    corpus_path / f"{split_type}-set" / "cantemist-coding" / f"{split_type}-coding.tsv", sep="\t"
                )
            df_split_type = df_split_type.groupby("file")["code"].apply(set).reset_index()
            df_split_type["split_type"] = split_type
            sentences = []
            amount = 0
            lens = []
            for _, row in df_split_type.iterrows():
                sentence = read_sentence(corpus_path, split_type, row["file"], n=n)
                length = len(sentence.split(" "))
                lens.append(length)
                if length > 512:
                    amount += 1
                sentences.append(sentence)
            logger.info(f"Documents with more than 512 tokens: {amount}")
            logger.info(f"Average length: {statistics.mean(lens)}")
            df_split_type["sentence"] = sentences
            df_split_type["labels"] = df_split_type["code"]
            df = pd.concat([df, df_split_type])
        df["filename"] = df["file"]
        return df[["sentence", "labels", "split_type", "filename"]].reset_index(drop=True)

    def assign_clusters_to_label(self, label: str) -> List[str]:
        category = label[:3]
        for cluster in CLUSTERS:
            if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
                return [cluster]

    def download_corpus(self):
        logger.info("Downloading CANTEMIST corpus")
        url = "https://zenodo.org/record/3978041/files/cantemist.zip?download=1"
        download_corpus(self.corpuses_path, "cantemist", url, create_containing_folder=True)
        logger.info("Downloading CIE 3 O descriptions")
        url = "https://eciemaps.mscbs.gob.es/ecieMaps/download?name=2018_CIEO31_TABLA_%20REFERENCIA_con_6_7_caracteres_final_20180111_5375350050755186721_7033700654037542595.xlsx"
        download_corpus(self.corpuses_path, "cie-o-3-codes", url, file_type="xlsx")

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        def read_brat_file(path_file: Path):
            with open(path_file, "r") as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    line = lines[i]
                    annotator_notes = lines[i + 1]
                    off0 = int(line.split("\t")[1].split(" ")[1])
                    off1 = int(line.split("\t")[1].split(" ")[2])
                    sentence = line.split("\t")[2].strip()
                    label = annotator_notes.split("\t")[2].strip()
                    yield (off0, off1, sentence, label)
                    i += 2

        df = pd.DataFrame()
        for split_type in ["train", "dev1", "dev2"]:
            path = (
                self.corpuses_path
                / self.corpus
                / ("train-set" if split_type == "train" else f"{split_type[:-1]}-set{split_type[-1]}")
                / "cantemist-norm"
            )
            document_names = set([doc.split(".")[0] for doc in os.listdir(path)])
            data = {"filename": [], "off0": [], "off1": [], "mention_text": [], "label": []}
            for document in document_names:
                for off0, off1, sentence, label in read_brat_file(path / f"{document}.ann"):
                    data["filename"].append(document)
                    data["off0"].append(off0)
                    data["off1"].append(off1)
                    data["mention_text"].append(sentence)
                    data["label"].append(label)
            df_split_type = pd.DataFrame(data=data)
            df = pd.concat([df, df_split_type])
        return df[["filename", "label", "mention_text", "off0", "off1"]].reset_index(drop=True)

    def process_augmentation_descriptions(self) -> pd.DataFrame:
        df = pd.read_excel(self.corpuses_path / "cie-o-3-codes.xlsx", sheet_name=1)
        df = pd.melt(df, id_vars=["codigo"], value_vars=["Descriptor Completo", "Descriptor Abreviado"])
        df["label"] = df["codigo"]
        df["description"] = df["value"]
        return df[["label", "description"]].reset_index(drop=True)


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
