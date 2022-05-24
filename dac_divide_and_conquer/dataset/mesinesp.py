from collections import defaultdict
import enum
import json
from pathlib import Path
from typing import List
from xml.etree.ElementTree import ElementTree

import pandas as pd
from lxml import etree
from dac_divide_and_conquer.corpora_downloader import download_corpus
from dac_divide_and_conquer.dataset.base import DACCorpus
from dac_divide_and_conquer import logger


class MesinespSubtask(enum.Enum):
    abstracts = "abstracts"
    clinical_trials = "clinical_trials"


class MESINESPCorpus(DACCorpus):
    def __init__(self, data_path: Path, subtask: MesinespSubtask):
        self.subtask = subtask
        super().__init__(f"mesinesp_{subtask}", data_path)
        self._parse_xml_and_create_label_map_()

    def process_corpus(self):
        corpus_path = self.corpuses_path / (
            "mesinesp-st1" if self.subtask == MesinespSubtask.clinical_trials else "mesinesp-st2"
        )
        split_type_maps = {"train": "Train", "dev": "Development", "test": "Test"}
        split_type_maps_2 = {
            MesinespSubtask.clinical_trials: {
                "train": "training_set_subtrack1_all",
                "test": "test_set_subtrack1",
                "dev": "development_set_subtrack1",
            },
            MesinespSubtask.abstracts: {
                "train": "training_set_subtrack2",
                "test": "test_set_subtrack2",
                "dev": "development_set_subtrack2",
            },
        }
        df = pd.DataFrame(columns=["sentence", "labels"])
        for split_type in ["train", "dev", "test"]:
            with open(
                corpus_path
                / (
                    "Subtrack1-Scientific_Literature"
                    if self.subtask == MesinespSubtask.clinical_trials
                    else "Subtrack2-Clinical_Trials"
                )
                / split_type_maps[split_type]
                / f"{split_type_maps_2[self.subtask][split_type]}.json"
            ) as f:
                data = json.load(f)
                p_data = defaultdict(list)
                for row in data["articles"]:
                    p_data["sentence"].append(row["title"] + row["abstractText"])
                    for label in row.get("decsCodes", []):
                        assert label.startswith("D")
                    p_data["labels"].append(row.get("decsCodes", []))
                df_split_type = pd.DataFrame(data=p_data)
                df_split_type["split_type"] = [split_type] * df_split_type.shape[0]
                df = pd.concat([df, df_split_type])
        return df[["sentence", "labels", "split_type"]].reset_index(drop=True)

    def _parse_xml_and_create_label_map_(self):
        tree: ElementTree = etree.parse(self.corpuses_path / "desc2022.xml")
        not_mesh = 0
        query = f"//DescriptorRecord"
        query_result = tree.xpath(query)
        self.xml_mappings = {}
        for i, result in enumerate(query_result):
            ui = result.find("DescriptorUI").text
            tree_locations = result.xpath("./TreeNumberList/TreeNumber/text()")
            if tree_locations:
                clusters = []
                for r in tree_locations:
                    if r[:11] in SPECIAL_ONES:
                        clusters.append(r[:11])
                    elif r[:7] in SPECIAL_ONES2:
                        clusters.append(r[:7])
                    elif r[:1] in SPECIAL_ONES3:
                        clusters.append(r[:1])
                    else:
                        clusters.append(r[:3])
                clusters = set(clusters)
            else:
                clusters = ["not-mesh"]
                not_mesh += 1
            self.xml_mappings[ui] = list(clusters)

    def assign_clusters_to_label(self, label: str) -> List[str]:
        return self.xml_mappings.get(label, ["not-mesh"])

    def download_corpus(self):
        logger.info("Downloading MESINESP subtrack 2 corpus")
        url = "https://zenodo.org/record/5602914/files/Subtrack2-Clinical_Trials.zip?download=1"
        download_corpus(self.corpuses_path, "mesinesp-st2", url, create_containing_folder=True)
        logger.info("Downloading DECS Codes")
        url = "https://zenodo.org/record/4707104/files/DeCS2020.tsv?download=1"
        download_corpus(self.corpuses_path, "decs-codes", url, file_type="tsv")
        logger.info("Downloading DECS Codes Hierarchy")
        url = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2022.zip"
        download_corpus(self.corpuses_path, "desc2022", url)
        logger.info("Downloading MESINESP subtrack 1 corpus")
        url = "https://zenodo.org/record/5602914/files/Subtrack1-Scientific_Literature.zip?download=1"
        download_corpus(self.corpuses_path, "mesinesp-st1", url, create_containing_folder=True)


SPECIAL_ONES = [
    "D12.776.124",
    "D12.776.157",
    "D12.776.377",
    "D12.776.395",
    "D12.776.467",
    "D12.776.476",
    "D12.776.543",
    "E01.370.225",
    "E01.370.350",
]

SPECIAL_ONES2 = [
    "B01.050",
    "B01.875",
    "C10.228",
    "C23.550",
    "C23.888",
    "D02.063",
    "D02.065",
    "D02.092",
    "D02.241",
    "D02.445",
    "D02.886",
    "D03.383",
    "D03.633",
    "E05.200",
    "E05.318",
    "D12.644",
    "D12.776",
    "M01.526",
    "N05.715",
    "N06.850",
]

SPECIAL_ONES3 = ["V"]
