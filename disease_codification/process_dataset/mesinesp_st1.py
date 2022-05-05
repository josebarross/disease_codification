from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import List
from xml.etree.ElementTree import ElementTree

import pandas as pd
from lxml import etree
from disease_codification import logger


def process_sentence(corpuses_path: Path) -> pd.DataFrame:
    corpus_path = corpuses_path / "mesinesp-st1"
    split_type_maps = {"train": "Train", "dev": "Development", "test": "Test"}
    split_type_maps_2 = {
        "train": "training_set_subtrack1_all",
        "test": "test_set_subtrack1",
        "dev": "development_set_subtrack1",
    }
    df = pd.DataFrame(columns=["sentence", "labels"])
    for split_type in ["train", "dev", "test"]:
        with open(
            corpus_path
            / "Subtrack1-Scientific_Literature"
            / split_type_maps[split_type]
            / f"{split_type_maps_2[split_type]}.json"
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
    return df.reset_index()


def process_labels(corpuses_path: Path, for_augmentation=True) -> pd.DataFrame:
    df_sentence = process_sentence(corpuses_path)
    sentences_labels = list(set(itertools.chain.from_iterable([label for label in df_sentence["labels"].values])))
    df = pd.DataFrame()
    df["labels"] = sentences_labels
    return df.reset_index()


def cluster_assigner(corpuses_path: Path, labels: List[str]):
    tree: ElementTree = etree.parse(corpuses_path / "desc2022.xml")
    not_mesh = 0
    more_than_one_cluster = 0
    query = f"//DescriptorRecord"
    query_result = tree.xpath(query)
    mappings = {}
    for i, result in enumerate(query_result):
        ui = result.find("DescriptorUI").text
        tree_locations = result.xpath("./TreeNumberList/TreeNumber/text()")
        if tree_locations:
            special_ones = [
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
            special_ones2 = [
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
            special_ones3 = ["V"]
            clusters = []
            for r in tree_locations:
                if r[:11] in special_ones2:
                    clusters.append(r[:11])
                elif r[:7] in special_ones:
                    clusters.append(r[:7])
                elif r[:1] in special_ones3:
                    clusters.append(r[:1])
                else:
                    clusters.append(r[:3])
            clusters = set(clusters)
        else:
            clusters = ["not-mesh"]
            not_mesh += 1
        if len(clusters) > 1:
            more_than_one_cluster += 1
        mappings[ui] = list(clusters)
    clusters_assigned = [mappings.get(label, ["not-mesh"]) for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned
