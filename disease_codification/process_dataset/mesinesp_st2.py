from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import List
from xml.etree.ElementTree import ElementTree

import pandas as pd
from lxml import etree


def process_sentence(corpuses_path: Path) -> pd.DataFrame:
    corpus_path = corpuses_path / "mesinesp-st2"
    split_type_maps = {"train": "Train", "dev": "Development", "test": "Test"}
    split_type_maps_2 = {
        "train": "training_set_subtrack2",
        "test": "test_set_subtrack2",
        "dev": "development_set_subtrack2",
    }
    df = pd.DataFrame(columns=["sentence", "labels"])
    for split_type in ["train", "dev", "test"]:
        with open(
            corpus_path
            / "Subtrack2-Clinical_Trials"
            / split_type_maps[split_type]
            / f"{split_type_maps_2[split_type]}.json"
        ) as f:
            data = json.load(f)
            p_data = defaultdict(list)
            for row in data["articles"]:
                if row.get("decsCodes"):
                    p_data["sentence"].append(row["title"] + row["abstractText"])
                    p_data["labels"].append(row["decsCodes"])
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
        if i % 100 == 0:
            print(i, not_mesh, more_than_one_cluster)
        ui = result.find("DescriptorUI").text
        tree_locations = result.xpath("./TreeNumberList/TreeNumber/text()")
        if tree_locations:
            clusters = set([r[0] for r in tree_locations])
        else:
            clusters = ["not-mesh"]
            not_mesh += 1
        if len(clusters) > 1:
            more_than_one_cluster += 1
        mappings[ui] = list(clusters)
    clusters_assigned = [mappings.get(label, ["not-mesh"]) for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned
