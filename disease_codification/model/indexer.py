from enum import Enum
import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, save_as_pickle, write_fasttext_file
from disease_codification.process_dataset.cie10 import cluster_assigner, get_cie10_df
from disease_codification.process_dataset.mapper import mapper_process_function


class ClusteringType(Enum):
    first_letter = "first_letter"
    cie10 = "cie10"


class Indexer:
    def __init__(
        self, corpus: str, corpuses_path: Path, indexers_path: Path, args_for_clustering: dict = {}, subset: int = None
    ):
        self.corpus = corpus
        self.corpuses_path = corpuses_path
        self.indexer_path = indexers_path / corpus
        self.args_for_clustering = args_for_clustering
        self.df_sentences, self.df_labels = mapper_process_function[corpus](corpuses_path)
        self.df_labels_dict = dict(zip(self.df_labels["label"], self.df_labels["spanish_description"]))
        if subset:
            self.df_sentences = self.df_sentences[:subset]
        self.__create_paths__()

    def __create_paths__(self):
        create_dir_if_dont_exist(self.indexer_path)
        create_dir_if_dont_exist(self.indexer_path / "corpus")
        create_dir_if_dont_exist(self.indexer_path / "matcher")
        create_dir_if_dont_exist(self.indexer_path / "ranker")
        create_dir_if_dont_exist(self.indexer_path / "description")

    def create_corpuses(self):
        self.__filter_labels_only_in_dataset__()
        self.mappings_label_to_cluster = self.__clusterize__()
        self.clusters = set(self.mappings_label_to_cluster.values())
        save_as_pickle(self.mappings_label_to_cluster, self.indexer_path / "mappings.pickle")
        self.__create_corpus__()
        self.__create_matcher_corpus__()
        self.__create_ranker_corpus__()
        self.__create_description_corpus__()
        self.print_info_of_labels()

    def __filter_labels_only_in_dataset__(self):
        print(f"Original shape: {len(self.df_labels_dict)}")
        labels = set(itertools.chain.from_iterable([ls for ls in self.df_sentences["labels"].to_list()]))
        self.df_labels_dict = {k: v for k, v in self.df_labels_dict.items() if k in labels}
        print(f"Final shape: {len(self.df_labels_dict)}")

    def __clusterize__(self, type: ClusteringType = ClusteringType.cie10) -> Dict[str, str]:
        if type == ClusteringType.first_letter:
            mappings = {label: label[0] for label in self.df_labels_dict.keys()}
        elif type == ClusteringType.cie10:
            clusters_assigned = cluster_assigner(self.df_labels_dict.keys())
            mappings = dict(zip(self.df_labels_dict.keys(), clusters_assigned))
        print(mappings)
        return mappings

    def __create_corpus__(self):
        for split_type in ["train", "test", "dev"]:
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            sentences = df_split_type["sentence"].tolist()
            labels = [ls for ls in df_split_type["labels"].to_list()]
            write_fasttext_file(sentences, labels, self.indexer_path / "corpus" / f"corpus_{split_type}.txt")

    def __create_matcher_corpus__(self):
        for split_type in ["train", "test", "dev"]:
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            sentences = df_split_type["sentence"].tolist()
            labels = [self.__labels_to_clusters__(ls) for ls in df_split_type["labels"].to_list()]
            write_fasttext_file(sentences, labels, self.indexer_path / "matcher" / f"matcher_{split_type}.txt")

    def __create_ranker_corpus__(self):
        self.df_sentences["clusters"] = self.df_sentences["labels"].apply(self.__labels_to_clusters__)
        for cluster in self.clusters:
            is_in_cluster = self.df_sentences["clusters"].apply(partial(self.__is_in_cluster__, cluster))
            df_cluster = self.df_sentences[is_in_cluster]
            for split_type in ["train", "test", "dev"]:
                df_split_type = df_cluster[df_cluster["split_type"] == split_type]
                sentences = df_split_type["sentence"].tolist()
                labels = []
                for ls in df_split_type["labels"].to_list():
                    labels.append([label for label in ls if self.mappings_label_to_cluster[label] == cluster])
                write_fasttext_file(
                    sentences, labels, self.indexer_path / "ranker" / f"ranker_{cluster}_{split_type}.txt"
                )

    def __labels_to_clusters__(self, ls):
        return list({self.mappings_label_to_cluster[l] for l in ls})

    def __is_in_cluster__(self, cluster, clusters):
        return cluster in clusters

    def __is_in_labels__(self, label, ls):
        return label in ls

    def print_info_of_labels(self):
        clusters = np.array(list(self.mappings_label_to_cluster.values()))
        unique, counts = np.unique(clusters, return_counts=True)
        print(f"Amount of labels: {clusters.shape}")
        print(dict(zip(unique, counts)))
        print(f"Mean: {np.mean(counts)}")
        print(f"Std: {np.std(counts)}")

    def __create_description_corpus__(self):
        sentences_codi, labels_codi = self._from_codiesp_descriptions()
        sentences_cie, labels_cie = self._from_cie10_descriptions()
        sentences, labels = sentences_codi + sentences_cie, labels_codi + labels_cie
        labels_matcher = [self.__labels_to_clusters__(ls) for ls in labels]
        write_fasttext_file(sentences, labels, self.indexer_path / "description" / "corpus_train.txt")
        write_fasttext_file(sentences, labels_matcher, self.indexer_path / "description" / "matcher_train.txt")
        for cluster in self.clusters:
            sentences_cluster = []
            labels_cluster = []
            for sentence, ls in zip(sentences, labels):
                ls = [l for l in ls if self.mappings_label_to_cluster[l] == cluster]
                if ls:
                    sentences_cluster.append(sentence)
                    labels_cluster.append(ls)
            write_fasttext_file(
                sentences_cluster, labels_cluster, self.indexer_path / "description" / f"ranker_{cluster}_train.txt"
            )

    def _from_codiesp_descriptions(self):
        sentences = []
        labels = []
        for label, sentence in self.df_labels_dict.items():
            if sentence != "Sin descripcion":
                sentences.append(sentence)
                labels.append([label])
        return sentences, labels

    def _from_cie10_descriptions(self):
        all_labels = self.mappings_label_to_cluster.keys()
        cie10_df = get_cie10_df(self.corpuses_path)
        sentences = []
        labels = []
        for _, row in cie10_df.iterrows():
            code = row["code"]
            if len(code.split("-")) == 2:
                start = code.split("-")[0]
                end = code.split("-")[1]
                ls = self._determine_labels_in_range(start, end, all_labels)
            else:
                ls = self._determine_labels_startswith(code, all_labels)
            if ls:
                sentences.append(row["description"])
                labels.append(ls)
        return sentences, labels

    def _determine_labels_startswith(self, code: str, all_labels: List[str]):
        return [label for label in all_labels if label.startswith(code.lower())]

    def _determine_labels_in_range(self, start: str, end: str, all_labels: List[str]):
        start_cat = start[0].lower()
        end_cat = end[0].lower()
        starts_int = start[1:3] if not start[1:3] == "01" else "00"
        ends_int = end[1:3]
        labels_in_range = []
        for label in all_labels:
            label_int = label[1:3]
            label_in_range = (
                (start_cat == end_cat and label.startswith(start_cat) and starts_int <= label_int <= ends_int)
                or (label.startswith(start_cat) and label_int >= starts_int)
                or (label.startswith(end_cat) and label_int <= ends_int)
            )
            if label_in_range:
                labels_in_range.append(label)
        return labels_in_range
