import itertools
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, save_as_pickle, write_fasttext_file
from disease_codification.process_dataset.mapper import mapper_process_function


class Indexer(ABC):
    def __init__(
        self, corpus: str, corpuses_path: Path, indexers_path: Path, args_for_clustering: dict = {}, subset: int = None
    ):
        self.corpus = corpus
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
        create_dir_if_dont_exist(self.indexer_path / "descriptions")
        create_dir_if_dont_exist(self.indexer_path / "incorrect-matcher")

    def create_corpuses(self):
        self.__filter_labels_only_in_dataset__()
        self.mappings_label_to_cluster = self.__clusterize__()
        self.clusters = set(self.mappings_label_to_cluster.values())
        save_as_pickle(self.mappings_label_to_cluster, self.indexer_path / "mappings.pickle")
        self.__create_corpus__()
        self.__create_matcher_corpus__()
        self.__create_ranker_corpus__()
        self.print_info_of_labels()

    def __filter_labels_only_in_dataset__(self):
        print(f"Original shape: {len(self.df_labels_dict)}")
        labels = set(itertools.chain.from_iterable([ls for ls in self.df_sentences["labels"].to_list()]))
        self.df_labels_dict = {k: v for k, v in self.df_labels_dict.items() if k in labels}
        print(f"Final shape: {len(self.df_labels_dict)}")

    @abstractmethod
    def __clusterize__(self) -> Dict[str, str]:
        raise NotImplementedError

    def __create_corpus__(self):
        for split_type in ["train", "test", "dev"]:
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            sentences = df_split_type["sentence"].tolist()
            labels = [ls for ls in df_split_type["labels"].to_list()]
            if split_type == "train":
                sentences += list(self.df_labels_dict.values())
                labels += [[label] for label in self.df_labels_dict.keys()]
            write_fasttext_file(sentences, labels, self.indexer_path / "corpus" / f"corpus_{split_type}.txt")

    def __create_matcher_corpus__(self):
        for split_type in ["train", "test", "dev"]:
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            sentences = df_split_type["sentence"].tolist()
            labels = [self.__labels_to_clusters__(ls) for ls in df_split_type["labels"].to_list()]
            if split_type == "train":
                sentences += list(self.df_labels_dict.values())
                labels += [[self.mappings_label_to_cluster[label]] for label in self.df_labels_dict.keys()]
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
                if split_type == "train":
                    df_cluster_descriptions = {
                        label: description
                        for label, description in self.df_labels_dict.items()
                        if self.mappings_label_to_cluster[label] == cluster
                    }
                    sentences += list(df_cluster_descriptions.values())
                    labels += [[label] for label in df_cluster_descriptions.keys()]
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


class IndexerWithGivenCluster(Indexer):
    def __clusterize__(self) -> Dict[str, str]:
        mappings = {
            label: cluster
            for label, cluster in zip(self.df_labels["label"].to_list(), self.df_labels["cluster"].to_list())
            if label in self.df_labels_dict
        }
        return mappings
