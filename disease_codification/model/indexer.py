import itertools
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, save_as_pickle, write_fasttext_file
from disease_codification.process_dataset.mapper import Augmentation, mapper_process_function


class Indexer:
    def __init__(
        self,
        corpus: str,
        corpuses_path: Path,
        indexers_path: Path,
        augmentation_matcher: List[Augmentation] = [Augmentation.descriptions_labels, Augmentation.ner_sentence],
        augmentation_ranker: List[Augmentation] = [
            Augmentation.descriptions_labels,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.ner_mention,
        ],
        augmentation_corpus: List[Augmentation] = [
            Augmentation.descriptions_labels,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.ner_mention,
        ],
        subset: int = None,
        multi_cluster: bool = False,
    ):
        self.corpus = corpus
        self.corpuses_path = corpuses_path
        self.indexer_path = indexers_path / corpus
        self.augmentation_matcher = augmentation_matcher
        self.augmentation_ranker = augmentation_ranker
        self.augmentation_corpus = augmentation_corpus
        self.df_sentences = mapper_process_function[corpus]["sentence"](corpuses_path)
        self.df_labels = mapper_process_function[corpus]["labels"](corpuses_path, for_augmentation=False)
        self.multi_cluster = multi_cluster
        self.__create_paths__()
        if subset:
            self.df_sentences = self.df_sentences[:subset]

    def __create_paths__(self):
        create_dir_if_dont_exist(self.indexer_path)
        create_dir_if_dont_exist(self.indexer_path / "corpus")
        create_dir_if_dont_exist(self.indexer_path / "matcher")
        create_dir_if_dont_exist(self.indexer_path / "ranker")
        for aug in self.augmentation_matcher:
            create_dir_if_dont_exist(self.indexer_path / f"matcher-{aug}")
        for aug in self.augmentation_ranker:
            create_dir_if_dont_exist(self.indexer_path / f"ranker-{aug}")
        for aug in self.augmentation_corpus:
            create_dir_if_dont_exist(self.indexer_path / f"corpus-{aug}")

    def create_corpuses(self):
        print("Creating corpuses for use by models")
        self.__filter_labels_only_in_dataset__()
        self.mappings_label_to_cluster = self.__clusterize__()
        if self.multi_cluster:
            self.clusters = set(itertools.chain.from_iterable(self.mappings_label_to_cluster.values()))
        else:
            self.clusters = set(self.mappings_label_to_cluster.values())
        save_as_pickle(self.mappings_label_to_cluster, self.indexer_path / "mappings.pickle")
        self.__create_corpus__()
        self.__create_matcher_corpus__()
        self.__create_ranker_corpus__()
        self.__create_augmentation_corpora__()
        self.print_info_of_labels()

    def __filter_labels_only_in_dataset__(self):
        labels_in_sentences = set(itertools.chain.from_iterable([ls for ls in self.df_sentences["labels"].to_list()]))
        labels_in_descriptions = self.df_labels["labels"].to_list()
        print(f"Original shape: {len(labels_in_descriptions)}")
        self.all_labels = [label for label in labels_in_descriptions if label in labels_in_sentences]
        print(f"Final shape: {len(self.all_labels)}")

    def __clusterize__(self) -> Dict[str, str]:
        clusters_assigned = mapper_process_function[self.corpus]["cluster_assigner"](
            self.corpuses_path, self.all_labels
        )
        mappings = dict(zip(self.all_labels, clusters_assigned))
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
        self.df_sentences["clusters"] = [
            self.__labels_to_clusters__(ls) for ls in self.df_sentences["labels"].to_list()
        ]
        for cluster in self.clusters:
            is_in_cluster = self.df_sentences["clusters"].apply(partial(self.__is_in_cluster__, cluster))
            df_cluster = self.df_sentences[is_in_cluster]
            for split_type in ["train", "test", "dev"]:
                df_split_type = df_cluster[df_cluster["split_type"] == split_type]
                sentences = df_split_type["sentence"].tolist()
                labels = []
                for ls in df_split_type["labels"].to_list():
                    ls_parsed = (
                        [l for l in ls if cluster in self.mappings_label_to_cluster[l]]
                        if self.multi_cluster
                        else [label for label in ls if self.mappings_label_to_cluster[label] == cluster]
                    )
                    labels.append(ls_parsed)
                write_fasttext_file(
                    sentences, labels, self.indexer_path / "ranker" / f"ranker_{cluster}_{split_type}.txt"
                )

    def __labels_to_clusters__(self, ls):
        if self.multi_cluster:
            return set(itertools.chain.from_iterable(self.mappings_label_to_cluster[l] for l in ls))
        else:
            return {self.mappings_label_to_cluster[l] for l in ls}

    def __is_in_cluster__(self, cluster, clusters):
        return cluster in clusters

    def __is_in_labels__(self, label, ls):
        return label in ls

    def print_info_of_labels(self):
        if self.multi_cluster:
            clusters = np.array(list(itertools.chain.from_iterable(self.mappings_label_to_cluster.values())))
        else:
            clusters = np.array(list(self.mappings_label_to_cluster.values()))
        print(f"Amount of labels: {len(self.mappings_label_to_cluster)}")
        unique, counts = np.unique(clusters, return_counts=True)
        print(dict(zip(unique, counts)))
        print(f"Mean: {np.mean(counts)}")
        print(f"Std: {np.std(counts)}")

    def __create_augmentation_corpora__(self):
        print("Matcher")
        for aug in self.augmentation_matcher:
            if aug not in mapper_process_function[self.corpus]:
                print(f"{aug} not implemented")
                continue
            print(aug)
            df = mapper_process_function[self.corpus][aug](self.corpuses_path)
            sentences = df["sentence"].to_list()
            labels = [self.__labels_to_clusters__(ls) for ls in df["labels"].to_list()]
            write_fasttext_file(
                sentences,
                labels,
                self.indexer_path / f"matcher-{aug}" / f"matcher_train.txt",
            )
        print("Ranker")
        for aug in self.augmentation_ranker:
            print(aug)
            if aug not in mapper_process_function[self.corpus]:
                print(f"{aug} not implemented")
                continue
            df = mapper_process_function[self.corpus][aug](self.corpuses_path)
            for cluster in self.clusters:
                cluster_sentences = [
                    sentence
                    for sentence, ls in zip(df["sentence"].to_list(), df["labels"].to_list())
                    if self.__is_in_cluster__(cluster, self.__labels_to_clusters__(ls))
                ]
                cluster_labels = [
                    [l for l in ls if self.__is_in_cluster__(cluster, self.__labels_to_clusters__([l]))]
                    for ls in df["labels"].to_list()
                    if self.__is_in_cluster__(cluster, self.__labels_to_clusters__(ls))
                ]
                write_fasttext_file(
                    cluster_sentences,
                    cluster_labels,
                    self.indexer_path / f"ranker-{aug}" / f"{cluster}_train.txt",
                )
        print("Corpus")
        for aug in self.augmentation_corpus:
            if aug not in mapper_process_function[self.corpus]:
                print(f"{aug} not implemented")
                continue
            print(aug)
            df = mapper_process_function[self.corpus][aug](self.corpuses_path)
            write_fasttext_file(
                df["sentence"].to_list(),
                df["labels"].to_list(),
                self.indexer_path / f"corpus-{aug}" / f"corpus_train.txt",
            )
