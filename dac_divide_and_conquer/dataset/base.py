import itertools
import json
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from pathlib import Path
import re
import statistics
from typing import Dict, List

import numpy as np
import pandas as pd
from dac_divide_and_conquer import logger
from dac_divide_and_conquer.corpora_downloader import create_directories
from dac_divide_and_conquer.custom_io import create_dir_if_dont_exist, save_as_pickle, write_fasttext_file
from dac_divide_and_conquer.dataset import Augmentation


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


class DACCorpus(ABC):
    def __init__(
        self,
        corpus: str,
        data_path: Path,
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
        first_n_digits_f1: int = 0,
    ):
        self.corpus = corpus
        self.corpuses_path, self.indexers_path, self.models_path = create_directories(data_path)
        self.indexer_path = self.indexers_path / corpus
        self.augmentation_matcher = augmentation_matcher
        self.augmentation_ranker = augmentation_ranker
        self.augmentation_corpus = augmentation_corpus
        self.first_n_digits_f1 = first_n_digits_f1
        path_filenames = self.indexer_path / "filenames.json"
        self.filenames = {} if not path_filenames.exists() else read_json(path_filenames)
        self.__create_paths__()

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
        logger.info("Creating corpuses for use by models")
        self.df_sentences = self.process_corpus()
        self.labels = set(itertools.chain.from_iterable([ls for ls in self.df_sentences["labels"].to_list()]))
        self.mappings_label_to_cluster = self.clusterize()
        self.cluster_counts = self._get_cluster_counts_()
        self.clusters = set(itertools.chain.from_iterable(self.mappings_label_to_cluster.values()))
        self.__store_filenames__()
        save_as_pickle(self.mappings_label_to_cluster, self.indexer_path / "mappings.pickle")
        with open(self.indexer_path / "filenames.json", "w") as f:
            json.dump(self.filenames, f)
        self.__create_corpus__()
        self.__create_matcher_corpus__()
        self.__create_ranker_corpus__()
        self.__create_augmentation_corpora__()
        self.log_info_of_labels()
        self._create_corpus_stats_file()

    @abstractmethod
    def process_corpus(self) -> pd.DataFrame:
        """
        :return: A pandas DataFrame with the following columns:
            sentence: String with the document text
            labels: List of strings with the document labels
            split_type: train, test or dev
            filename: Filename or id of the document
        """
        pass

    @abstractmethod
    def assign_clusters_to_label(self, label: str) -> List[str]:
        """
        :param label: Label string
        :return: A list of the label clusters
        """
        pass

    def download_corpus(self):
        return

    def process_augmentation_ne_mentions(self) -> pd.DataFrame:
        """
        :return: A pandas DataFrame with the following columns:
            filename: Filename or id of the document
            mention_text: Mention string of an entity
            label: Label of the mentioned entity
            off0: Start offset of the mention
            off1: End offset of the mention
        """
        return pd.DataFrame()

    def process_augmentation_descriptions(self) -> pd.DataFrame:
        """
        :return: A pandas DataFrame with the following columns:
            label: Label string
            description: Text description of label
        """
        return pd.DataFrame()

    def clusterize(self) -> Dict[str, List[str]]:

        return {label: self._assign_clusters_to_label_(label) for label in self.labels}

    def _assign_clusters_to_label_(self, label: str) -> List[str]:
        clusters_assigned = self.assign_clusters_to_label(label)
        assert type(clusters_assigned) == list
        return clusters_assigned

    def __process_augmentation_ne_mentions__(self) -> pd.DataFrame:
        df = self.process_augmentation_ne_mentions()
        if df.empty:
            return pd.DataFrame()
        df["labels"] = df["label"].apply(lambda x: [x])
        df = df.drop_duplicates(subset=["mention_text", "label"])
        df["sentence"] = df["mention_text"]
        return df[["sentence", "labels"]]

    def process_augmentation_ne_sentences(self) -> pd.DataFrame:
        sentences_df = pd.DataFrame()
        documents_df = self.process_corpus()
        ne_mentions_df = self.process_augmentation_ne_mentions()
        if ne_mentions_df.empty:
            return pd.DataFrame()
        sentences = []
        labels = []
        disgregated = 0
        for _, row_documents in documents_df.iterrows():
            if row_documents["split_type"] == "test":
                continue
            document_sentences = [s for s in re.split("\.|\n", row_documents["sentence"])]
            document_labels = ne_mentions_df[ne_mentions_df["filename"] == row_documents["filename"]]
            end_char = 0
            for sentence in document_sentences:
                start_char = end_char
                end_char += len(sentence) + 1
                labels_sentence = []
                for _, row_label in document_labels.iterrows():
                    if start_char <= int(row_label["off0"]) and end_char >= int(row_label["off1"]):
                        try:
                            mention_text_words = (
                                row_label["mention_text"].replace(".", "").replace(",", "").replace("\n", "").split(" ")
                            )
                            assert all(
                                word in sentence.replace(".", "").replace(",", "") for word in mention_text_words
                            )
                            labels_sentence.append(row_label["label"])
                        except AssertionError:
                            logger.info(f"Label searching: {row_label['mention_text']}")
                            logger.info(f"{start_char}-{end_char}: {sentence}")
                            logger.warning("Possible error in ne_mention processing function")
                            disgregated += 1
                if sentence:
                    sentences.append(sentence.strip())
                    labels.append(labels_sentence)
        if disgregated:
            logger.warn(f"{disgregated} labels are in more than one sentence so werent added")
        sentences_df["sentence"] = sentences
        sentences_df["labels"] = labels
        return sentences_df

    def process_augmentation_ne_stripped(self) -> pd.DataFrame:
        ne_mentions_df = self.process_augmentation_ne_mentions()
        if ne_mentions_df.empty:
            return pd.DataFrame()
        group = ne_mentions_df.groupby("filename")
        labels = group["label"].apply(set)
        sentences = group["mention_text"].apply(list).apply(" ".join)
        df = labels.to_frame("label").join(sentences.to_frame("mention_text"))
        df["sentence"] = df["mention_text"]
        df["labels"] = df["label"]
        df = df[["sentence", "labels"]].reset_index(drop=True)
        return df

    def __store_filenames__(self):
        for split_type in ["train", "test", "dev"]:
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            self.filenames[split_type] = [l for l in df_split_type["filename"].to_list()]

    def __process_augmentation_descriptions__(self):
        df = self.process_augmentation_descriptions()
        if df.empty:
            return df
        for label in self.labels:
            if label not in df["label"].values:
                df.loc[len(df.index)] = [label, "Sin descripcion"]
        df = df[df["label"].isin(self.labels)]
        df = df[df["description"] != "Sin descripcion"]
        df = df.dropna(subset=["description"])
        df = df.drop_duplicates(["description", "label"])
        df["labels"] = df["label"].apply(lambda x: [x])
        df["sentence"] = df["description"]
        return df[["sentence", "labels"]].reset_index(drop=True)

    def _get_cluster_counts_(self):
        clusters = np.array(list(itertools.chain.from_iterable(self.mappings_label_to_cluster.values())))
        unique, counts = np.unique(clusters, return_counts=True)
        return dict(zip(unique, [int(c) for c in counts]))

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
                    ls_parsed = [l for l in ls if cluster in self.mappings_label_to_cluster[l]]
                    labels.append(ls_parsed)
                if len(sentences) > 50000 and self.cluster_counts[cluster] > 200:
                    print(cluster)
                write_fasttext_file(
                    sentences, labels, self.indexer_path / "ranker" / f"ranker_{cluster}_{split_type}.txt"
                )

    def __labels_to_clusters__(self, ls):
        return set(itertools.chain.from_iterable(self.mappings_label_to_cluster[l] for l in ls))

    def __is_in_cluster__(self, cluster, clusters):
        return cluster in clusters

    def __is_in_labels__(self, label, ls):
        return label in ls

    def log_info_of_labels(self):
        logger.info(f"Amount of clusters: {len(self.clusters)}")
        logger.info(f"Amount of labels: {len(self.mappings_label_to_cluster)}")
        counts = [c for c in self.cluster_counts.values()]
        logger.info(f"Amount of labels multi-cluster: {sum(counts)}")
        logger.info(json.dumps(self.cluster_counts, indent=2))
        logger.info(f"Mean: {np.mean(counts)}")
        logger.info(f"Std: {np.std(counts)}")

    def __create_augmentation_corpora__(self):
        logger.info("Matcher")
        augs_map = {
            Augmentation.ner_mention: self.__process_augmentation_ne_mentions__,
            Augmentation.descriptions_labels: self.__process_augmentation_descriptions__,
            Augmentation.ner_sentence: self.process_augmentation_ne_sentences,
            Augmentation.ner_stripped: self.process_augmentation_ne_stripped,
        }
        for aug in self.augmentation_matcher:
            logger.info(aug)
            df = augs_map[aug]()
            if df.empty:
                logger.info(f"{aug} not implemented")
                continue
            sentences = df["sentence"].to_list()
            labels = [self.__labels_to_clusters__(ls) for ls in df["labels"].to_list()]
            write_fasttext_file(
                sentences,
                labels,
                self.indexer_path / f"matcher-{aug}" / f"matcher_train.txt",
            )
        logger.info("Ranker")
        for aug in self.augmentation_ranker:
            logger.info(aug)
            df = augs_map[aug]()
            if df.empty:
                logger.info(f"{aug} not implemented")
                continue
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
        logger.info("Corpus")
        for aug in self.augmentation_corpus:
            logger.info(aug)
            df = augs_map[aug]()
            if df.empty:
                logger.info(f"{aug} not implemented")
                continue
            write_fasttext_file(
                df["sentence"].to_list(),
                df["labels"].to_list(),
                self.indexer_path / f"corpus-{aug}" / f"corpus_train.txt",
            )

    def _create_corpus_stats_file(self):
        stats_corpus = {}
        for split_type in ["train", "test", "dev"]:
            stats_split = {}
            df_split_type = self.df_sentences[self.df_sentences["split_type"] == split_type]
            stats_split["documents"] = df_split_type.shape[0]
            stats_split["avg_document_len"] = statistics.mean([len(sentence) for sentence in df_split_type["sentence"]])
            stats_split["avg_labels_per_document"] = statistics.mean(
                [len(labels) for labels in df_split_type["labels"]]
            )
            stats_split["avg_clusters_per_document"] = statistics.mean(
                [len(self.__labels_to_clusters__(ls)) for ls in df_split_type["labels"].to_list()]
            )
            stats_corpus[split_type] = stats_split
        logger.info(stats_corpus)
        with open(self.indexer_path / "stats.json", "w") as f:
            json.dump(stats_corpus, f)
