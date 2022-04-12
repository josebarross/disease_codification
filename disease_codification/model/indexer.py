import itertools
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, save_as_pickle, write_fasttext_file
from disease_codification.process_dataset.mapper import Augmentation, mapper_process_function


class ClusteringType(Enum):
    first_letter = "first_letter"
    cie10 = "cie10"


class Indexer:
    def __init__(
        self,
        corpus: str,
        corpuses_path: Path,
        indexers_path: Path,
        clustering_type: ClusteringType = ClusteringType.cie10,
        augmentation_matcher: List[Augmentation] = [Augmentation.descriptions_codiesp_cie, Augmentation.ner_sentence],
        augmentation_ranker: List[Augmentation] = [
            Augmentation.descriptions_codiesp_cie,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.ner_mention,
        ],
        augmentation_corpus: List[Augmentation] = [
            Augmentation.descriptions_codiesp_cie,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.ner_mention,
        ],
        subset: int = None,
    ):
        self.corpus = corpus
        self.corpuses_path = corpuses_path
        self.indexer_path = indexers_path / corpus
        self.clustering_type = clustering_type
        self.augmentation_matcher = augmentation_matcher
        self.augmentation_ranker = augmentation_ranker
        self.augmentation_corpus = augmentation_corpus
        self.df_sentences = mapper_process_function[corpus]["sentence"](corpuses_path)
        self.df_labels = mapper_process_function[corpus]["labels"](corpuses_path, for_augmentation=False)
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
        if self.clustering_type == ClusteringType.first_letter:
            mappings = {label: label[0] for label in self.all_labels}
        elif self.clustering_type == ClusteringType.cie10:
            clusters_assigned = cluster_assigner(self.all_labels)
            mappings = dict(zip(self.all_labels, clusters_assigned))
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
        labels = list({self.mappings_label_to_cluster[l] for l in ls})
        not_found_labels = [i for i, l in enumerate(labels) if not l]
        if not_found_labels:
            print(f"Labels not found: {[ls[i] for i in not_found_labels]}")
        return labels

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

    def __create_augmentation_corpora__(self):
        print("Matcher")
        for aug in self.augmentation_matcher:
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
            print(aug)
            df = mapper_process_function[self.corpus][aug](self.corpuses_path)
            write_fasttext_file(
                df["sentence"].to_list(),
                df["labels"].to_list(),
                self.indexer_path / f"corpus-{aug}" / f"corpus_train.txt",
            )


clusters = {
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
    "s00-t88": "Traumatismos, envenenamientos y algunas otras consecuencias de causas externas",
    "v00-y99": "Causas externas de morbilidad y de mortalidad",
    "z00-z99": "Factores que influyen en el estado de salud y contacto con los servicios de salud",
}


def cluster_assigner(labels: List[str]):
    clusters_assigned = [assign_label(label) for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned


def assign_label(label):
    category = label[:3]
    for cluster in clusters:
        if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
            return cluster
