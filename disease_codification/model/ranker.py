from email.mime import base
from json import load
import os
import statistics
from pathlib import Path
from typing import List
from disease_codification.custom_io import create_dir_if_dont_exist, load_pickle, save_as_pickle

from disease_codification.flair_utils import read_corpus
from flair.data import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

from disease_codification.gcp import download_blob_file, upload_blob_file, storage_client
from disease_codification.model.matcher import Matcher


class Ranker:
    def __init__(
        self, indexers_path: Path, models_path: Path, indexer: str, cluster_classifier={}, cluster_label_classifier={}
    ):
        self.indexers_path = indexers_path
        self.models_path = models_path
        self.indexer = indexer
        self.clusters = {filename.split("_")[1] for filename in os.listdir(indexers_path / indexer / "ranker")}
        self.cluster_classifier = cluster_classifier
        self.cluster_label_binarizer = cluster_label_classifier
        self.mappings = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")
        Ranker.create_directories(models_path, indexer)

    @classmethod
    def create_directories(cls, models_path, indexer):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / "ranker")

    @classmethod
    def load(cls, indexers_path: Path, models_path: Path, indexer: str, load_from_gcp: bool = True):
        cls.create_directories(models_path, indexer)
        clusters = set(load_pickle(indexers_path / indexer / "mappings.pickle").values())
        cluster_classifier = {}
        cluster_label_binarizer = {}
        for cluster in clusters:
            label_binarizer_path = models_path / indexer / "ranker" / f"label_binarizer_{cluster}.pickle"
            classifier_path = models_path / indexer / "ranker" / f"classifier_{cluster}.pickle"
            if load_from_gcp:
                download_blob_file(f"{indexer}/ranker/label_binarizer_{cluster}.pickle", label_binarizer_path)
                download_blob_file(f"{indexer}/ranker/classifier_{cluster}.pickle", classifier_path)
            cluster_classifier[cluster] = load_pickle(classifier_path)
            cluster_label_binarizer[cluster] = load_pickle(label_binarizer_path)
        return cls(indexers_path, models_path, indexer, cluster_classifier, cluster_label_binarizer)

    def _read_sentences(self, cluster: str, split_types: List[str]) -> List[Sentence]:
        filename = f"ranker_{cluster}"
        corpus = read_corpus(self.indexers_path / self.indexer / "ranker", filename)
        sentences = []
        for split_type in split_types:
            sentences += list(getattr(corpus, split_type))
        has_incorrect = (
            self.indexers_path / self.indexer / "incorrect-matcher" / f"incorrect_{cluster}_train.txt"
        ).exists()
        if has_incorrect and "train" in split_types:
            incorrect_matcher_corpus = read_corpus(
                self.indexers_path / self.indexer / "incorrect-matcher", f"incorrect_{cluster}", only_train=True
            )
            sentences += list(incorrect_matcher_corpus.train)
        return sentences

    def _set_multi_label_binarizer(self, cluster, sentences):
        mlb = MultiLabelBinarizer()
        labels_cluster = {f"<{label}>" for label, cluster_label in self.mappings.items() if cluster_label == cluster}
        mlb.fit([list(labels_cluster) + ["<incorrect-matcher>"]])
        self.cluster_label_binarizer[cluster] = mlb

    def _get_labels_matrix(self, cluster, sentences):
        mlb = self.cluster_label_binarizer[cluster]
        labels_matrix = mlb.transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        return labels_matrix

    def _get_embeddings(self, sentences: List[Sentence]):
        return [s.to_original_text() for s in sentences]

    def train(self, upload_to_gcp: bool = False, split_types_train=["train", "dev"]):
        len_clusters = len(self.clusters)
        for i, cluster in enumerate(self.clusters):
            print(f"Training for cluster {cluster} - {i}/{len_clusters}")
            sentences = self._read_sentences(cluster, split_types_train)
            embeddings = self._get_embeddings(sentences)
            self._set_multi_label_binarizer(cluster, sentences)
            labels = self._get_labels_matrix(cluster, sentences)
            if not labels.any() or not sentences:
                raise Exception
            pipe = Pipeline([("tfidf", TfidfVectorizer()), ("xgboost", XGBClassifier(eval_metric="logloss"))])
            clf = OneVsRestClassifier(pipe).fit(embeddings, labels)
            self.cluster_classifier[cluster] = clf
        print("Training Complete")
        self.save()
        if upload_to_gcp:
            self.upload_to_gcp()

    def predict(self, sentences):
        print("Predicting ranker")
        for cluster in self.clusters:
            print(cluster)
            embeddings = self._get_embeddings(sentences)
            predictions = self.cluster_classifier[cluster].predict_proba(embeddings)
            classes = self.cluster_label_binarizer[cluster].classes_
            for sentence, prediction in zip(sentences, predictions):
                for i, label in enumerate(classes):
                    sentence.add_label("ranker", label, prediction[i])

    def eval_weighted(self, split_types: List[str] = ["dev", "test"]):
        print(f"Calculating MAP Weighted for {split_types}")
        map_clusters = {}
        for split_type in split_types:
            for cluster in self.clusters:
                sentences = self._read_sentences(cluster, [split_type])
                embeddings = self._get_embeddings(sentences)
                predictions = self.cluster_classifier[cluster].predict_proba(embeddings)
                labels_matrix = self._get_labels_matrix(cluster, sentences)
                avg_precs = []
                for y_true, y_scores in zip(labels_matrix, predictions):
                    avg_precs.append(average_precision_score(y_true, y_scores))
                map_clusters[cluster] = (statistics.mean(avg_precs), len(embeddings))
            weighted_map = sum(map_stat * d_points for map_stat, d_points in map_clusters.values()) / sum(
                d_points for _, d_points in map_clusters.values()
            )
            print(map_clusters)
            print(weighted_map)

    def save(self):
        print("Saving model")
        base_path = self.models_path / self.indexer / "ranker"
        for cluster in self.clusters:
            save_as_pickle(self.cluster_label_binarizer[cluster], base_path / f"label_binarizer_{cluster}.pickle")
            save_as_pickle(self.cluster_classifier[cluster], base_path / f"classifier_{cluster}.pickle")

    def upload_to_gcp(self):
        base_path = self.models_path / self.indexer / "ranker"
        for cluster in self.clusters:
            upload_blob_file(
                base_path / f"label_binarizer_{cluster}.pickle",
                f"{self.indexer}/ranker/label_binarizer_{cluster}.pickle",
            )
            upload_blob_file(
                base_path / f"classifier_{cluster}.pickle", f"{self.indexer}/ranker/classifier_{cluster}.pickle"
            )
