import os
import statistics
from pathlib import Path
from typing import List

from disease_codification.flair_utils import read_corpus
from flair.data import Sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier


class Ranker:
    def __init__(self, indexers_path: Path, models_path: Path, indexer: str):
        self.indexers_path = indexers_path
        self.models_path = models_path
        self.indexer = indexer
        self.clusters = {filename.split("_")[1] for filename in os.listdir(self.data_folder)}
        self.cluster_label_binarizer = {}
        self.cluster_classifier = {}

    def _read_sentences(self, cluster: str, split_types: List[str]) -> List[Sentence]:
        filename = f"ranker_{cluster}"
        corpus = read_corpus(self.indexers_path / self.indexer / "ranker", filename)
        sentences = []
        for split_type in split_types:
            sentences += list(getattr(corpus, split_type))
        return sentences

    def _set_multi_label_binarizer(self, cluster, sentences):
        mlb = MultiLabelBinarizer()
        mlb.fit_transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        self.cluster_label_binarizer[cluster] = mlb

    def _get_labels_matrix(self, cluster, sentences):
        mlb = self.cluster_label_binarizer[cluster]
        labels_matrix = mlb.transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        return labels_matrix

    def _get_embeddings(self, sentences: List[Sentence]):
        return [s.to_original_text() for s in sentences]

    def train(self):
        len_clusters = len(self.clusters)
        for i, cluster in enumerate(self.clusters):
            print(f"Training for cluster {cluster} - {i}/{len_clusters}")
            sentences = self._read_sentences(cluster, ["train"])
            embeddings = self._get_embeddings(sentences)
            self._set_multi_label_binarizer(cluster, sentences)
            labels = self._get_labels_matrix(cluster, sentences)
            if not labels.any() or not sentences:
                raise Exception
            pipe = Pipeline([("tfidf", TfidfVectorizer()), ("xgboost", XGBClassifier())])
            clf = OneVsRestClassifier(pipe).fit(embeddings, labels)
            self.cluster_classifier[cluster] = clf
        print("Training Complete")

    def eval_weighted(self, split_types: List[str]):
        print(f"Calculating MAP Weighted for {split_type}")
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
                map_clusters[cluster] = (statistics.mean(avg_precs), embeddings.shape[0])
            weighted_map = sum(map_stat * d_points for map_stat, d_points in map_clusters.values()) / sum(
                d_points for _, d_points in map_clusters.values()
            )
            print(map_clusters)
            print(weighted_map)
