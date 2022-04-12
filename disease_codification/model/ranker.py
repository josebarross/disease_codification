import numpy as np
import statistics
from pathlib import Path
from typing import Dict, List
from disease_codification.custom_io import create_dir_if_dont_exist, load_pickle, save_as_pickle

from disease_codification.flair_utils import read_corpus
from flair.data import Sentence, MultiCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

from disease_codification.gcp import download_blob_file, upload_blob_file
from disease_codification.process_dataset.mapper import NERAugmentation
from disease_codification.utils import chunks


class Ranker:
    def __init__(
        self,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        cluster_classifier={},
        cluster_label_binarizer={},
        cluster_tfidf: Dict[str, TfidfVectorizer] = {},
        transformer_for_embedding: str = None,
        use_incorrect_matcher_predictions: bool = False,
        subset: int = 0,
        augmentation: List[Augmentation] = [
            NERAugmentation.mention,
            NERAugmentation.sentence,
            NERAugmentation.stripped,
        ],
    ):
        self.indexers_path: Path = indexers_path
        self.models_path: Path = models_path
        self.indexer: str = indexer
        self.cluster_classifier: Dict[str, OneVsRestClassifier] = cluster_classifier
        self.cluster_label_binarizer: Dict[str, MultiLabelBinarizer] = cluster_label_binarizer
        self.cluster_tfidf = cluster_tfidf
        self.mappings: Dict[str, str] = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")
        self.clusters: List[str] = set(self.mappings.values())
        self.transformer_for_embedding = (
            TransformerDocumentEmbeddings(transformer_for_embedding, layers="-1,-2,-3,-4", fine_tune=False)
            if transformer_for_embedding
            else None
        )
        self.subset = subset
        self.use_incorrect_matcher_predictions = use_incorrect_matcher_predictions
        self.ner_augmentation = ner_augmentation
        Ranker.create_directories(models_path, indexer)

    @classmethod
    def create_directories(cls, models_path, indexer):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / "ranker")

    @classmethod
    def load(cls, indexers_path: Path, models_path: Path, indexer: str, load_from_gcp: bool = False):
        cls.create_directories(models_path, indexer)
        clusters = set(load_pickle(indexers_path / indexer / "mappings.pickle").values())
        cluster_classifier = {}
        cluster_label_binarizer = {}
        cluster_tfidf = {}
        for cluster in clusters:
            label_binarizer_path = models_path / indexer / "ranker" / f"label_binarizer_{cluster}.pickle"
            classifier_path = models_path / indexer / "ranker" / f"classifier_{cluster}.pickle"
            tfidf_path = models_path / indexer / "ranker" / f"tfidf_{cluster}.pickle"
            if load_from_gcp:
                download_blob_file(f"{indexer}/ranker/label_binarizer_{cluster}.pickle", label_binarizer_path)
                download_blob_file(f"{indexer}/ranker/classifier_{cluster}.pickle", classifier_path)
                download_blob_file(f"{indexer}/ranker/tfidf_{cluster}.pickle", tfidf_path)
            cluster_classifier[cluster] = load_pickle(classifier_path)
            cluster_label_binarizer[cluster] = load_pickle(label_binarizer_path)
            cluster_tfidf[cluster] = load_pickle(tfidf_path)
        return cls(
            indexers_path,
            models_path,
            indexer,
            cluster_classifier,
            cluster_label_binarizer,
            cluster_tfidf,
        )

    def _read_sentences(self, cluster: str, split_types: List[str]) -> List[Sentence]:
        filename = f"ranker_{cluster}"
        corpus = read_corpus(self.indexers_path / self.indexer / "ranker", filename)
        corpus_descriptions = read_corpus(self.indexers_path / self.indexer / "description", filename, only_train=True)
        corpuses = [corpus, corpus_descriptions]
        incorrect_matcher_path = self.models_path / self.indexer / "incorrect-matcher"
        if self.use_incorrect_matcher_predictions and incorrect_matcher_path.exists():
            incorrect_matcher_corpus = read_corpus(incorrect_matcher_path, f"incorrect_{cluster}", only_train=True)
            corpuses.append(incorrect_matcher_corpus)
        for aug in self.ner_augmentation:
            ner_aug = read_corpus(self.indexers_path / self.indexer / str(aug), f"ranker_{cluster}", only_train=True)
            corpuses.append(ner_aug)
        multi_corpus = MultiCorpus(corpuses)
        sentences = []
        for split_type in split_types:
            sentences += list(getattr(multi_corpus, split_type))
        if self.subset and len(sentences) < self.subset:
            sentences = sentences[: self.subset]
        return sentences

    def _set_multi_label_binarizer(self, cluster: str):
        mlb = MultiLabelBinarizer()
        labels_cluster = {f"<{label}>" for label, cluster_label in self.mappings.items() if cluster_label == cluster}
        mlb.fit([list(labels_cluster) + ["<incorrect-matcher>"]])
        self.cluster_label_binarizer[cluster] = mlb

    def _set_tfidf(self, cluster: str):
        tfidf = TfidfVectorizer()
        tfidf.fit(s.to_original_text() for s in self._read_sentences(cluster, ["train", "dev"]))
        self.cluster_tfidf[cluster] = tfidf

    def _get_labels_matrix(self, cluster: str, sentences: List[Sentence]):
        mlb = self.cluster_label_binarizer[cluster]
        labels_matrix = mlb.transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        return labels_matrix

    def _get_embeddings(self, cluster: str, sentences: List[Sentence]):
        embeddings = self.cluster_tfidf[cluster].transform(s.to_original_text() for s in sentences)
        if self.transformer_for_embedding:
            self.transformer_for_embedding.model.eval()
            for chunk in chunks(sentences):
                self.transformer_for_embedding.embed(chunk)
                for sentence in chunk:
                    sentence.embedding.to("cpu")
            transformer_embeddings = np.array([sentence.embedding.to("cpu").numpy() for sentence in sentences])
            embeddings = np.hstack((embeddings.toarray(), transformer_embeddings))
        print("Got embeddings")
        return embeddings

    def train(self, upload_to_gcp: bool = False, split_types_train: List[str] = ["train", "dev"]):
        len_clusters = len(self.clusters)
        for i, cluster in enumerate(self.clusters):
            print(f"Training for cluster {cluster} - {i}/{len_clusters}")
            self._set_multi_label_binarizer(cluster)
            self._set_tfidf(cluster)
            sentences = self._read_sentences(cluster, split_types_train)
            embeddings = self._get_embeddings(cluster, sentences)
            labels = self._get_labels_matrix(cluster, sentences)
            if not labels.any() or not sentences:
                raise Exception
            clf = OneVsRestClassifier(XGBClassifier(eval_metric="logloss")).fit(embeddings, labels)
            self.cluster_classifier[cluster] = clf
        print("Training Complete")
        self.save()
        if upload_to_gcp:
            self.upload_to_gcp()

    def predict(self, sentences: List[Sentence]):
        print("Predicting ranker")
        for cluster in self.clusters:
            print(cluster)
            embeddings = self._get_embeddings(cluster, sentences)
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
                embeddings = self._get_embeddings(cluster, sentences)
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

    def save(self):
        print("Saving model")
        base_path = self.models_path / self.indexer / "ranker"
        for cluster in self.clusters:
            save_as_pickle(self.cluster_label_binarizer[cluster], base_path / f"label_binarizer_{cluster}.pickle")
            save_as_pickle(self.cluster_classifier[cluster], base_path / f"classifier_{cluster}.pickle")
            save_as_pickle(self.cluster_tfidf[cluster], base_path / f"tfidf_{cluster}.pickle")

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
            upload_blob_file(base_path / f"tfidf_{cluster}.pickle", f"{self.indexer}/ranker/tfidf_{cluster}.pickle")
