import itertools
import multiprocessing
import statistics
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from dac_divide_and_conquer import logger
from dac_divide_and_conquer.custom_io import create_dir_if_dont_exist, load_mappings, load_pickle, save_as_pickle
from dac_divide_and_conquer.flair_utils import read_augmentation_corpora, read_corpus
from dac_divide_and_conquer.gcp import download_blob_file, upload_blob_file
from dac_divide_and_conquer.metrics import Metrics
from dac_divide_and_conquer.dataset import Augmentation
from dac_divide_and_conquer.utils import chunks, label_in_cluster
from flair.data import MultiCorpus, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier


class Ranker:
    def __init__(
        self,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        cluster_classifier={},
        cluster_label_binarizer={},
        cluster_tfidf: Dict[str, TfidfVectorizer] = {},
        seed: int = 0,
    ):
        """
        :param indexers_path: path of preprocessing files created by DACCorpus
        :param models_path: path of where to save the models
        :param indexer: corpus name given to DACCorpus
        :param seed: Seed to set for XGBoost
        """
        self.indexers_path: Path = indexers_path
        self.models_path: Path = models_path
        self.indexer: str = indexer
        self.cluster_classifier: Dict[str, OneVsRestClassifier] = cluster_classifier
        self.cluster_label_binarizer: Dict[str, MultiLabelBinarizer] = cluster_label_binarizer
        self.cluster_tfidf = cluster_tfidf
        self.mappings, self.clusters, self.multi_cluster = load_mappings(self.indexers_path, self.indexer)
        self.seed = seed
        self.dir_name = f"ranker-{seed}"
        self.transformer_for_embedding = None
        Ranker.create_directories(models_path, indexer, seed)

    @classmethod
    def create_directories(cls, models_path: Path, indexer: str, seed: int):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / f"ranker-{seed}")

    @classmethod
    def load(cls, indexers_path: Path, models_path: Path, indexer: str, load_from_gcp: bool = False, seed: int = 0):
        cls.create_directories(models_path, indexer, seed)
        _, clusters, _ = load_mappings(indexers_path, indexer)
        cluster_classifier = {}
        cluster_label_binarizer = {}
        cluster_tfidf = {}
        dir_name = f"ranker-{seed}"
        for cluster in clusters:
            label_binarizer_path = models_path / indexer / dir_name / f"label_binarizer_{cluster}.pickle"
            classifier_path = models_path / indexer / dir_name / f"classifier_{cluster}.pickle"
            tfidf_path = models_path / indexer / dir_name / f"tfidf_{cluster}.pickle"
            if load_from_gcp:
                download_blob_file(f"{indexer}/{dir_name}/label_binarizer_{cluster}.pickle", label_binarizer_path)
                download_blob_file(f"{indexer}/{dir_name}/classifier_{cluster}.pickle", classifier_path)
                download_blob_file(f"{indexer}/{dir_name}/tfidf_{cluster}.pickle", tfidf_path)
            cluster_label_binarizer[cluster] = load_pickle(label_binarizer_path)
            try:
                cluster_classifier[cluster] = load_pickle(classifier_path)
                cluster_tfidf[cluster] = load_pickle(tfidf_path)
            except FileNotFoundError:
                cluster_classifier[cluster] = None
                cluster_tfidf[cluster] = None
        return cls(
            indexers_path,
            models_path,
            indexer,
            cluster_classifier=cluster_classifier,
            cluster_label_binarizer=cluster_label_binarizer,
            cluster_tfidf=cluster_tfidf,
            seed=seed,
        )

    def _read_sentences(
        self,
        cluster: str,
        split_types: List[str],
        augmentation: List[Augmentation] = [],
        use_incorrect_matcher_predictions: bool = False,
        subset: int = 0,
    ) -> List[Sentence]:
        filename = f"ranker_{cluster}"
        corpus = read_corpus(self.indexers_path / self.indexer / "ranker", filename)
        corpora = [corpus]
        incorrect_matcher_path = self.models_path / self.indexer / "incorrect-matcher"
        if use_incorrect_matcher_predictions and incorrect_matcher_path.exists():
            incorrect_matcher_corpus = read_corpus(incorrect_matcher_path, f"incorrect_{cluster}", only_train=True)
            corpora.append(incorrect_matcher_corpus)
        corpora += read_augmentation_corpora(augmentation, self.indexers_path, self.indexer, "ranker", cluster)
        multi_corpus = MultiCorpus(corpora)
        sentences = itertools.chain.from_iterable(
            getattr(multi_corpus, split_type) for split_type in split_types if getattr(multi_corpus, split_type)
        )
        if subset and len(sentences) < subset:
            sentences = sentences[:subset]
        return list(sentences)

    def _set_multi_label_binarizer(self, cluster: str):
        mlb = MultiLabelBinarizer()
        labels_cluster = {
            f"<{label}>"
            for label in self.mappings.keys()
            if label_in_cluster(cluster, label, self.mappings, self.multi_cluster)
        }
        mlb.fit([list(labels_cluster) + ["<incorrect-matcher>"]])
        self.cluster_label_binarizer[cluster] = mlb

    def _set_tfidf(self, cluster: str, sentences: List[str]):
        logger.info("Fitting tf-idf")
        tfidf = TfidfVectorizer()
        tfidf.fit(s.to_original_text() for s in sentences)
        self.cluster_tfidf[cluster] = tfidf
        logger.info("Finished fitting tf-idf")

    def _get_labels_matrix(self, cluster: str, sentences: List[Sentence]):
        mlb = self.cluster_label_binarizer[cluster]
        labels_matrix = mlb.transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        return labels_matrix

    def _get_embeddings(self, cluster: str, sentences: List[Sentence], transformer_for_embedding: str = None):
        self.transformer_for_embedding = (
            TransformerDocumentEmbeddings(transformer_for_embedding, layers="-1,-2,-3,-4", fine_tune=False)
            if not self.transformer_for_embedding and transformer_for_embedding
            else None
        )
        embeddings = self.cluster_tfidf[cluster].transform(s.to_original_text() for s in sentences)
        if self.transformer_for_embedding:
            self.transformer_for_embedding.model.eval()
            for chunk in chunks(sentences):
                self.transformer_for_embedding.embed(chunk)
                for sentence in chunk:
                    sentence.embedding.to("cpu")
            transformer_embeddings = np.array([sentence.embedding.to("cpu").numpy() for sentence in sentences])
            embeddings = np.hstack((embeddings.toarray(), transformer_embeddings))
        return embeddings

    def train(
        self,
        upload_to_gcp: bool = False,
        split_types_train: List[str] = ["train", "dev"],
        augmentation: List[Augmentation] = [
            Augmentation.ner_mention,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.descriptions_labels,
        ],
        use_incorrect_matcher_predictions: bool = False,
        subset: int = 0,
        transformer_for_embedding: str = None,
        log_statistics_while_train: bool = True,
        train_starting_from_cluster: int = 0,
        train_until_cluster: int = None,
        n_jobs_ova: Union[float, int] = 1,
        n_jobs_xgb: Union[float, int] = -1,
        scale_pos_weight: str = "max",
        tree_method: str = "auto",
        booster: str = "dart",  # gbtree gblinear
        subsample: float = 0.6,
        colsample_bytree: float = 0.6,
    ):
        """
        :param upload_to_gcp: Wether to upload model to Google Cloud Storage after training
        :param split_types_train: Which splits to use for training
        :param augmentation: Which Augmentation techniques to use for training
        :param use_incorrect_matcher_predictions: Use incorrect matcher predictions for training
        :param subset: Subset of data to train
        :param transformer_for_embedding: Add transformer contextual embedding to tf-idf embedding, should provide transformer name to use for embedding
        :param log_statistics_while_train: Output stats of test evaluation during training
        :param train_starting_from_cluster: From which cluster index to start. Useful for pausing and resuming training.
        :param train_until_cluster: With which cluster index to end. Useful for pausing and resuming training.
        :param n_jobs_ova: How many hard forks to create for one vs rest. Usefull for paralellizing the training.
        :param n_jobs_xgb: How many threads to use in XGBoost train. Default all.
        :param scale_pos_weight: Wether to use the max or the mean of the cluster as scale_pos_weight
        :param tree_method: Tree method for XGBoost
        :param booster: Booster for XGBoost
        :param sumsample: Row subsample for XGBoost
        :param colsample_bytre: Column subsample for XGBoost
        """
        clusters_to_train = (
            self.clusters[train_starting_from_cluster:]
            if not train_until_cluster
            else self.clusters[train_starting_from_cluster:train_until_cluster]
        )
        len_clusters = len(clusters_to_train)
        logger.info(f"Clusters to train: {clusters_to_train}")
        if 0 < n_jobs_ova < 1:
            n_jobs_ova = max(1, int(multiprocessing.cpu_count() * n_jobs_ova))
        elif n_jobs_ova < 0:
            n_jobs_ova = max(1, multiprocessing.cpu_count() + n_jobs_ova)
        if 0 < n_jobs_xgb < 1:
            n_jobs_xgb = max(1, int(multiprocessing.cpu_count() * n_jobs_xgb))
        for i, cluster in enumerate(clusters_to_train):
            logger.info(f"Training for cluster {cluster} - {i}/{len_clusters}")
            self.train_cluster(
                cluster,
                upload_to_gcp=upload_to_gcp,
                split_types_train=split_types_train,
                augmentation=augmentation,
                use_incorrect_matcher_predictions=use_incorrect_matcher_predictions,
                subset=subset,
                transformer_for_embedding=transformer_for_embedding,
                log_statistics_while_train=log_statistics_while_train,
                n_jobs_ova=n_jobs_ova,
                n_jobs_xgb=n_jobs_xgb,
                scale_pos_weight=scale_pos_weight,
                tree_method=tree_method,
                booster=booster,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
            )

        logger.info("Training Complete")

    def train_cluster(
        self,
        cluster: str,
        upload_to_gcp: bool = False,
        split_types_train: List[str] = ["train", "dev"],
        augmentation: List[Augmentation] = [
            Augmentation.ner_mention,
            Augmentation.ner_sentence,
            Augmentation.ner_stripped,
            Augmentation.descriptions_labels,
        ],
        use_incorrect_matcher_predictions: bool = False,
        subset: int = 0,
        transformer_for_embedding: str = None,
        log_statistics_while_train: bool = True,
        n_jobs_ova: int = 1,
        n_jobs_xgb: int = 1,
        scale_pos_weight: str = "max",  # mean, max
        tree_method: str = "auto",  # hist exact
        booster: str = "dart",  # gbtree gblinear
        subsample: float = 0.6,
        colsample_bytree: float = 0.6,
    ):
        self._set_multi_label_binarizer(cluster)
        sentences = self._read_sentences(
            cluster, split_types_train, augmentation, use_incorrect_matcher_predictions, subset
        )

        if len(sentences) == 0:
            logger.info("Cluster has no sentences")
            self.cluster_tfidf[cluster] = None
            self.cluster_classifier[cluster] = None
            return

        self._set_tfidf(cluster, sentences)
        embeddings = self._get_embeddings(cluster, sentences, transformer_for_embedding)
        labels = self._get_labels_matrix(cluster, sentences)

        del sentences

        if scale_pos_weight == "max":
            scale_pos_weight = labels.shape[1] / np.max(labels.sum(axis=1))
        elif scale_pos_weight == "mean":
            scale_pos_weight = labels.shape[1] / np.mean(labels.sum(axis=1))

        logger.info(f"CPU to use: OVA-{n_jobs_ova}, XGBoost-{n_jobs_xgb}")
        logger.info(f"Tree method: {tree_method}")
        logger.info(f"Scale pos weight: {scale_pos_weight}")
        logger.info(f"Booster: {booster}")
        logger.info(f"Subsample: {subsample}-{colsample_bytree}")

        xgb_classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            tree_method=tree_method,
            n_jobs=n_jobs_xgb,
            scale_pos_weight=scale_pos_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            booster=booster,
            seed=self.seed,
        )
        clf = OneVsRestClassifier(xgb_classifier, n_jobs=n_jobs_ova).fit(embeddings, labels)
        self.cluster_classifier[cluster] = clf

        if log_statistics_while_train:
            logger.info(f'MAP: {self.eval_cluster(cluster, Metrics.map, "test")}')
            logger.info(f'F1: {self.eval_cluster(cluster, Metrics.summary, "test")}')

        self.save_cluster(cluster)
        if upload_to_gcp:
            self.upload_cluster_to_gcp(cluster)

    def predict(self, sentences: List[Sentence], return_probabilities=True):
        logger.info("Predicting ranker")
        for cluster in self.clusters:
            logger.info(cluster)
            classes = self.cluster_label_binarizer[cluster].classes_
            classifier = self.cluster_classifier.get(cluster)
            if not classifier:
                for sentence in sentences:
                    for label in classes:
                        sentence.add_label("ranker_proba", label, 0.0)
            elif return_probabilities:
                embeddings = self._get_embeddings(cluster, sentences)
                predictions = classifier.predict_proba(embeddings)
                for sentence, prediction in zip(sentences, predictions):
                    for i, label in enumerate(classes):
                        sentence.add_label("ranker_proba", label, prediction[i])
            else:
                embeddings = self._get_embeddings(cluster, sentences)
                predictions = self.cluster_classifier[cluster].predict(embeddings)
                classes = self.cluster_label_binarizer[cluster].classes_
                for sentence, prediction in zip(sentences, predictions):
                    for i, pred in enumerate(prediction):
                        if pred:
                            sentence.add_label("ranker", classes[i], 1.0)

    def eval_weighted(
        self, split_types: List[str] = ["test"], eval_weighted_metrics: List[Metrics] = [Metrics.map, Metrics.summary]
    ):
        scores = {}
        for metric in eval_weighted_metrics:
            logger.info(f"Calculating {metric} weighted for {split_types}")
            metric_clusters = {}
            for split_type in split_types:
                for cluster in self.clusters:
                    result, weight = self.eval_cluster(cluster, metric, split_type)
                    if result:
                        metric_clusters[cluster] = (result, weight)
                weighted_metric = sum(map_stat * d_points for map_stat, d_points in metric_clusters.values()) / sum(
                    d_points for _, d_points in metric_clusters.values()
                )
            logger.info(metric)
            logger.info(metric_clusters)
            logger.info(weighted_metric)
            scores[metric.value] = weighted_metric
        return scores

    def eval_cluster(self, cluster, metric, split_type):
        sentences = self._read_sentences(cluster, [split_type])
        if not sentences:
            return 0, 0
        classifier = self.cluster_classifier.get(cluster)
        labels_matrix = self._get_labels_matrix(cluster, sentences)
        if metric == Metrics.map:
            if not classifier:
                predictions = np.zeros_like(labels_matrix)
                logger.info(predictions.shape)
            else:
                embeddings = self._get_embeddings(cluster, sentences)
                predictions = classifier.predict_proba(embeddings)
            aps = []
            for y_true, y_scores in zip(labels_matrix, predictions):
                aps.append(average_precision_score(y_true, y_scores))
            return statistics.mean(aps), len(sentences)
        elif metric == Metrics.summary:
            if not classifier:
                predictions = np.zeros_like(labels_matrix)
            else:
                embeddings = self._get_embeddings(cluster, sentences)
                predictions = self.cluster_classifier[cluster].predict(embeddings)
            return f1_score(labels_matrix, predictions, average="micro"), len(sentences)

    def save(self):
        logger.info("Saving model")
        for cluster in self.clusters:
            self.save_cluster(cluster)

    def save_cluster(self, cluster):
        logger.info(f"Saving model created for cluster {cluster}")
        base_path = self.models_path / self.indexer / self.dir_name
        save_as_pickle(self.cluster_label_binarizer[cluster], base_path / f"label_binarizer_{cluster}.pickle")
        save_as_pickle(self.cluster_classifier[cluster], base_path / f"classifier_{cluster}.pickle")
        save_as_pickle(self.cluster_tfidf[cluster], base_path / f"tfidf_{cluster}.pickle")

    def upload_to_gcp(self):
        logger.info("Uploading ranker to GCP")
        for cluster in self.clusters:
            self.upload_cluster_to_gcp(cluster)

    def upload_cluster_to_gcp(self, cluster):
        logger.info(f"Uploading model created for cluster {cluster} to GCP")
        base_path = self.models_path / self.indexer / self.dir_name
        upload_blob_file(
            base_path / f"label_binarizer_{cluster}.pickle",
            f"{self.indexer}/{self.dir_name}/label_binarizer_{cluster}.pickle",
        )
        upload_blob_file(
            base_path / f"classifier_{cluster}.pickle", f"{self.indexer}/{self.dir_name}/classifier_{cluster}.pickle"
        )
        upload_blob_file(
            base_path / f"tfidf_{cluster}.pickle", f"{self.indexer}/{self.dir_name}/tfidf_{cluster}.pickle"
        )
