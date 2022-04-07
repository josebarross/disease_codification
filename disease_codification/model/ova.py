from pathlib import Path
from typing import List

from disease_codification.custom_io import create_dir_if_dont_exist, load_pickle, save_as_pickle
from disease_codification.flair_utils import read_corpus
from disease_codification.gcp import download_blob_file, upload_blob_file
from flair.data import Sentence, MultiCorpus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier

from disease_codification.metrics import calculate_mean_average_precision


class OVA:
    def __init__(self, indexers_path: Path, models_path: Path, indexer: str, classifier=None, label_binarizer=None):
        self.indexers_path = indexers_path
        self.models_path = models_path
        self.indexer = indexer
        self.classifier = classifier
        self.label_binarizer = label_binarizer
        self.mappings = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")
        OVA.create_directories(models_path, indexer)

    @classmethod
    def create_directories(cls, models_path, indexer):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / "ova")

    @classmethod
    def load(cls, indexers_path: Path, models_path: Path, indexer: str, load_from_gcp: bool = False):
        cls.create_directories(models_path, indexer)
        label_binarizer_path = models_path / indexer / "ova" / f"label_binarizer.pickle"
        classifier_path = models_path / indexer / "ova" / f"classifier.pickle"
        if load_from_gcp:
            download_blob_file(f"{indexer}/ova/label_binarizer.pickle", label_binarizer_path)
            download_blob_file(f"{indexer}/ova/classifier.pickle", classifier_path)
        classifier = load_pickle(classifier_path)
        label_binarizer = load_pickle(label_binarizer_path)
        return cls(indexers_path, models_path, indexer, classifier, label_binarizer)

    def save(self):
        print("Saving model")
        base_path = self.models_path / self.indexer / "ova"
        save_as_pickle(self.label_binarizer, base_path / f"label_binarizer.pickle")
        save_as_pickle(self.classifier, base_path / f"classifier.pickle")

    def upload_to_gcp(self):
        base_path = self.models_path / self.indexer / "ova"
        upload_blob_file(base_path / f"label_binarizer.pickle", f"{self.indexer}/ova/label_binarizer.pickle")
        upload_blob_file(base_path / f"classifier.pickle", f"{self.indexer}/ova/classifier.pickle")

    def _get_embeddings(self, sentences: List[Sentence]):
        return [s.to_original_text() for s in sentences]

    def _set_multi_label_binarizer(self):
        mlb = MultiLabelBinarizer()
        labels = {f"<{label}>" for label in self.mappings.keys()}
        mlb.fit([list(labels)])
        self.label_binarizer = mlb

    def _get_labels_matrix(self, sentences):
        mlb = self.label_binarizer
        labels_matrix = mlb.transform([[label.value for label in s.get_labels("gold")] for s in sentences])
        return labels_matrix

    def train(self, upload_to_gcp: bool = False, split_types_train=["train", "dev"]):
        print(f"Training OVA")
        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        corpus_descriptions = read_corpus(self.indexers_path / self.indexer / "description", "corpus", only_train=True)
        multi_corpus = MultiCorpus([corpus, corpus_descriptions])
        sentences = []
        for split_type in split_types_train:
            sentences += list(getattr(multi_corpus, split_type))
        embeddings = self._get_embeddings(sentences)
        self._set_multi_label_binarizer()
        labels = self._get_labels_matrix(sentences)
        if not labels.any() or not sentences:
            raise Exception
        pipe = Pipeline([("tfidf", TfidfVectorizer()), ("xgboost", XGBClassifier(eval_metric="logloss"))])
        clf = OneVsRestClassifier(pipe).fit(embeddings, labels)
        self.classifier = clf
        print("Training Complete")
        self.save()
        if upload_to_gcp:
            self.upload_to_gcp()

    def predict(self, sentences):
        print("Predicting for OVA")
        embeddings = self._get_embeddings(sentences)
        predictions = self.classifier.predict_proba(embeddings)
        classes = self.label_binarizer.classes_
        for sentence, prediction in zip(sentences, predictions):
            for i, label in enumerate(classes):
                sentence.add_label("ova", label, prediction[i])

    def eval(self):
        print("Eval OVA")
        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        sentences = [s for s in corpus.test]
        self.predict(sentences)
        calculate_mean_average_precision(sentences, self.mappings.keys(), label_name_predicted="ova")
