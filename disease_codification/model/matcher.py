import statistics
from pathlib import Path
from typing import List

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist
from disease_codification.flair_utils import read_corpus, train_transformer_classifier
from disease_codification.gcp import download_blob_file, upload_blob_file
from flair.models import TextClassifier
from sklearn.metrics import average_precision_score


class Matcher:
    def __init__(self, indexers_path: Path, models_path: Path, indexers: List[str], classifier: TextClassifier = None):
        # Matcher is trained by the indexers in the order they are provided, the one to predict should be the last provided
        self.indexers_path = indexers_path
        self.indexers = indexers
        self.models_path = models_path
        self.classifier = classifier
        for indexer in indexers:
            Matcher.create_directories(models_path, indexer)

    @classmethod
    def create_directories(cls, models_path, indexer):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / "matcher")
        create_dir_if_dont_exist(models_path / indexer / "matcher-dev")

    @classmethod
    def load(cls, indexers_path, models_path, indexer, train_with_dev: bool = False, load_from_gcp: bool = True):
        cls.create_directories(models_path, indexer)
        filename = f'{indexer}/{"matcher" if not train_with_dev else "matcher-dev"}/final-model.pt'
        if load_from_gcp:
            download_blob_file(filename, models_path / filename)
        classifier = TextClassifier.load(models_path / filename)
        return cls(indexers_path, models_path, [indexer], classifier)

    def save_to_gcp(self, filename: str):
        upload_blob_file(self.models_path / filename, filename)

    def train(self, training_params):
        assert len(self.indexers) == len(training_params)
        for indexer, params in zip(self.indexers, training_params):
            print(f"Finetuning for {indexer}")
            corpus = read_corpus(self.indexers_path / indexer / "matcher", "matcher")
            train_with_dev = params.get("train_with_dev")
            filepath = f'{indexer}/{"matcher" if not train_with_dev else "matcher-dev"}'
            self.classifier = train_transformer_classifier(
                self.classifier, corpus, self.models_path / filepath, **params
            )
            self.save_to_gcp(f"{filepath}/final-model.pt")
            self.eval(corpus.dev)
            self.eval(corpus.test)

    def predict(self, sentences):
        self.classifier.predict(sentences, label_name="predicted", return_probabilities_for_all_classes=True)

    def eval(self, sentences):
        print("Starting evaluation for MAP of Matcher")
        labels = self.classifier.label_dictionary.get_items()
        index_labels = {ll: i for i, ll in enumerate(labels)}
        self.predict(sentences)
        avg_precs = []
        for sentence in sentences:
            avg_precs.append(self._get_aps_of_sentence(sentence, labels, index_labels))
        map_score = statistics.mean(avg_precs)
        print(f"\n\nMAP SCORE: {map_score}\n\n")

    def _get_aps_of_sentence(self, sentence, labels, index_labels):
        y_true = np.zeros(len(labels))
        y_scores = np.zeros(len(labels))
        for ll in sentence.get_labels("gold"):
            y_true[index_labels[ll.value]] = 1.0
        for ll in sentence.get_labels("predicted"):
            y_scores[index_labels[ll.value]] = ll.score
        return average_precision_score(y_true, y_scores)
