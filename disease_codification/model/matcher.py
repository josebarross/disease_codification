from collections import defaultdict
from pathlib import Path

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, load_pickle, write_fasttext_file
from disease_codification.flair_utils import get_label_value, read_corpus, train_transformer_classifier
from disease_codification.gcp import download_blob_file, upload_blob_file
from flair.models import TextClassifier
from flair.data import MultiCorpus
from sklearn.metrics import average_precision_score

from disease_codification.metrics import calculate_mean_average_precision


class Matcher:
    def __init__(self, indexers_path: Path, models_path: Path, indexer: str, classifier: TextClassifier = None):
        # Matcher is trained by the indexers in the order they are provided, the one to predict should be the last provided
        self.indexers_path = indexers_path
        self.indexer = indexer
        self.models_path = models_path
        self.classifier = classifier
        self.mappings = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")
        Matcher.create_directories(models_path, indexer)

    @classmethod
    def create_directories(cls, models_path, indexer):
        create_dir_if_dont_exist(models_path / indexer)
        create_dir_if_dont_exist(models_path / indexer / "matcher")
        create_dir_if_dont_exist(models_path / indexer / "incorrect-matcher")

    @classmethod
    def load(cls, indexers_path, models_path, indexer, load_from_gcp: bool = True):
        cls.create_directories(models_path, indexer)
        filename = f"{indexer}/matcher/final-model.pt"
        if load_from_gcp:
            download_blob_file(filename, models_path / filename)
        classifier = TextClassifier.load(models_path / filename)
        return cls(indexers_path, models_path, indexer, classifier)

    def save(self):
        self.classifier.save(self.models_path / self.indexer / "matcher" / "final-model.pt")

    def upload_to_gcp(self):
        filename = f"{self.indexer}/matcher/final-model.pt"
        upload_blob_file(self.models_path / filename, filename)

    def train(self, training_params: dict = {}, upload_to_gcp: bool = False):
        print(f"Finetuning for {self.indexer}")
        corpus = read_corpus(self.indexers_path / self.indexer / "matcher", "matcher")
        corpus_descriptions = read_corpus(self.indexers_path / self.indexer / "description", "matcher", only_train=True)
        multi_corpus = MultiCorpus([corpus, corpus_descriptions])
        filepath = f"{self.indexer}/matcher"
        self.classifier = train_transformer_classifier(
            self.classifier, multi_corpus, self.models_path / filepath, **training_params
        )
        if upload_to_gcp:
            self.upload_to_gcp()
        self.eval([s for s in corpus.dev])
        self.eval([s for s in corpus.test])

    def predict(self, sentences, return_probabilities=True):
        print("Predicting matcher")
        self.classifier.predict(
            sentences, label_name="matcher", return_probabilities_for_all_classes=return_probabilities
        )

    def eval(self, sentences):
        print("Evaluation of Matcher")
        self.predict(sentences)
        calculate_mean_average_precision(sentences, self.mappings.values(), label_name_predicted="matcher")

    def _get_aps_of_sentence(self, sentence, labels, index_labels):
        y_true = np.zeros(len(labels))
        y_scores = np.zeros(len(labels))
        for ll in sentence.get_labels("gold"):
            y_true[index_labels[ll.value]] = 1.0
        for ll in sentence.get_labels("matcher"):
            y_scores[index_labels[ll.value]] = ll.score
        return average_precision_score(y_true, y_scores)

    def create_corpus_of_incorrectly_predicted(self):
        mappings = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")
        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        sentences = [sentence for sentence in corpus.dev]
        self.predict(sentences, return_probabilities=False)
        sentences_incorrect = defaultdict(list)
        for sentence in sentences:
            for label in sentence.get_labels("predicted"):
                if get_label_value(label) == "unk":
                    continue
                cluster_predicted = get_label_value(label)
                clusters_gold = {mappings[get_label_value(ll)] for ll in sentence.get_labels("gold")}
                if cluster_predicted not in clusters_gold:
                    sentences_incorrect[cluster_predicted].append(sentence.to_original_text())
        for cluster in sentences_incorrect.keys():
            write_fasttext_file(
                sentences_incorrect[cluster],
                [["incorrect-matcher"]] * len(sentences_incorrect[cluster]),
                self.models_path / self.indexer / "incorrect-matcher" / f"incorrect_{cluster}_train.txt",
            )
