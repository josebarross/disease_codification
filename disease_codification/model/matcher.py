from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from disease_codification.custom_io import create_dir_if_dont_exist, load_pickle, write_fasttext_file
from disease_codification.flair_utils import (
    CustomMultiCorpus,
    get_label_value,
    read_corpus,
    train_transformer_classifier,
)
from disease_codification.gcp import download_blob_file, upload_blob_file
from flair.models import TextClassifier
from sklearn.metrics import average_precision_score

from disease_codification.metrics import calculate_mean_average_precision
from disease_codification.process_dataset.mapper import Augmentation


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

    def train(
        self,
        upload_to_gcp: bool = False,
        augmentation: List[Augmentation] = [
            Augmentation.ner_sentence,
            Augmentation.descriptions_codiesp_cie,
        ],
        max_epochs: int = 15,
        mini_batch_size: int = 10,
        remove_after_running: bool = False,
        downsample: int = 0.0,
        train_with_dev: bool = True,
        layers="-1",
        transformer_name="PlanTL-GOB-ES/roberta-base-biomedical-es",
    ):
        print(f"Finetuning for {self.indexer}")
        corpus = read_corpus(self.indexers_path / self.indexer / "matcher", "matcher")
        corpora = [corpus]
        for aug in augmentation:
            aug = read_corpus(self.indexers_path / self.indexer / f"matcher-{aug}", "matcher", only_train=True)
            corpora.append(aug)
        multi_corpus = CustomMultiCorpus(corpora)
        filepath = f"{self.indexer}/matcher"
        self.classifier = train_transformer_classifier(
            self.classifier,
            multi_corpus,
            self.models_path / filepath,
            max_epochs=max_epochs,
            mini_batch_size=mini_batch_size,
            remove_after_running=remove_after_running,
            downsample=downsample,
            train_with_dev=train_with_dev,
            layers=layers,
            transformer_name=transformer_name,
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
