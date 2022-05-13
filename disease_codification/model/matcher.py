import itertools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from disease_codification import logger
from disease_codification.custom_io import create_dir_if_dont_exist, load_mappings, write_fasttext_file
from disease_codification.flair_utils import (
    CustomMultiCorpus,
    get_label_value,
    read_augmentation_corpora,
    read_corpus,
    train_transformer_classifier,
)
from disease_codification.gcp import download_blob_file, upload_blob_file
from disease_codification.metrics import Metrics, calculate_mean_average_precision, calculate_summary
from disease_codification.process_dataset.mapper import Augmentation
from flair.data import Sentence
from flair.models import TextClassifier


class Matcher:
    def __init__(
        self,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        transformer: str = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        seed: int = 0,
        classifier: TextClassifier = None,
    ):
        # Matcher is trained by the indexers in the order they are provided, the one to predict should be the last provided
        self.indexers_path = indexers_path
        self.indexer = indexer
        self.models_path = models_path
        self.transformer = transformer
        self.seed = seed
        self.name = f"{transformer}-{seed}"
        self.classifier = classifier
        self.mappings, self.clusters, self.multi_cluster = load_mappings(indexers_path, indexer)
        Matcher.create_directories(models_path, indexer, self.transformer, self.seed)

    @classmethod
    def create_directories(cls, models_path: Path, indexer: Path, transformer: str, seed: int):
        name = f"{transformer}-{seed}"
        create_dir_if_dont_exist(models_path / indexer / "matcher" / name.split("/")[0] / name.split("/")[1])
        create_dir_if_dont_exist(models_path / indexer / "incorrect-matcher")

    @classmethod
    def load(
        cls,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        transformer: str = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        seed: int = 0,
        load_from_gcp: bool = False,
    ):
        cls.create_directories(models_path, indexer, transformer, seed)
        name = f"{transformer}-{seed}"
        filename = f"{indexer}/matcher/{name}/final-model.pt"
        if load_from_gcp:
            download_blob_file(filename, models_path / filename)
        classifier = TextClassifier.load(models_path / filename)
        return cls(indexers_path, models_path, indexer, transformer=transformer, classifier=classifier, seed=seed)

    def save(self):
        self.classifier.save(self.models_path / self.indexer / "matcher" / self.name / "final-model.pt")

    def upload_to_gcp(self):
        filename = f"{self.indexer}/matcher/{self.name}/final-model.pt"
        upload_blob_file(self.models_path / filename, filename)

    def train(
        self,
        upload_to_gcp: bool = False,
        augmentation: List[Augmentation] = [
            Augmentation.ner_sentence,
            Augmentation.descriptions_labels,
        ],
        max_epochs: int = 15,
        mini_batch_size: int = 10,
        remove_after_running: bool = False,
        downsample: int = 0.0,
        train_with_dev: bool = True,
        layers="-1",
        num_workers=2,
        save_model_each_k_epochs: int = 0,
    ):
        logger.info(f"Finetuning for {self.indexer}")
        corpus = read_corpus(self.indexers_path / self.indexer / "matcher", "matcher")
        corpora = [corpus] + read_augmentation_corpora(
            augmentation, self.indexers_path, self.indexer, "matcher", "matcher"
        )
        multi_corpus = CustomMultiCorpus(corpora)
        filepath = create_dir_if_dont_exist(
            self.models_path / self.indexer / "matcher" / self.name.split("/")[0] / self.name.split("/")[1]
        )
        self.classifier = train_transformer_classifier(
            self.classifier,
            multi_corpus,
            filepath,
            max_epochs=max_epochs,
            mini_batch_size=mini_batch_size,
            remove_after_running=remove_after_running,
            downsample=downsample,
            train_with_dev=train_with_dev,
            layers=layers,
            transformer_name=self.transformer,
            num_workers=num_workers,
            save_model_each_k_epochs=save_model_each_k_epochs,
        )
        if upload_to_gcp:
            self.upload_to_gcp()
        self.eval([s for s in corpus.dev])
        self.eval([s for s in corpus.test])

    def predict(self, sentences: List[Sentence], return_probabilities=True):
        logger.info("Predicting matcher")
        label_name = "matcher_proba" if return_probabilities else "matcher"
        self.classifier.predict(
            sentences, label_name=label_name, return_probabilities_for_all_classes=return_probabilities
        )

    def eval(self, sentences, eval_metrics: List[Metrics] = [Metrics.map, Metrics.summary]):
        if not sentences:
            return
        logger.info("Evaluation of Matcher")
        for metric in eval_metrics:
            labels_list = (
                itertools.chain.from_iterable(self.mappings.values()) if self.multi_cluster else self.mappings.values()
            )
            if metric == Metrics.map:
                self.predict(sentences, return_probabilities=True)
                calculate_mean_average_precision(sentences, labels_list, label_name_predicted="matcher_proba")
            elif metric == Metrics.summary:
                self.predict(sentences, return_probabilities=False)
                calculate_summary(sentences, labels_list, label_name_predicted="matcher")

    def create_corpus_of_incorrectly_predicted(self):
        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        sentences = [sentence for sentence in corpus.dev]
        self.predict(sentences, return_probabilities=False)
        sentences_incorrect = defaultdict(list)
        for sentence in sentences:
            for label in sentence.get_labels("predicted"):
                if get_label_value(label) == "unk":
                    continue
                cluster_predicted = get_label_value(label)
                clusters_gold = [self.mappings[get_label_value(ll)] for ll in sentence.get_labels("gold")]
                if self.multi_cluster:
                    clusters_gold = itertools.chain.from_iterable(clusters_gold)
                if cluster_predicted not in clusters_gold:
                    sentences_incorrect[cluster_predicted].append(sentence.to_original_text())
        for cluster in sentences_incorrect.keys():
            write_fasttext_file(
                sentences_incorrect[cluster],
                [["incorrect-matcher"]] * len(sentences_incorrect[cluster]),
                self.models_path / self.indexer / "incorrect-matcher" / f"incorrect_{cluster}_train.txt",
            )
