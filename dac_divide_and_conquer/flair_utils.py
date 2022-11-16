import json
import os
import shutil
from pathlib import Path
from typing import List

import torch
from flair.data import ConcatDataset, Corpus, Sentence
from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.optim import LinearSchedulerWithWarmup
from flair.trainers import ModelTrainer

from dac_divide_and_conquer import logger
from dac_divide_and_conquer.custom_io import create_dir_if_dont_exist
from dac_divide_and_conquer.dataset import Augmentation
from dac_divide_and_conquer.gcp import download_blob_file


def read_corpus(data_folder: Path, filename: str, only_train: bool = False):
    try:
        if only_train:
            return ClassificationCorpus(
                data_folder, train_file=f"{filename}_train.txt", label_type="gold", sample_missing_splits=False
            )
        return ClassificationCorpus(
            data_folder,
            test_file=f"{filename}_test.txt",
            dev_file=f"{filename}_dev.txt",
            train_file=f"{filename}_train.txt",
            label_type="gold",
        )
    except RuntimeError as e:
        logger.info(e)
        logger.info(f"Error loading data from {data_folder}/{filename}. only_train={only_train}")
        return None


def read_augmentation_corpora(
    augmentation: List[Augmentation], indexers_path: Path, indexer: str, caller: str, filename: str
):
    corpora = []
    for aug in augmentation:
        path_aug = indexers_path / indexer / f"{caller}-{aug}"
        if not os.listdir(path_aug):
            logger.info(f"Augmentation {aug} not implemented")
            continue
        aug = read_corpus(path_aug, filename, only_train=True)
        if aug:
            corpora.append(aug)
    return corpora


def train_transformer_classifier(
    classifier,
    corpus,
    results_path: Path,
    max_epochs: int = 15,
    mini_batch_size: int = 10,
    remove_after_running: bool = False,
    downsample: int = 0.0,
    train_with_dev: bool = True,
    layers="-1",
    transformer_name="PlanTL-GOB-ES/roberta-base-biomedical-es",
    num_workers=2,
    save_model_each_k_epochs: int = 0,
    optimizer: torch.optim = torch.optim.AdamW,
    scheduler=LinearSchedulerWithWarmup,
    **kwargs,
):
    create_dir_if_dont_exist(results_path)
    if downsample:
        corpus = corpus.downsample(downsample)
    if not classifier:
        label_dict = corpus.make_label_dictionary(label_type="gold")
        transformer_embeddings = TransformerDocumentEmbeddings(transformer_name, fine_tune=True, layers=layers)

        classifier = TextClassifier(
            transformer_embeddings, label_dictionary=label_dict, label_type="gold", multi_label=True
        )

    trainer = ModelTrainer(classifier, corpus)
    trainer.fine_tune(
        results_path,
        learning_rate=5.0e-5,
        mini_batch_size=mini_batch_size,
        max_epochs=max_epochs,
        train_with_dev=train_with_dev,
        num_workers=num_workers,
        save_model_each_k_epochs=save_model_each_k_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs,
    )
    if remove_after_running:
        shutil.rmtree(results_path)
    return classifier


def fetch_model(filename_gcp: str, filename_out: Path):
    download_blob_file(filename_gcp, filename_out)
    return TextClassifier.load(filename_out)


def get_label_value(label):
    return label.value.replace("<", "").replace(">", "")


class CustomMultiCorpus(Corpus):
    def __init__(self, corpora: List[Corpus], name: str = "multicorpus", **corpusargs):
        self.corpora: List[Corpus] = corpora

        train_parts = []
        dev_parts = []
        test_parts = []
        for corpus in self.corpora:
            if corpus.train:
                train_parts.append(corpus.train)
            if corpus.dev:
                dev_parts.append(corpus.dev)
            if corpus.test:
                test_parts.append(corpus.test)

        super(CustomMultiCorpus, self).__init__(
            ConcatDataset(train_parts) if len(train_parts) > 0 else None,
            ConcatDataset(dev_parts) if len(dev_parts) > 0 else None,
            ConcatDataset(test_parts) if len(test_parts) > 0 else None,
            name=name,
            **corpusargs,
        )

    def __str__(self):
        output = (
            f"MultiCorpus: "
            f"{len(self.train) if self.train else 0} train + "
            f"{len(self.dev) if self.dev else 0} dev + "
            f"{len(self.test) if self.test else 0} test sentences\n - "
        )
        output += "\n - ".join([f"{type(corpus).__name__} {str(corpus)} - {corpus.name}" for corpus in self.corpora])
        return output


def save_predictions_to_file(
    path: Path,
    filename: str,
    sentences: List[Sentence],
    label_name: str,
    return_probabilities: bool,
    filenames: List[str],
):
    path_predictions = path / filename
    create_dir_if_dont_exist(path_predictions.parent)
    if path_predictions.exists():
        with open(path_predictions) as f:
            predictions = json.load(f)
    else:
        predictions = {}
    for filename, sentence in zip(filenames, sentences):
        if filename not in predictions:
            predictions[filename] = {"gold": [], "predicted_probabilities": {}, "predicted": [], "text": ""}
        predictions[filename]["gold"] = [get_label_value(l) for l in sentence.get_labels("gold")]
        predictions[filename]["text"] = sentence.to_original_text()
        if return_probabilities:
            predictions[filename]["predicted_probabilities"] = {
                get_label_value(l): l.score for l in sentence.get_labels(label_name)
            }
        else:
            predictions[filename]["predicted"] = [get_label_value(l) for l in sentence.get_labels(label_name)]
    with open(path_predictions, "w") as f:
        json.dump(predictions, f)
