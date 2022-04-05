import shutil
from pathlib import Path

from flair.datasets import ClassificationCorpus
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from gcp import download_blob_file


def read_corpus(data_folder: Path, filename: str):
    corpus = ClassificationCorpus(
        data_folder,
        test_file=f"{filename}_test.txt",
        dev_file=f"{filename}_dev.txt",
        train_file=f"{filename}_train.txt",
        label_type="gold",
    )
    return corpus


def train_transformer_classifier(
    classifier,
    corpus,
    results_path: Path,
    max_epochs: int = 5,
    mini_batch_size: int = 10,
    remove_after_running: bool = False,
    downsample: int = 0.0,
    train_with_dev: bool = False,
    layers="-1",
    transformer_name="dccuchile/bert-base-spanish-wwm-cased",
    label_type="gold",
):
    if not results_path.exists():
        results_path.mkdir()
    if downsample:
        corpus = corpus.downsample(downsample)
    if not classifier:
        label_dict = corpus.make_label_dictionary(label_type=label_type)
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
    )
    if remove_after_running:
        shutil.rmtree(results_path)
    return classifier


def fetch_model(filename_gcp: str, filename_out: Path):
    download_blob_file("trained-models-jose", filename_gcp, filename_out)
    return TextClassifier.load(filename_out)
