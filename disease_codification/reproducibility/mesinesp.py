from pathlib import Path
from disease_codification.corpora_downloader import download_mesinesp_corpus
from disease_codification.gcp import upload_blob
from disease_codification.model.indexer import Indexer
from disease_codification.model.xova import XOVA
from disease_codification import logger


def reproduce_model(corpuses_path: Path, indexers_path: Path, models_path: Path):
    download_mesinesp_corpus(corpuses_path)
    corpus = "mesinesp_st1"
    Indexer(corpus, corpuses_path, indexers_path, clustering_type="multi_cluster")
    xova = XOVA(indexers_path, models_path, corpus)
    matcher_train_args = {"max_epochs": 30, "train_with_dev": False}
    xova.train(matcher_train_args=matcher_train_args)
    xova.eval()
