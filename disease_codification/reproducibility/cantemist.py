from pathlib import Path
from disease_codification.corpora_downloader import create_directories, download_cantemist_corpus
from disease_codification.model.indexer import Indexer
from disease_codification.model.dac import DACModel


def reproduce_model(data_path: Path):
    corpuses_path, indexers_path, models_path = create_directories(data_path)
    download_cantemist_corpus(corpuses_path)
    Indexer("cantemist", corpuses_path, indexers_path)
    xova = DACModel(indexers_path, models_path, "cantemist")
    xova.train()
    xova.eval()
    return xova
