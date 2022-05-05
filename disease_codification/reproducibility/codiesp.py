from pathlib import Path
from disease_codification.corpora_downloader import create_directories, download_codiesp_corpus
from disease_codification.model.indexer import Indexer
from disease_codification.model.xova import XOVA


def reproduce_model_codiesp(data_path: Path):
    corpuses_path, indexers_path, models_path = create_directories(data_path)
    download_codiesp_corpus(corpuses_path)
    Indexer("codiesp", corpuses_path, indexers_path)
    xova = XOVA(indexers_path, models_path, "codiesp")
    xova.train()
    xova.eval(first_n_digits_summary=3)
    return xova
