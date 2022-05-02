from pathlib import Path
from disease_codification.corpora_downloader import download_mesinesp_corpus
from disease_codification.gcp import upload_blob
from disease_codification.model.indexer import Indexer
from disease_codification.model.xova import XOVA


def train_mesinesp_on_supercomputer(corpuses_path: Path, indexers_path: Path, models_path: Path):
    print("Testing uploading to gcp")
    upload_blob("Hola", "testing.pickle")
    download_mesinesp_corpus(corpuses_path)
    corpus = "mesinesp_st1"
    indexer = Indexer(corpus, corpuses_path, indexers_path, multi_cluster=True)
    indexer.create_corpuses()
    xova = XOVA(indexers_path, models_path, corpus)
    xova.train(upload_matcher_to_gcp=True, upload_ranker_to_gcp=True)
    xova.eval()
