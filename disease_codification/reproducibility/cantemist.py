from pathlib import Path
from typing import List
from disease_codification.corpora_downloader import create_directories, download_cantemist_corpus
from disease_codification.evaluation import eval_ensemble, eval_mean
from disease_codification.model.indexer import Indexer
from disease_codification.model.dac import DACModel


def train_models(data_path: Path, transformers: List[str], seeds: List[int]):
    corpus = "cantemist"
    _, indexers_path, models_path = create_directories(data_path)
    for transformer, i in zip(transformers, seeds):
        model = DACModel(indexers_path, models_path, corpus, seed=i, matcher_transformer=transformer)
        model.train()


def reproduce_mean_models(data_path: Path):
    corpus = "cantemist"
    corpuses_path, indexers_path, models_path = create_directories(data_path)
    download_cantemist_corpus(corpuses_path)
    Indexer(corpus, corpuses_path, indexers_path)
    transformers = ["PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"] * 5
    seeds = range(5)
    train_models(data_path, transformers, seeds)
    eval_mean(indexers_path, models_path, corpus, transformers=transformers, seeds=seeds, first_n_digits=0)


def reproduce_ensemble_models(data_path: Path):
    corpus = "cantemist"
    corpuses_path, indexers_path, models_path = create_directories(data_path)
    download_cantemist_corpus(corpuses_path)
    Indexer(corpus, corpuses_path, indexers_path)
    transformers = (
        5 * ["PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"]
        + 5 * ["PlanTL-GOB-ES/roberta-base-biomedical-es"]
        + 5 * ["dccuchile/bert-base-spanish-wwm-cased"]
    )
    seeds = range(15)
    train_models(data_path, transformers, seeds)
    eval_ensemble(indexers_path, models_path, corpus, transformers=transformers, seeds=seeds, first_n_digits=0)
