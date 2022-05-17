from collections import defaultdict
import json
import statistics
from typing import List
from disease_codification.flair_utils import get_label_value, read_corpus
from disease_codification.metrics import Metrics, calculate_mean_average_precision, calculate_summary

from disease_codification.model.dac import DACModel
from disease_codification import logger


def eval_mean(
    indexers_path,
    models_path,
    indexer,
    transformers: List[str],
    seeds: List[int],
    load_matcher_from_gcp: bool = False,
    load_ranker_from_gcp: bool = False,
    metrics: List[Metrics] = [Metrics.map, Metrics.summary],
    first_n_digits: int = 0,
):
    assert len(transformers) == len(seeds)
    maps = []
    f1_scores = []
    corpus = read_corpus(indexers_path / indexer / "corpus", "corpus")
    sentences = [s for s in corpus.test]
    for transformer, seed in zip(transformers, seeds):
        dac = DACModel.load(
            indexers_path,
            models_path,
            indexer,
            matcher_transformer=transformer,
            seed=seed,
            load_ranker_from_gcp=load_ranker_from_gcp,
            load_matcher_from_gcp=load_matcher_from_gcp,
        )
        logger.info(f"Predicting {dac.name}")
        for metric in metrics:
            return_probabilities = metric == Metrics.map
            label_name = "label_predicted_proba" if return_probabilities else "label_predicted"
            dac.predict(sentences, return_probabilities=return_probabilities)
            if metric == Metrics.map:
                map_s = calculate_mean_average_precision(sentences, dac.mappings.keys(), label_name)
                logger.info(f"MAP {dac.name}: {map_s}")
                maps.append(map_s)
            elif metric == Metrics.summary:
                f1 = calculate_summary(sentences, dac.mappings.keys(), first_n_digits=first_n_digits, output_full=False)
                logger.info(f"F1 {dac.name}: {f1}")
                f1_scores.append(f1)

    logger.info(f"MAPS: {maps}, mean: {statistics.mean(maps)}, sd: {statistics.stdev(maps)}")
    logger.info(f"F1s: {f1_scores}, mean: {statistics.mean(f1_scores)}, sd: {statistics.stdev(f1_scores)}")


def eval_ensemble(
    indexers_path,
    models_path,
    indexer,
    transformers: List[str],
    seeds: List[int],
    ensemble_type: str = "sum",  # max sum
    load_from_gcp: bool = False,
    metrics: List[Metrics] = [Metrics.map, Metrics.summary],
    first_n_digits: int = 0,
):
    assert len(transformers) == len(seeds)
    results_evaluation = {}
    for metric in metrics:
        return_probabilities = metric == Metrics.map
        corpus = read_corpus(indexers_path / indexer / "corpus", "corpus")
        sentences = [s for s in corpus.test]
        for transformer, seed in zip(transformers, seeds):
            dac = DACModel.load(
                indexers_path,
                models_path,
                indexer,
                matcher_transformer=transformer,
                seed=seed,
                load_ranker_from_gcp=load_from_gcp,
                load_matcher_from_gcp=load_from_gcp,
            )
            logger.info(f"Predicting {dac.name}")
            dac.predict(sentences, label_name=dac.name, return_probabilities=return_probabilities)

        label_name = "label_predicted_proba" if return_probabilities else "label_predicted"
        for sentence in sentences:
            labels = defaultdict(list)
            for transformer, seed in zip(transformers, seeds):
                name = f"{transformer}-{seed}"
                for label in sentence.get_labels(name):
                    labels[get_label_value(label)].append(label.score)
            for label, scores in labels.items():
                if metric == Metrics.map:
                    score = max(scores) if ensemble_type == "max" else sum(scores)
                    sentence.add_label(label_name, label, score)
                elif metric == Metrics.summary:
                    sentence.add_label(label_name, label, 1.0)

        final_score = (
            calculate_mean_average_precision(sentences, dac.mappings.keys(), label_name)
            if metric == Metrics.map
            else calculate_summary(
                sentences, dac.mappings.keys(), label_name, first_n_digits=first_n_digits, output_full=False
            )
        )
        results_evaluation[str(metric)] = final_score

    logger.info(json.dumps(results_evaluation, indent=2))
