from pathlib import Path
from typing import List

from disease_codification.custom_io import load_mappings
from disease_codification.flair_utils import get_label_value, read_corpus
from disease_codification.metrics import Metrics, calculate_mean_average_precision, calculate_summary
from disease_codification.model.matcher import Matcher

from disease_codification.model.ranker import Ranker
from flair.data import Sentence
from disease_codification import logger


class DACModel:
    def __init__(
        self,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        matcher_transformer: str = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        seed: int = 0,
        ranker: Ranker = None,
        matcher: Matcher = None,
    ):
        self.indexers_path = indexers_path
        self.models_path = models_path
        self.indexer = indexer
        self.ranker = ranker or Ranker(indexers_path, models_path, indexer)
        self.matcher = matcher or Matcher(
            indexers_path, models_path, indexer, transformer=matcher_transformer, seed=seed
        )
        self.mappings, self.clusters, self.multi_cluster = load_mappings(indexers_path, indexer)

    @classmethod
    def load(
        cls,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        matcher_transformer: str = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        seed: int = 0,
        load_ranker_from_gcp: bool = False,
        load_matcher_from_gcp: bool = False,
    ):
        ranker = Ranker.load(indexers_path, models_path, indexer, load_from_gcp=load_ranker_from_gcp)
        matcher = Matcher.load(
            indexers_path,
            models_path,
            indexer,
            transformer=matcher_transformer,
            seed=seed,
            load_from_gcp=load_matcher_from_gcp,
        )
        return cls(
            indexers_path,
            models_path,
            indexer,
            ranker=ranker,
            matcher=matcher,
            matcher_transformer=matcher_transformer,
            seed=seed,
        )

    def train(
        self,
        upload_matcher_to_gcp: bool = False,
        upload_ranker_to_gcp: bool = False,
        train_matcher: bool = True,
        train_ranker: bool = True,
        ranker_train_args: dict = {},
        matcher_train_args: dict = {},
    ):
        if train_matcher:
            self.matcher.train(upload_to_gcp=upload_matcher_to_gcp, **matcher_train_args)
        if ranker_train_args.get("use_incorrect_matcher_predictions"):
            self.matcher.create_corpus_of_incorrectly_predicted()
        if train_ranker:
            self.ranker.train(upload_to_gcp=upload_ranker_to_gcp, **ranker_train_args)

    def upload_to_gcp(self):
        self.matcher.upload_to_gcp()
        self.ranker.upload_to_gcp()

    def save(self):
        self.matcher.save()
        self.ranker.save()

    def predict(self, sentences: List[Sentence], return_probabilities: bool = True):
        self.matcher.predict(sentences, return_probabilities=return_probabilities)
        self.ranker.predict(sentences, return_probabilities=return_probabilities)
        if return_probabilities:
            self.mix_with_probabilities(sentences)
        else:
            self.predict_only_matched_clusters(sentences)

    def mix_with_probabilities(self, sentences: List[Sentence]):
        logger.info("Joining probabilities")
        for sentence in sentences:
            matcher = sentence.get_labels("matcher_proba")
            ranker = sentence.get_labels("ranker_proba")
            for label in ranker:
                if label.value == "<incorrect-matcher>":
                    continue
                if self.multi_cluster:
                    max_score = 0
                    for cluster in self.mappings[get_label_value(label)]:
                        try:
                            score = next(l.score for l in matcher if cluster == get_label_value(l)) * label.score
                        except StopIteration:
                            score = 0.0
                        if score > max_score:
                            max_score = score
                    score = max_score
                else:
                    cluster = self.mappings[get_label_value(label)]
                    try:
                        score = next(l.score for l in matcher if cluster == get_label_value(l)) * label.score
                    except StopIteration:
                        score = 0.0
                sentence.add_label("label_predicted_proba", label.value, score)

    def predict_only_matched_clusters(self, sentences: List[Sentence]):
        for sentence in sentences:
            matcher_predictions = [get_label_value(label_matcher) for label_matcher in sentence.get_labels("matcher")]
            ranker_predictions = sentence.get_labels("ranker")
            for label_ranker in ranker_predictions:
                if label_ranker.value == "<incorrect-matcher>":
                    continue
                if self.multi_cluster and any(
                    c in matcher_predictions for c in self.mappings[get_label_value(label_ranker)]
                ):
                    sentence.add_label("label_predicted", label_ranker.value, 1.0)
                elif self.mappings[get_label_value(label_ranker)] in matcher_predictions:
                    sentence.add_label("label_predicted", label_ranker.value, 1.0)

    def eval(
        self,
        eval_metrics: List[Metrics] = [Metrics.summary, Metrics.map],
        first_n_digits_summary: int = 0,
        eval_ranker: bool = True,
        eval_matcher: bool = True,
    ):
        if eval_matcher:
            corpus_matcher = read_corpus(self.indexers_path / self.indexer / "matcher", "matcher")
            sentences_matcher = [s for s in corpus_matcher.test]
            self.matcher.eval(sentences_matcher, eval_metrics=eval_metrics)
        if eval_ranker:
            self.ranker.eval_weighted(eval_weighted_metrics=eval_metrics)

        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        sentences = [s for s in corpus.test]
        for metric in eval_metrics:
            if metric == Metrics.map:
                self.predict(sentences)
                calculate_mean_average_precision(
                    sentences, self.mappings.keys(), label_name_predicted="label_predicted_proba"
                )
            elif metric == Metrics.summary:
                self.predict(sentences, return_probabilities=False)
                calculate_summary(
                    sentences,
                    self.mappings.keys(),
                    label_name_predicted="label_predicted",
                    first_n_digits=first_n_digits_summary,
                    output_full=False,
                )
                calculate_mean_average_precision(
                    sentences, self.mappings.keys(), label_name_predicted="label_predicted"
                )
