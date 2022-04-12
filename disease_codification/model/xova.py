from pathlib import Path
from disease_codification.custom_io import load_pickle
from disease_codification.flair_utils import get_label_value, read_corpus
from disease_codification.metrics import calculate_mean_average_precision
from disease_codification.model.matcher import Matcher

from disease_codification.model.ranker import Ranker


class XOVA:
    def __init__(
        self, indexers_path: Path, models_path: Path, indexer: str, ranker: Ranker = None, matcher: Matcher = None
    ):
        self.indexers_path = indexers_path
        self.models_path = models_path
        self.indexer = indexer
        self.ranker = ranker or Ranker(indexers_path, models_path, indexer)
        self.matcher = matcher or Matcher(indexers_path, models_path, indexer)
        self.mappings = load_pickle(self.indexers_path / self.indexer / "mappings.pickle")

    @classmethod
    def load(
        cls,
        indexers_path: Path,
        models_path: Path,
        indexer: str,
        load_ranker_from_gcp: bool = False,
        load_matcher_from_gcp: bool = False,
    ):
        ranker = Ranker.load(indexers_path, models_path, indexer, load_from_gcp=load_ranker_from_gcp)
        matcher = Matcher.load(indexers_path, models_path, indexer, load_from_gcp=load_matcher_from_gcp)
        return cls(indexers_path, models_path, indexer, ranker, matcher)

    def train(
        self,
        upload_matcher_to_gcp: bool = False,
        upload_ranker_to_gcp: bool = False,
        train_matcher=True,
        train_ranker=True,
        ranker_train_args: dict = {},
        matcher_train_args: dict = {},
    ):
        if train_matcher:
            self.matcher.train(upload_to_gcp=upload_matcher_to_gcp, **matcher_train_args)
        self.matcher.create_corpus_of_incorrectly_predicted()
        if train_ranker:
            self.ranker.train(upload_to_gcp=upload_ranker_to_gcp, **ranker_train_args)

    def upload_to_gcp(self):
        self.matcher.upload_to_gcp()
        self.ranker.upload_to_gcp()

    def save(self):
        self.matcher.save()
        self.ranker.save()

    def predict(self, sentences):
        self.matcher.predict(sentences)
        self.ranker.predict(sentences)
        self.mix_with_probabilities(sentences)

    def mix_with_probabilities(self, sentences):
        print("Joining probabilities")
        for sentence in sentences:
            matcher = sentence.get_labels("matcher")
            ranker = sentence.get_labels("ranker")
            for label in ranker:
                if label.value == "<incorrect-matcher>":
                    continue
                cluster = self.mappings[get_label_value(label)]
                score = next(l.score for l in matcher if cluster == get_label_value(l)) * label.score
                sentence.add_label("label_predicted", label.value, score)

    def eval(self):
        corpus = read_corpus(self.indexers_path / self.indexer / "corpus", "corpus")
        corpus_matcher = read_corpus(self.indexers_path / self.indexer / "matcher", "matcher")
        sentences_matcher = [s for s in corpus_matcher.test]
        self.matcher.eval(sentences_matcher)
        self.ranker.eval_weighted()
        sentences = [s for s in corpus.test]
        self.predict(sentences)
        calculate_mean_average_precision(sentences, self.mappings.keys())
