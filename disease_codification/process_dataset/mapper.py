from disease_codification.process_dataset.codiesp import (
    preprocess_ner_mentions,
    preprocess_ner_sentences,
    preprocess_ner_stripped,
    process_codiesp_labels,
    process_codiesp_sentence,
)
from enum import Enum


class Augmentation(Enum):
    ner_sentence = "ner_sentence"
    ner_stripped = "ner_stripped"
    ner_mention = "ner_mention"
    descriptions_codiesp_cie = "descriptions_codiesp_cie"


mapper_process_function = {
    "codiesp": {
        "sentence": process_codiesp_sentence,
        "labels": process_codiesp_labels,
        Augmentation.ner_sentence: preprocess_ner_sentences,
        Augmentation.ner_mention: preprocess_ner_mentions,
        Augmentation.ner_stripped: preprocess_ner_stripped,
        Augmentation.descriptions_codiesp_cie: process_codiesp_labels,
    }
}
