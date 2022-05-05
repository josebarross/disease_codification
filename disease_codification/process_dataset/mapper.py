from functools import partial
from disease_codification.process_dataset import codiesp, cantemist, falp, mesinesp_st1, mesinesp_st2
from enum import Enum


class Augmentation(Enum):
    ner_sentence = "ner_sentence"
    ner_stripped = "ner_stripped"
    ner_mention = "ner_mention"
    descriptions_labels = "descriptions_labels"


mapper_process_function = {
    "codiesp": {
        "sentence": codiesp.process_sentence,
        "labels": codiesp.process_labels,
        "cluster_assigner": codiesp.cluster_assigner,
        Augmentation.ner_sentence: codiesp.process_ner_sentences,
        Augmentation.ner_mention: codiesp.process_ner_mentions,
        Augmentation.ner_stripped: codiesp.process_ner_stripped,
        Augmentation.descriptions_labels: codiesp.process_labels,
    },
    "codiesp_procedures": {
        "sentence": partial(codiesp.process_sentence, subtask="procedures"),
        "labels": partial(codiesp.process_labels, subtask="procedures"),
        "cluster_assigner": partial(codiesp.cluster_assigner, subtask="procedures"),
        Augmentation.ner_sentence: partial(codiesp.process_ner_sentences, subtask="procedures"),
        Augmentation.ner_mention: partial(codiesp.process_ner_mentions, subtask="procedures"),
        Augmentation.ner_stripped: partial(codiesp.process_ner_stripped, subtask="procedures"),
        Augmentation.descriptions_labels: partial(codiesp.process_labels, subtask="procedures"),
    },
    "cantemist": {
        "sentence": cantemist.process_sentence,
        "labels": cantemist.process_labels,
        "cluster_assigner": cantemist.cluster_assigner,
        Augmentation.ner_mention: cantemist.process_ner_mentions,
        Augmentation.ner_sentence: cantemist.process_ner_sentences,
        Augmentation.ner_stripped: cantemist.process_ner_stripped,
        Augmentation.descriptions_labels: cantemist.process_labels,
    },
    "falp": {
        "sentence": falp.process_sentence,
        "labels": falp.process_labels,
        "cluster_assigner": falp.cluster_assigner,
        Augmentation.ner_mention: falp.process_ner_mentions,
        Augmentation.ner_sentence: falp.process_ner_sentences,
        Augmentation.ner_stripped: falp.process_ner_stripped,
        Augmentation.descriptions_labels: falp.process_labels,
    },
    "mesinesp_st1": {
        "sentence": mesinesp_st1.process_sentence,
        "labels": mesinesp_st1.process_labels,
        "cluster_assigner": mesinesp_st1.cluster_assigner,
    },
    "mesinesp_st2": {
        "sentence": mesinesp_st2.process_sentence,
        "labels": mesinesp_st2.process_labels,
        "cluster_assigner": mesinesp_st2.cluster_assigner,
    },
}
