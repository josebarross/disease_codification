from disease_codification.process_dataset import codiesp, livingner, cantemist, falp, mesinesp_st1
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
    "livingner": {
        "sentence": livingner.process_sentence,
        "labels": livingner.process_labels,
        "cluster_assigner": livingner.cluster_assigner,
        Augmentation.ner_sentence: livingner.process_ner_sentences,
        Augmentation.ner_mention: livingner.process_ner_mentions,
        Augmentation.ner_stripped: livingner.process_ner_stripped,
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
}
