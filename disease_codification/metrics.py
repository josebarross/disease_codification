from enum import Enum
import itertools
import statistics
from typing import List
import numpy as np
from sklearn.metrics import average_precision_score, classification_report

from disease_codification.flair_utils import get_label_value


class Metrics(Enum):
    map = "map"
    summary = "summary"  # f1, precision, recall, accuracy


def calculate_mean_average_precision(sentences, labels_list: List[str], label_name_predicted="label_predicted_proba"):
    print(f"Calculating map for {label_name_predicted}")
    label_indexes = {label: i for i, label in enumerate(set(labels_list))}
    avg_precs = []
    for sentence in sentences:
        gold = np.zeros(len(label_indexes))
        pred = np.zeros(len(label_indexes))
        for label in sentence.get_labels("gold"):
            if label.value != "<unk>":
                gold[label_indexes[get_label_value(label)]] = 1
        for label in sentence.get_labels(label_name_predicted):
            if label.value != "<unk>":
                pred[label_indexes[get_label_value(label)]] = label.score
        if gold.any():
            avg_precs.append(average_precision_score(gold, pred))
    map_s = statistics.mean(avg_precs)
    print(map_s)
    return map_s


def calculate_summary(
    sentences, labels_list: List[str], label_name_predicted="label_predicted", first_n_digits: int = 0
):
    print(f"Calculating summary statistics for {label_name_predicted}")
    labels_list = set(labels_list)
    if first_n_digits:
        labels_list = set(l[:first_n_digits] for l in labels_list)

    label_indexes = {label: i for i, label in enumerate(labels_list)}
    gold = np.zeros((len(sentences), len(label_indexes)))
    pred = np.zeros((len(sentences), len(label_indexes)))
    for i, sentence in enumerate(sentences):
        for label in sentence.get_labels("gold"):
            label_value = get_label_value(label) if not first_n_digits else get_label_value(label)[:first_n_digits]
            if label.value != "<unk>":
                gold[i, label_indexes[label_value]] = 1
        for label in sentence.get_labels(label_name_predicted):
            label_value = get_label_value(label) if not first_n_digits else get_label_value(label)[:first_n_digits]
            if label.value != "<unk>":
                pred[i, label_indexes[label_value]] = 1
    report = classification_report(
        gold, pred, digits=4, zero_division=0, target_names=[label for label in label_indexes.keys()]
    )
    print(report)
