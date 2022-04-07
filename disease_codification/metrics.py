import statistics
from typing import List
import numpy as np
from sklearn.metrics import average_precision_score

from disease_codification.flair_utils import get_label_value


def calculate_mean_average_precision(sentences, labels_list: List[str], label_name_predicted="label_predicted"):
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
