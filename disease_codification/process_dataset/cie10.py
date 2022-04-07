from pathlib import Path
from typing import List

import pandas as pd

clusters = {
    "a00-b99": "Ciertas enfermedades infecciosas y parasitarias",
    "c00-d49": "Tumores [neoplasias]",
    "d50-d89": "Enfermedades de la sangre y de los órganos hematopoyéticos, y ciertos trastornos que afectan el mecanismo de la inmunidad",
    "e00-e89": "Enfermedades endocrinas, nutricionales y metabolicas",
    "f00-f99": "Trastornos mentales y del comportamiento",
    "g00-g99": "Enfermedades del sistema nervioso",
    "h00-h59": "Enfermedades del ojo y sus anexos",
    "h60-h95": "Enfermedades del oído y de la apófisis mastoides",
    "i00-i99": "Enfermedades del sistema circulatorio",
    "j00-j99": "Enfermedades del sistema respiratorio",
    "k00-k95": "Enfermedades del sistema digestivo",
    "l00-l99": "Enfermedades de la piel y del tejido subcutáneo",
    "m00-m99": "Enfermedades del sistema osteomuscular y del tejido conjuntivo",
    "n00-n99": "Enfermedades del sistema genitourinario",
    "o00-o9a": "Embarazo, parto y puerperio",
    "p00-p96": "Ciertas afecciones originadas en el período perinatal",
    "q00-q99": "Malformaciones congénitas, deformidades y anomalías cromosómicas",
    "r00-r99": " Síntomas, signos y hallazgos anormales clínicos y de laboratorio, no clasificados en otra parte",
    "s00-t88": "Traumatismos, envenenamientos y algunas otras consecuencias de causas externas",
    "v00-y99": "Causas externas de morbilidad y de mortalidad",
    "z00-z99": "Factores que influyen en el estado de salud y contacto con los servicios de salud",
}


def get_cie10_df(corpuses_path: Path):
    df = pd.read_csv(corpuses_path / "cie-10.csv", sep=",")
    return df


def cluster_assigner(labels: List[str]):
    clusters_assigned = [assign_label(label) for label in labels]
    assert len(labels) == len(clusters_assigned)
    return clusters_assigned


def assign_label(label):
    category = label[:3]
    for cluster in clusters:
        if cluster.split("-")[0] <= category <= cluster.split("-")[1]:
            return cluster
