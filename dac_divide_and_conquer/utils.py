from collections import defaultdict
import itertools
import json
from pathlib import Path
from statistics import mean


def chunks(lst, n: int = 10):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def label_in_cluster(cluster: str, label: str, mappings: dict, multi_cluster: bool):
    if multi_cluster:
        return cluster in mappings[label]
    else:
        return cluster == mappings[label]


def create_latex_table_with_component_analysis(component_analysis_filepath: Path):
    with open(component_analysis_filepath) as f:
        component_analysis = json.load(f)
    name_transformers = {
        "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es": "BioClinical RoBERTa",
        "PlanTL-GOB-ES/roberta-base-biomedical-es": "BioMedical RoBERTa",
        "dccuchile/bert-base-spanish-wwm-cased": "BETO",
    }
    tabular_str = ""
    component_analysis = {**component_analysis, **mean_per_transformer_rows(component_analysis)}
    for transformer, results in component_analysis.items():
        name_transformer = "-".join(transformer.split("-")[:-1])
        seed = transformer.split("-")[-1]
        map_matcher = round(results["scores_matcher"]["map"], 3)
        f1_matcher = round(results["scores_matcher"]["summary"], 3)
        map_ranker = round(results["scores_ranker"]["map"], 3)
        f1_ranker = round(results["scores_ranker"]["summary"], 3)
        map_dac = round(results["scores_dac"]["map"], 3)
        f1_dac = round(results["scores_dac"]["summary"], 3)
        line_model = f"{name_transformers.get(name_transformer, name_transformer)} - {seed} & {map_matcher:.3f} & {f1_matcher:.3f} & {map_ranker:.3f} & {f1_ranker:.3f} & {map_dac:.3f} & {f1_dac:.3f}"
        tabular_str += f"{line_model}\\\\ \n"
    print(tabular_str)


def mean_per_transformer_rows(component_analysis):
    grouped_results = {
        "-".join(transformer.split("-")[:-1]): {
            "scores_matcher": {"map": [], "summary": []},
            "scores_ranker": {"map": [], "summary": []},
            "scores_dac": {"map": [], "summary": []},
        }
        for transformer in component_analysis
    }

    for transformer_name, component, metric in itertools.product(
        component_analysis.keys(), ["scores_matcher", "scores_ranker", "scores_dac"], ["map", "summary"]
    ):
        grouped_results["-".join(transformer_name.split("-")[:-1])][component][metric].append(
            component_analysis[transformer_name][component][metric]
        )

    for grouped_transformer in grouped_results.values():
        for component in grouped_transformer.values():
            for metric, results_metrics in component.items():
                component[metric] = mean(results_metrics)
    return grouped_results
