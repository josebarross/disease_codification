def chunks(lst, n: int = 10):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def label_in_cluster(cluster: str, label: str, mappings: dict, multi_cluster: bool):
    if multi_cluster:
        return cluster in mappings[label]
    else:
        return cluster == mappings[label]
