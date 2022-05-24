import shutil
import zipfile
from pathlib import Path

import wget

from dac_divide_and_conquer.custom_io import create_dir_if_dont_exist
from dac_divide_and_conquer import logger


def create_directories(data_path: Path):
    corpuses_path = create_dir_if_dont_exist(data_path / "corpuses")
    indexers_path = create_dir_if_dont_exist(data_path / "indexers")
    models_path = create_dir_if_dont_exist(data_path / "models")
    return corpuses_path, indexers_path, models_path


def download_corpus(
    corpuses_path: Path,
    corpus_name: str,
    url: str,
    old_name: str = None,
    create_containing_folder: bool = False,
    file_type: str = "zip",
):
    if not Path(corpuses_path / f"{corpus_name}.{file_type}").is_file():
        wget.download(url, f"{corpuses_path}/{corpus_name}.{file_type}")
        path = corpuses_path
        if create_containing_folder:
            path = create_dir_if_dont_exist(corpuses_path / corpus_name)
        if file_type == "zip":
            with zipfile.ZipFile(f"{corpuses_path}/{corpus_name}.zip", "r") as zf:
                zf.extractall(path)
        if old_name:
            shutil.move(corpuses_path / old_name, corpuses_path / corpus_name)
