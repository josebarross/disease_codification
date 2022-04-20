from pathlib import Path
from venv import create
import zipfile
import shutil
import wget

from disease_codification.custom_io import create_dir_if_dont_exist


def create_directories(data_path: Path):
    corpuses_path = create_dir_if_dont_exist(data_path / "corpuses")
    indexers_path = create_dir_if_dont_exist(data_path / "indexers")
    models_path = create_dir_if_dont_exist(data_path / "models")
    return corpuses_path, indexers_path, models_path


def download_codiesp_corpus(corpuses_path: Path):
    print("Downloading codiesp corpus")
    url = "https://zenodo.org/record/3837305/files/codiesp.zip?download=1"
    download_zipped_corpus(corpuses_path, "codiesp", url, old_name="final_dataset_v4_to_publish")
    print("Downloading codiesp labels")
    url = "https://zenodo.org/record/3632523/files/codiesp_codes.zip?download=1"
    download_zipped_corpus(corpuses_path, "codiesp_codes", url, create_containing_folder=True)


def download_livingner_corpus(corpuses_path: Path):
    print("Downloading LIVINGNER corpus")
    url = "https://zenodo.org/record/6421410/files/training_valid.zip?download=1"
    download_zipped_corpus(corpuses_path, "livingner", url, create_containing_folder=True)


def download_zipped_corpus(
    corpuses_path: Path, corpus_name: str, url: str, old_name: str = None, create_containing_folder: bool = False
):
    if not Path(corpuses_path / f"{corpus_name}.zip").is_file():
        wget.download(url, f"{corpuses_path}/{corpus_name}.zip")
        with zipfile.ZipFile(f"{corpuses_path}/{corpus_name}.zip", "r") as zf:
            path = corpuses_path
            if create_containing_folder:
                path = create_dir_if_dont_exist(corpuses_path / corpus_name)
            zf.extractall(path)
    if old_name:
        shutil.move(corpuses_path / old_name, corpuses_path / corpus_name)
