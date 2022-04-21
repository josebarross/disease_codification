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


def download_cantemist_corpus(corpuses_path: Path):
    print("Downloading CANTEMIST corpus")
    url = "https://zenodo.org/record/3978041/files/cantemist.zip?download=1"
    download_corpus(corpuses_path, "cantemist", url, create_containing_folder=True)
    print("Downloading CIE 3 O descriptions")
    url = "https://eciemaps.mscbs.gob.es/ecieMaps/download?name=2018_CIEO31_TABLA_%20REFERENCIA_con_6_7_caracteres_final_20180111_5375350050755186721_7033700654037542595.xlsx"
    download_corpus(corpuses_path, "cie-o-3-codes", url, file_type="xlsx")


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
