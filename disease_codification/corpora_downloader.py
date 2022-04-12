from pathlib import Path
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
    if not Path(corpuses_path / "codiesp.zip").is_file():
        url = "https://zenodo.org/record/3837305/files/codiesp.zip?download=1"
        wget.download(url, f"{corpuses_path}/codiesp.zip")
        with zipfile.ZipFile(f"{corpuses_path}/codiesp.zip", "r") as zf:
            zf.extractall(corpuses_path)
        shutil.move(corpuses_path / "final_dataset_v4_to_publish", corpuses_path / "codiesp")
    if not Path(corpuses_path / "codiesp_codes.zip").is_file():
        url = "https://zenodo.org/record/3632523/files/codiesp_codes.zip?download=1"
        wget.download(url, f"{corpuses_path}/codiesp_codes.zip")
        with zipfile.ZipFile(f"{corpuses_path}/codiesp_codes.zip", "r") as zf:
            codes_path = create_dir_if_dont_exist(corpuses_path / "codiesp_codes")
            zf.extractall(codes_path)
