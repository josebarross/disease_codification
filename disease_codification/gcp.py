# Cloud Storage
import os
import pickle

from dotenv import load_dotenv
from google.cloud import storage

from disease_codification import logger

load_dotenv()


def upload_blob(python_obj: object, filename: str):
    """Uploads a python object to the bucket."""
    logger.info(f"Uploading to {filename}")
    storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
    pickle_out = pickle.dumps(python_obj)
    bucket = storage_client.bucket(os.getenv("BUCKET"))
    blob = bucket.blob(filename)
    blob.upload_from_string(pickle_out)


def upload_blob_file(filename_in: object, filename_out: str):
    """Uploads a python object to the bucket."""
    logger.info(f"Uploading {filename_in} to {filename_out}")
    storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
    bucket = storage_client.bucket(os.getenv("BUCKET"))
    blob = bucket.blob(filename_out)
    blob.upload_from_filename(filename_in)


def download_blob(filename: str):
    """Uploads a python object to the bucket."""
    logger.info(f"Downloading to {filename}")
    storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
    bucket = storage_client.bucket(os.getenv("BUCKET"))
    blob = bucket.blob(filename)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)


def download_blob_file(filename_in: str, filename_out: str):
    logger.info(f"Downloading {filename_in} to {filename_out}")
    storage_client = storage.Client(project=os.getenv("PROJECT_ID"))
    bucket = storage_client.bucket(os.getenv("BUCKET"))
    blob = bucket.blob(filename_in)
    blob.download_to_filename(filename_out)
