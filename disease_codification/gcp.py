# Cloud Storage
import pickle
from google.cloud import storage

storage_client = storage.Client(project="tesis-334522")


def upload_blob(bucket_name: str, python_obj: object, filename: str):
    """Uploads a python object to the bucket."""
    pickle_out = pickle.dumps(python_obj)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(pickle_out)


def upload_blob_file(bucket_name: str, filename_in: object, filename_out: str):
    """Uploads a python object to the bucket."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename_out)
    blob.upload_from_filename(filename_in)


def download_blob(bucket_name: str, filename: str):
    """Uploads a python object to the bucket."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)


def download_blob_file(bucket_name: str, filename_in: str, filename_out: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename_in)
    blob.download_to_filename(filename_out)
