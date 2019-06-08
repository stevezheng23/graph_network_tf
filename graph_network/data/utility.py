import os.path
import urllib.request
import shutil
import tarfile
import zipfile

__all__ = ["download_from_url", "extract_zip", "extract_tar", "valid_path", "clear_path", "read_file"]

def download_from_url(url,
                      path):
    data = urllib.request.urlopen(url)
    with open(path, 'wb') as file:
        file.write(data.read())

def extract_zip(zip_path,
                file_path):
    with zipfile.ZipFile(zip_path, 'r') as file:
        file.extractall(file_path)

def extract_tar(tar_path,
                file_path,
                mode='r:gz'):
    with tarfile.open(tar_path, mode) as file:
        file.extractall(file_path)

def valid_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def clear_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def read_file(path):
    with open(path) as file:
        data_list = [line.rstrip('\n') for line in file]
        return data_list