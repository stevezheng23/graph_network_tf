import os.path
import urllib.request
import tarfile
import zipfile

__all__ = ["download_from_url", "extract_zip", "extract_tar"]

def download_from_url(url,
                      path):
    if not os.path.exists(path):
        os.mkdir(path)
    
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
