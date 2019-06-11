import os.path
import json
import pickle
import urllib.request
import shutil
import tarfile
import zipfile

__all__ = ["download_from_url", "extract_zip", "extract_tar", "valid_path", "clear_path",
           "read_text", "read_json", "read_pickle", "save_text", "save_json", "save_pickle"]

def download_from_url(data_url,
                      data_path):
    data = urllib.request.urlopen(data_url)
    with open(data_path, 'wb') as file:
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

def valid_path(data_path):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

def clear_path(data_path):
    if os.path.exists(data_path):
        shutil.rmtree(data_path)

def read_text(data_path):
    if os.path.exists(data_path):
        with open(data_path, "r") as file:
            return [line.rstrip('\n') for line in file]
    else:
        raise FileNotFoundError("input file not found")

def read_json(data_path):
    if os.path.exists(data_path):
        with open(data_path, "r") as file:
            return json.load(file)
    else:
        raise FileNotFoundError("input file not found")

def read_pickle(data_path):
    if os.path.exists(data_path):
        with open(data_path, "rb") as file:
            return pickle.load(file, encoding='latin1')
    else:
        raise FileNotFoundError("input file not found")

def save_text(data_list,
              data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "w") as file:
        for data in data_list:
            file.write("{0}\n".format(data))

def save_json(data_list,
              data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "w") as file:  
        json.dump(data_list, file, indent=4)

def save_pickle(data_list,
                data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "wb") as file:  
        pickle.dump(data_list, file)
