import os.path
import urllib.request

__all__ = ["download_from_url"]

def download_from_url(url,
                      path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    data = urllib.request.urlopen(url)
    with open(path, 'wb') as file:
        file.write(data.read())
