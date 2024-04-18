'''
Utils for downloading dataset files from the web.

Mostly borrowed from torchvision.datasets.utils
'''

import hashlib
import logging
import os
from pathlib import Path
import requests
import sys
from typing import Any, Optional, Union, List
import tarfile

logger = logging.Logger(name=__name__)
logger.setLevel(logging.root.level)

# md5 utils from torchvision.datasets.utils

def calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()

def check_md5(fpath: Union[str, Path], md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)

def check_integrity(fpath: Union[str, Path], md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def download_from_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    chunk_size: Optional[int] = 256 * 32,
    extract_tars: bool = True
) -> None:
    """download_from_url downloads a file from a url with 
    an optional md5 checksum.

    Parameters
    ----------
    url : str
        url where dataset archive file is stored
    root : Union[str, Path]
        root of dataset folder on local machine
    filename : Optional[Union[str, Path]], optional
        name into which to download, by default None
    md5 : Optional[str], optional
        md5 checksum to verify file correctness, by default None
    chunk_size : int, optional
        size of chunks to use when streaming files
    extract_tars: bool, optional
        whether to extract .tgz archives, by default True
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, filename))

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        logger.info("Using downloaded and verified file: " + fpath)
        return

    # download and stream file in chunks
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fpath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    else:
        logger.info('saved to {fpath} successfully')
        # extract tarfiles to their given filenames on root
        if extract_tars:
            logger.info("Extracting {fpath}...")
            archive = tarfile.open(fpath)
            archive.extractall(fpath=root)
            name_str = ", ".join(archive.getnames())
            logger.info(f"Extracted {name_str}")

def download_from_zenodo_record(record_id: str, 
                                root: Union[str, Path], 
                                files_to_download: Optional[List[str]] = None):
    """download_from_zenodo_record _summary_

    Parameters
    ----------
    record_id : str
        ID of record in Zenodo's database
    root : Union[str, Path]
        root directory into which to download archive
    files_to_download : Optional[List[str]]
        list of filenames to download from record
    """
    zenodo_api_url = "https://zenodo.org/api/records/"
    url = f"{zenodo_api_url}{record_id}"
    resp = requests.get(url)
    assert resp.status_code == 200, f"Error: request failed with status code {resp.status_code}"
    response_json = resp.json()
    for file_record in response_json['files']:
        fname = file_record['key']
        if fname in files_to_download:
            download_from_url(url=file_record['links']['self'],
                              md5=file_record['checksum'],
                              root=root,
                              filename=fname,
                              extract_tars=True)
        
    


