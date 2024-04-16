'''
Utils for downloading dataset files from the web.

Mostly borrowed from torchvision.datasets.utils
'''

import hashlib
from logging import Logger
import os
from pathlib import Path
import requests
import sys
from typing import Any, Optional, Union

logger = Logger(name='neuralop.datasets.web_utils')

def download_from_url(url: str, filename: Union[str, Path], chunk_size: int = 256 * 32) -> None:
    """download_from_url is a simple requests util to stream
       a file download url into a file

    Parameters
    ----------
    url : str
        url at which file is stored
    filename : Union[str, Path]
        path into which to save streamed data
    chunk_size : int, optional
        size of chunks to stream, by default 256*32
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

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

def download_dataset(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
) -> None:
    """download_dataset downloads a file from a url with 
    an optional md5 checksum.

    Parameters
    ----------
    url : str
        _description_
    root : Union[str, Path]
        _description_
    filename : Optional[Union[str, Path]], optional
        _description_, by default None
    md5 : Optional[str], optional
        _description_, by default None

    Raises
    ------
    RuntimeError
        _description_
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

    download_from_url(url, fpath)

    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
    else:
        logger.info