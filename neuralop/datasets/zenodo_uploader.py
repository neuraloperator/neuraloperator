import os
import requests
import sys

from pathlib import Path

zenodo_token = os.getenv("ZENODO_TOKEN")
headers = {"Content-Type": "application/json"}
params = {'access_token': zenodo_token}

r = requests.post('https://sandbox.zenodo.org/api/deposit/depositions',
                   params=params,
                   json={},
                   headers=headers)

if r.status_code == 201:
    bucket_url = r.json()["links"]["bucket"]
else:
    print("Error: bad request. Terminating.")
    sys.exit(1)

splits = ['train', 'test']
resolutions = [32, 64]
data_root = Path("/home/dpitt/work/data/darcy")
for split in splits:
    for resolution in resolutions:
        fname = f"darcy_{split}_{resolution}.pt"
        fpath = data_root / fname
        with open(fpath, "rb") as fp:
            r = requests.put(
                "%s/%s" % (bucket_url, fname),
                data=fp,
                params=params,
            )
        deposit_id = r.json().id
    data = {
        'metadata': {
            'title': f'{resolution}x{resolution} Darcy Flow',
            'upload_type': 'poster',
            'description': '{resolution}x{resolution} dataset of input-output pairs of PyTorch Tensors governed by Darcy\'s Law',
            'creators': [{'name': 'Kovachki, Nik',
                        'affiliation': 'NVIDIA/Caltech'}]
        }
    }
    r = requests.put('https://zenodo.org/api/deposit/depositions/%s' % deposition_id,
                  params={'access_token': zenodo_token}, data=json.dumps(data),
                  headers=headers)
