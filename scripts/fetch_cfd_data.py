"""
Script to fetch and preprocess car-CFD dataset
this assumes that at ~/data/cfd, you have downloaded 
the preprocessed folder car-pressure-data.zip
"""
import argparse
import os
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("--data_root", help="root directory in which car-pressure-data.zip is downloaded and unzipped")
parser.add_argument("--verbose", "-v", action="store_true")

args = parser.parse_args()

if args.data_root:
    data_root = args.data_root
else:
    home_dir = os.getenv("HOME")
    data_root = f"{home_dir}/data/cfd/car-pressure-data"


with open(f"{data_root}/watertight_meshes.txt", "r") as f:
    mesh_inds = [x.rstrip() for x in f.readlines()]

train_inds = ','.join(mesh_inds[:500])
test_inds = ','.join(mesh_inds[500:])

with open(f"{data_root}/train.txt", "w+") as f:
    f.write(train_inds)
    f.close()

with open(f"{data_root}/test.txt", "w+") as f:
    f.write(test_inds)
    f.close()

for ind in mesh_inds:
    if args.verbose:
        print(f"Copied file #{ind}/{len(mesh_inds)}", end='\r')
    os.mkdir(f"{data_root}/data/{ind}/")
    shutil.copyfile(f"{data_root}/data/mesh_{ind}.ply", f"{data_root}/data/{ind}/tri_mesh.ply")
    shutil.copyfile(f"/{data_root}/data/press_{ind}.npy", f"{data_root}/data/{ind}/press.npy")