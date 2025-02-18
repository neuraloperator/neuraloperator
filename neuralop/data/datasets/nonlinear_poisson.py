import torch
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import pickle

from torch.utils.data import DataLoader
from .dict_dataset import DictDataset


path = Path(__file__).resolve().parent.joinpath('data')

def generate_latent_queries(query_res, pad=0, domain_lims=[[-1.4,1.4],[-1.4,1.4]]):
    oneDMeshes = []
    for lower,upper in domain_lims:
        oneDMesh = np.linspace(lower,upper,query_res)
        if pad > 0:
            start = np.linspace(lower - pad/query_res, lower, pad+1)
            stop = np.linspace(upper, upper + pad/query_res, pad+1)
            oneDMesh = np.concatenate([start,oneDMesh,stop])
        oneDMeshes.append(oneDMesh)
    grid = np.stack(np.meshgrid(*oneDMeshes,indexing='xy')) # c, x, y, z(?)
    grid = torch.from_numpy(grid.astype(np.float32))
    latent_queries = grid.permute(*list(range(1,len(domain_lims)+1)), 0)
    return latent_queries

def generate_output_grid(grid_res, coefs, domain_lims=[[-1.4,1.4],[-1.4,1.4]], tol=1e-7):
    xi = domain_lims[0][0] + (domain_lims[0][1] - domain_lims[0][0]) * torch.rand(grid_res)
    yi = domain_lims[1][0] + (domain_lims[1][1] - domain_lims[1][0]) * torch.rand(grid_res)

    c1, c2 = coefs['c1'].item(), coefs['c2'].item()
    
    # Rejection sampling
    theta = torch.arctan2(xi, yi)
    length = torch.sqrt(xi**2 + yi**2)
    r0 = 1.0 + c1 * torch.cos(4 * theta) + c2 * torch.cos(8 * theta)
    mask = r0 > length - tol

    X_filtered = xi[mask]
    Y_filtered = yi[mask]

    XY_filtered = torch.stack((X_filtered, Y_filtered), dim=-1)

    return XY_filtered

def source(x, beta, mu_1, mu_2):
    # Calculate the squared differences between x and the means
    diff = (x[:, 0].unsqueeze(1) - torch.tensor(mu_1)) ** 2 + (x[:, 1].unsqueeze(1) - torch.tensor(mu_2)) ** 2
    exponent = torch.exp(-diff)
    
    # Multiply by beta and sum along the appropriate axis
    result = torch.tensor(beta) * exponent
    source_terms = result.sum(dim=1)

    return source_terms


def load_nonlinear_poisson_pt(
        data_path, 
        query_res=48, 
        domain_padding=0, 
        encode=True, 
        val_on_same_instance=False,
        n_train=1, 
        n_test=1,
        n_in=6000,
        n_out=6000,
        n_eval=6000,
        n_bound = 1024,
        out_sub_level=None,
        train_out_res=None
        ):
    try:
        with open(data_path, 'rb') as file:
            data = pickle.load(file)
            print("Dictionary loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file was not found.")
    random.shuffle(data)

    train_end = int(0.7*len(data))
    
    if n_train > train_end:
        n_train = train_end
        print('WARNING: Max n_train is 0.7 of the length of the data file. Overriding.')
    print(f"{n_train=}")
    if n_test > len(data) - train_end:
        n_test =  len(data) - train_end
    print(f"{n_test=}")

    data_list = []

    for idx, instance in enumerate(data):
        f_f = torch.tensor(instance['train_source_terms_domain'][:n_in], dtype=torch.float32)
        f_g = torch.tensor(instance['train_bc_domain'][:n_in], dtype=torch.float32)
        f_dist = torch.tensor(instance['train_distances_domain'][:n_in], dtype=torch.float32)

        input_geom = torch.tensor(instance['train_points_domain'][:n_in], dtype=torch.float32)

        if idx < n_train:
            if train_out_res:
                # Using uniform output mesh
                out_p_domain = generate_output_grid(train_out_res, instance['coefs'])
                out_source_domain = source(out_p_domain, instance['coefs']['beta'], instance['coefs']['mu_1'], instance['coefs']['mu_2'])
                y_domain = torch.ones(out_p_domain.shape[0])
                out_p_domain.requires_grad = True
            else:
                # Using randomly sampled Fenics mesh
                out_p_domain = torch.tensor(instance['val_points_domain'][:n_out], dtype=torch.float32)
                out_source_domain = torch.tensor(instance['val_source_terms_domain'][:n_out], dtype=torch.float32)
                y_domain = torch.tensor(instance['val_values_domain'][:n_out], dtype=torch.float32)
                out_p_domain.requires_grad = True

            # Give the boundary points for the output during training
            out_p_bound = torch.tensor(instance['val_points_boundary'][:n_bound], dtype=torch.float32)
            out_source_bound = torch.tensor(instance['val_source_terms_boundary'][:n_bound], dtype=torch.float32)
            y_bound = torch.tensor(instance['val_values_boundary'][:n_bound], dtype=torch.float32)

            out_p = torch.cat((out_p_bound, out_p_domain))
            out_source = torch.cat((out_source_bound, out_source_domain))
            y = torch.cat((y_bound, y_domain))
        else:
            out_p = torch.tensor(instance['val_points_domain'][:n_eval], dtype=torch.float32)
            out_source = torch.tensor(instance['val_source_terms_domain'][:n_eval], dtype=torch.float32)
            y = torch.tensor(instance['val_values_domain'][:n_eval], dtype=torch.float32)

        f_f = torch.cat((torch.tensor(instance['train_source_terms_boundary'][:n_in], dtype=torch.float32), f_f))
        f_g = torch.cat((torch.tensor(instance['train_bc_boundary'][:n_in], dtype=torch.float32), f_g))
        f_dist = torch.cat((torch.zeros(n_in), f_dist))
        input_geom = torch.vstack((torch.tensor(instance['train_points_boundary'][:n_in], dtype=torch.float32), input_geom))

        f_f = f_f.unsqueeze(dim=-1)
        f_g = f_g.unsqueeze(dim=-1)
        f_dist = f_dist.unsqueeze(dim=-1)

        y = y.unsqueeze(dim=-1)

        f = torch.cat((f_f, f_g, f_dist), dim=-1)
        latent_queries = generate_latent_queries(query_res=query_res,
                                                pad=domain_padding
                                                )
        
        data_dict = {'x': f, 
                    'input_geom': input_geom,
                    'latent_queries': latent_queries,
                    'output_queries_domain': out_p_domain,
                    'output_source_terms_domain': out_source_domain,
                    'y_domain': y_domain,
                    'output_queries_bound': out_p_bound,
                    'output_source_terms_bound': out_source_bound,
                    'y_bound': y_bound,
                    'output_queries': out_p,
                    'output_source_terms': out_source,
                    'y': y.unsqueeze(0),
                    'coefs': instance['coefs'],
                    'num_boundary': n_bound,
                    'out_sub_level': out_sub_level if out_sub_level else 1
                    }
        data_list.append(data_dict)
    
    train_data = data_list[:n_train]

    if val_on_same_instance:
        test_data = train_data.copy()
    else:
        test_data = data_list[train_end:train_end+n_test]


    train_dataloader = DataLoader(DictDataset(train_data)) 
    test_dataloader = DataLoader(DictDataset(test_data)) 
    
    return train_dataloader, test_dataloader, None

if __name__ == "__main__":
    train_data = load_nonlinear_poisson_pt(str(path))
    print(np.array(train_data).shape)