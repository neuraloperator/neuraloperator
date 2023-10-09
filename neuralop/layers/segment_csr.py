"""
Replacement for torch_scatter.segment_csr

"""
from typing import Literal
import torch

def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: Literal['mean', 'sum']):
    """segment_csr reduces all entries of a CSR-formatted 
    matrix by summing or averaging over neighbors. 

    Used to reduce features over neighborhoods 
    in neuralop.layers.IntegralTransform
    
    Parameters
    ----------
    src : torch.Tensor
        tensor of features for each point
    indptr : torch.Tensor
        splits representing start and end indices 
        of each neighborhood in src
    reduce : Literal['mean', 'sum']
        how to reduce a neighborhood. if mean,
        reduce by taking the average of all neighbors.
        Otherwise take the sum. 
    """
    if reduce not in ['mean', 'sum']:
        raise ValueError("reduce must be one of \'mean\', \'sum\'")

    n_nbrs = indptr[1:] - indptr[:-1] # end indices - start indices
    n_neighborhoods = indptr.shape[0] - 1
    output_shape = list(src.shape)
    output_shape[0] = n_neighborhoods

    out = torch.zeros(output_shape, device=src.device)

    for i in range(n_neighborhoods):
        nbr_idx = indptr[i]

        accum = src[nbr_idx]
        for j in range(1,n_nbrs[i]):
            accum += src[nbr_idx + j]
        if reduce == 'mean':
            accum /= n_nbrs[i]
        
        out[i] = accum
    
    return out

            


    
    
