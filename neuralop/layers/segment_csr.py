from typing import Literal
import imp

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
    
    # Check if torch_scatter is installed
    try:
        imp.find_module('torch_scatter')
        scatter_avail = True
    except ImportError:
        scatter_avail = False

    if torch.backends.cuda.is_built() and scatter_avail:
        """only import torch_scatter when cuda is available"""
        import torch_scatter.segment_csr as scatter_segment_csr
        return scatter_segment_csr(src, indptr, reduce)
    else:
        n_nbrs = indptr[1:] - indptr[:-1] # end indices - start indices
        output_shape = list(src.shape)
        output_shape[0] = indptr.shape[0] - 1

        out = torch.zeros(output_shape, device=src.device)

        for i,start in enumerate(indptr[:-1]):
            if start == src.shape[0]: # if the last neighborhoods are empty, skip
                break
            accum = 0
            for j in range(n_nbrs[i]):
                accum += src[start + j]
            if reduce == 'mean':        
                accum /= n_nbrs[i]
            
            out[i] = accum
        
        return out



    
    
