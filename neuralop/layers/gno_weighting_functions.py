from functools import partial
import torch

def bump_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = x.clip(0., radius) / radius
    out = - 1 / ((1 - out ** 2) + eps)
    return out.exp() * torch.e * scale

def half_cos_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (0.5 * torch.cos(torch.pi * x) + 0.5)

def quadr_cutoff(x, radius=1., scale=1.):
    x = x / radius
    left = 1 - 2 * x ** 2
    right = 2 * (1 - x) ** 2
    return scale * torch.where(x < 0.5, left, right)

def quartic_cutoff(x, radius=1., scale=1.):
    a = scale / radius ** 4
    c = - 2 * scale / radius ** 2 
    return a * x ** 4 + c * x ** 2 + scale

def octic_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (-3 * x ** 8 + 8 * x ** 6 - 6 * x ** 4  + 1)

WEIGHTING_FN_REGISTRY = {
    "bump": bump_cutoff,
    "half_cos": half_cos_cutoff,
    "quadr": quadr_cutoff,
    "quartic": quartic_cutoff,
    "octic": octic_cutoff,
}

def dispatch_weighting_fn(weight_function_name : str, sq_radius: float, scale: float):
    '''
    Select a GNO weighting function for use in output GNO 
    of a Mollified Graph Neural Operator-based model. See [1]_ (add later)

    Parameters
    ----------
    weight_function_name : str Literal
        name of weighting function to use, keyed to ``WEIGHTING_FN_REGISTRY`` above
    sq_radius : float
        squared radius of GNO neighborhoods for Nyström approximation
    scale : float
        factor by which to scale all weights
    '''
    base_func = WEIGHTING_FN_REGISTRY.get(weight_function_name)
    if base_func is None:
        raise NotImplementedError(f"weighting function should be one of {list(WEIGHTING_FN_REGISTRY.keys())}, got {weight_function_name}")
    return partial(base_func, radius=sq_radius, scale=scale)
