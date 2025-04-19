
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Sequence
from typing import List, Optional, Tuple, Union
from .resample import resample
from ..utils import validate_scaling_factor
from .shape_enforcer import ShapeEnforcer
from .base_spectral_conv import BaseSpectralConv

Number = Union[int, float]


def _compute_dt(
    shape, 
    start_points: List=None, 
    end_points: List=None
    ):
    """
    Compute uniform spacing (dt) for each dimension based on domain lengths, step sizes,
    start points, and end points. Defaults to a unit domain if not specified.

    Parameters:
    shape (Sequence[int]): The shape of the input excluding batch and channel, i.e. (d_1, d_2, ..., d_n).
    step_sizes (Sequence[float], optional): Step sizes for each dimension. Defaults to shape-based uniform spacing.
    start_points (Sequence[float], optional): Start points for each dimension. Defaults to 0.0 for all dimensions.
    end_points (Sequence[float], optional): End points for each dimension. Defaults to 1.0 for all dimensions.

    Returns:
    dt_list (Sequence[float]): A list of spacings, one per dimension.
    grid (List[torch.Tensor]): A list of grid points for each dimension based on the spacing and domain.
    """
    dim = len(shape)
    # Set default start and end points if not provided
    if start_points is None:
        start_points = torch.zeros(dim).tolist()
    if end_points is None:
        end_points = torch.ones(dim).tolist()

    # Validate that start_points and end_points match the number of dimensions
    if len(start_points) != dim or len(end_points) != dim:
        raise ValueError("Start points and end points must match the number of input dimensions ({dim}).")

    # Compute domain lengths from start and end points
    domain_lengths = [end_points[i] - start_points[i] for i in range(dim)]

    # Generate grid points for each dimension using torch.linspace
    grid = [torch.linspace(start_points[i], end_points[i], steps=shape[i]) for i in range(dim)]

    # Compute dt directly from the grid
    dt_list = [(grid[i][1] - grid[i][0]).item() for i in range(dim)]

    return dt_list, grid



# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================
class SpectralConvLaplace1D(BaseSpectralConv):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None, 
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None, 
        
        ):
        super(SpectralConvLaplace1D, self).__init__(device=device)
        
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        self.n_modes = n_modes
        
        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
        
        max_n_modes, = self.n_modes
        self.max_n_modes = self.n_modes
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Initialize single weight tensor combining poles and residues
        
        total_modes = max_n_modes + 1 * max_n_modes
        self.weight = nn.Parameter(
            self.scale * torch.rand(
                in_channels, 
                out_channels, 
                total_modes, 
                dtype=torch.cfloat, 
                )
        )
        
        self.shape_enforcer = ShapeEnforcer()
       
    
    def transform(
        self, 
        x, 
        output_shape=None
        ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
    
    def output_PR(
        self, 
        lambda1,
        alpha, 
        weights_pole, 
        weights_residue
        ):   
        
        Hw=torch.zeros(
            weights_residue.shape[0],
            weights_residue.shape[0],
            weights_residue.shape[2],
            lambda1.shape[0], 
            device=alpha.device, 
            dtype=torch.cfloat
        )
        
        term1=torch.div(1,
                        torch.sub(lambda1,
                                  weights_pole
                        )
                        )

        Hw=weights_residue*term1

        output_residue1=torch.einsum("bix,xiok->box", 
                                     alpha, 
                                     Hw
                                     ) 
        output_residue2=torch.einsum("bix,xiok->bok", 
                                     alpha, 
                                     -Hw
                                     ) 
        
        return output_residue1,output_residue2    

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        
        batchsize, channels, *mode_sizes = x.shape
                
        modes1, = self.n_modes
        L = x.shape[-1]
        
        # Ensure we do not exceed the actual resolution
        modes1 = min(modes1, L)
        
        # if self.linspace_steps is None:
        #     self.linspace_steps = x.shape[2:]
        self.linspace_steps = x.shape[2:]
            
        dt_list, shape = _compute_dt(
            shape=self.linspace_steps, 
            start_points=self.linspace_startpoints, end_points=self.linspace_endpoints
        )
        
        t = shape[0]
        t = t.to(x.device)
        dt = dt_list[0]        
        
        alpha = torch.fft.fft(x, dim=-1)
        lambda0=torch.fft.fftfreq(t.shape[0], dt, device=alpha.device)*2*np.pi*1j
        # lambda1=lambda0[:modes1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Slice alpha and weights to match the truncated modes
        # alpha = alpha[:, :, :modes1]
        
        
        weights_pole = self.weight[:, :, :modes1].view(self.weight.size(0), self.weight.size(1), modes1)
        weights_residue = self.weight[:, :, modes1:(modes1 * 2)].view(self.weight.size(0), self.weight.size(1), modes1)
    
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, 
                                                        alpha, 
                                                        weights_pole, 
                                                        weights_residue
                                                        )
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        
        x2=torch.zeros(output_residue2.shape[0],
                       output_residue2.shape[1],
                       t.shape[0],
                       device=alpha.device,
                       dtype=torch.cfloat)    
        
        term1=torch.einsum("iok,az->iokz", 
                           weights_pole, 
                           t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        
        x2=torch.einsum("bok,iokz->boz",
                        output_residue2,
                        term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        
        x = x1+x2
        
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
            
        if output_shape is not None:
            mode_sizes = output_shape
            
        # Ensuring the ouputshape is matched with desired, if it's specified
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)
            
        return x


class SpectralConvLaplace2D(BaseSpectralConv):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None
        ):
        
        super(SpectralConvLaplace2D, self).__init__(device=device)
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        
        self.n_modes = n_modes

        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
        
        """
            Commulating all weights into a single weight attribute on the class, and break it down for different applications like weights_pole1, etc. in convolution process. 
        """
        max_modes1, max_modes2 = self.n_modes
        self.max_n_modes = self.n_modes
        
        # if max_n_modes is None:
        #     max_modes1, max_modes2 = self.n_modes
        # elif isinstance(max_n_modes, int):
        #     max_modes1, max_modes2 = max_n_modes

        self.scale = 1 / (in_channels * out_channels)
        
        # Initialize single weight tensor combining poles and residues
        # Shape: (in_channels, out_channels, modes1 + modes2 + modes1 * modes2)
        total_modes = max_modes1 + max_modes2 + (max_modes1 * max_modes2)
        self.weight = nn.Parameter(
            self.scale * torch.rand(
                in_channels, 
                out_channels, 
                total_modes, 
                dtype=torch.cfloat
            )
        )
        
        self.shape_enforcer = ShapeEnforcer()
        
    def transform(
        self, 
        x, 
        output_shape=None
    ):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
    
    def output_PR(
        self, 
        lambda1, 
        lambda2, 
        alpha, 
        weights_pole1, 
        weights_pole2, 
        weights_residue
    ):
        Hw=torch.zeros(weights_residue.shape[0],
                       weights_residue.shape[0],
                       weights_residue.shape[2],
                       weights_residue.shape[3],
                       lambda1.shape[0], 
                       lambda2.shape[0], 
                       device=alpha.device, 
                       dtype=torch.cfloat
                       )
        
        term1=torch.div(1,
                        torch.einsum("pbix,qbik->pqbixk",
                                     torch.sub(lambda1,weights_pole1),
                                     torch.sub(lambda2,weights_pole2)))
        
        Hw=torch.einsum("bixk,pqbixk->pqbixk",
                        weights_residue,
                        term1)
        
        Pk=Hw  # for ode, Pk=-Hw; for 2d pde, Pk=Hw; for 3d pde, Pk=-Hw; 
        output_residue1=torch.einsum("biox,oxikpq->bkox", 
                                     alpha, 
                                     Hw) 
        output_residue2=torch.einsum("biox,oxikpq->bkpq",
                                     alpha, 
                                     Pk) 
        return output_residue1,output_residue2

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        batchsize, channels, *mode_sizes = x.shape
        
        modes1, modes2 = self.n_modes
        H, W = x.shape[-2], x.shape[-1]

        # Ensure we do not exceed the actual resolution
        modes1 = min(modes1, H)
        modes2 = min(modes2, W)
            
        # if self.linspace_steps is None:
        #     self.linspace_steps = x.shape[2:]
        self.linspace_steps = x.shape[2:]

        dt_list, shape = _compute_dt(
            shape=self.linspace_steps, 
            start_points=self.linspace_startpoints, 
            end_points=self.linspace_endpoints
        )

        ty = shape[0]
        tx = shape[1]
        ty = ty.to(x.device)
        tx = tx.to(x.device)
        dty = dt_list[0]
        dtx = dt_list[1]
                
        alpha = torch.fft.fft2(x, dim=[-2, -1])

        # Compute frequency grids
        omega1 = torch.fft.fftfreq(ty.shape[0], dty, device=alpha.device)*2*np.pi*1j
        omega2 = torch.fft.fftfreq(tx.shape[0], dtx, device=alpha.device)*2*np.pi*1j

        # Slice frequencies to match the chosen modes
        # omega1 = omega1[:modes1]
        # omega2 = omega2[:modes2]

        omega1 = omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2 = omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1 = omega1
        lambda2 = omega2

        # Slice alpha to only consider the selected modes
        # alpha = alpha[:, :, :modes1, :modes2]

        # Slice weights to match the truncated modes
        weights_pole1 = self.weight[:, :, :modes1].view(self.weight.size(0), self.weight.size(1), modes1)
        weights_pole2 = self.weight[:, :, modes1:(modes1+modes2)].view(self.weight.size(0), self.weight.size(1), modes2)
        weights_residue = self.weight[:, :, (modes1+modes2):(modes1+modes2+modes1*modes2)].view(self.weight.size(0), self.weight.size(1), modes1, modes2)
        # Proceed with the existing logic
        output_residue1, output_residue2 = self.output_PR(lambda1, 
                                                          lambda2, 
                                                          alpha, 
                                                          weights_pole1, 
                                                          weights_pole2, 
                                                          weights_residue)
        
        x1 = torch.fft.ifft2(output_residue1, s=(x.size(-2), x.size(-1)))
        x1 = torch.real(x1)  # shape: (b, o, H, W)

        # Now ty has length H and tx has length W
        term1 = torch.einsum("iop,z->iopz", 
                             weights_pole1, 
                             ty.type(torch.complex64))  # (i, o, p, H)
        term2 = torch.einsum("ioq,x->ioqx", 
                             weights_pole2, 
                             tx.type(torch.complex64))  # (i, o, q, W)        

        term1 = torch.exp(term1)  # (i, o, p, H)
        term2 = torch.exp(term2)  # (i, o, q, W)

        term3 = torch.einsum("iopz,ioqx->iopqzx", 
                             term1, 
                             term2)  # (i, o, p, q, H, W)

        # output_residue2: (b, o, p, q)
        # term3: (o, p, q, H, W)
        x2 = torch.einsum("bopq,iopqzx->bozx", 
                          output_residue2, 
                          term3)  # (b, o, H, W)

        x2 = torch.real(x2) / (x.size(-1) * x.size(-2))
        
        x = x1+x2  # Both are (b, o, H, W)
        
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
            
        if output_shape is not None:
            mode_sizes = output_shape
            
        # Ensuring the ouputshape is matched with desired, if it's specified
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)
            
        return x

class SpectralConvLaplace3D(BaseSpectralConv):
    def __init__(
        self, 
        in_channels, 
        out_channels,  
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        xno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        linspace_steps=None, 
        linspace_startpoints=None, 
        linspace_endpoints=None
        ):
        super(SpectralConvLaplace3D, self).__init__(device=device)
        
        self.linspace_steps = linspace_steps
        self.linspace_startpoints = linspace_startpoints
        self.linspace_endpoints = linspace_endpoints
        self.n_modes = n_modes
        
        self.order = len(self.n_modes)
        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.order)
        
        self.modes1, self.modes2, self.modes3 = self.n_modes
         
        max_modes1, max_modes2, max_modes3 = self.n_modes
        self.max_n_modes = self.n_modes
         
        # if max_n_modes is None:
        #     max_modes1, max_modes2, max_modes3 = self.n_modes
        # elif isinstance(max_n_modes, int):
        #     max_modes1, max_modes2, max_modes3 = max_n_modes
            
        # Compute total number of modes to combine into single weight tensor
        total_modes = max_modes1 + max_modes2 + max_modes3 + (max_modes1 * max_modes2 * max_modes3)

        self.scale = (1 / (in_channels * out_channels))
        self.weight = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, total_modes, dtype=torch.cfloat)
        )
        
        self.shape_enforcer = ShapeEnforcer()
    
    def transform(
        self, 
        x, 
        output_shape=None
    ):
        
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)
    
    def output_PR(
        self, 
        lambda1, 
        lambda2, 
        lambda3, 
        alpha, 
        weights_pole1, 
        weights_pole2, 
        weights_pole3, 
        weights_residue
        ):
        
        Hw=torch.zeros(
            weights_residue.shape[0],
            weights_residue.shape[0],
            weights_residue.shape[2],
            weights_residue.shape[3],
            weights_residue.shape[4],
            lambda1.shape[0], 
            lambda2.shape[0], 
            lambda2.shape[3], 
            device=alpha.device, 
            dtype=torch.cfloat)
        
        term1=torch.div(
            1,
            torch.einsum("pbix,qbik,rbio->pqrbixko",
            torch.sub(lambda1,weights_pole1),
            torch.sub(lambda2,weights_pole2),
            torch.sub(lambda3,weights_pole3)))
        
        Hw=torch.einsum("bixko,pqrbixko->pqrbixko",
                        weights_residue,
                        term1)

        output_residue1=torch.einsum("bioxs,oxsikpqr->bkoxs", 
                                     alpha, 
                                     Hw) 
        output_residue2=torch.einsum("bioxs,oxsikpqr->bkpqr", 
                                     alpha, 
                                     -Hw) 
        return output_residue1,output_residue2
    

    def forward(
        self, 
        x: torch.Tensor, 
        output_shape: Optional[Tuple[int]] = None
    ):
        
        batchsize, channels, *mode_sizes = x.shape

        modes1, modes2, modes3 = self.n_modes
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        
        modes1, modes2, modes3 = min(modes1, D), min(modes2, H), min(modes3, W)
        
        # if self.linspace_steps is None:
        #     self.linspace_steps = x.shape[2:]
        self.linspace_steps = x.shape[2:]
        
        dt_list, shape = _compute_dt(shape=self.linspace_steps, 
                                     start_points=self.linspace_startpoints, 
                                     end_points=self.linspace_endpoints
                                     )
        tz = shape[0]
        tx = shape[1]
        ty = shape[2]
        tz = tz.to(x.device)
        tx = tx.to(x.device)
        ty = ty.to(x.device)
        # #Compute input poles and resudes by FFT
        dtz = dt_list[0] # this can be time dimension, instead of Z dimension
        dtx = dt_list[1]
        dty = dt_list[2] 
        
        alpha = torch.fft.fftn(x, dim=[-3,-2,-1])
        
        # Frequency grids
        omega1=torch.fft.fftfreq(tz.shape[0], dtz, device=alpha.device)*2*np.pi*1j   
        omega2=torch.fft.fftfreq(tx.shape[0], dtx, device=alpha.device)*2*np.pi*1j   
        omega3=torch.fft.fftfreq(ty.shape[0], dty, device=alpha.device)*2*np.pi*1j   
        
        # Slice frequencies to match the chosen modes
        omega1 = omega1[:modes1]
        omega2 = omega2[:modes2]
        omega3 = omega3[:modes3]
        
        omega1=omega1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega2=omega2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        omega3=omega3.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=omega1
        lambda2=omega2    
        lambda3=omega3
        
        # Slice alpha to only consider the selected modes
        alpha = alpha[:, :, :modes1, :modes2, :modes3]

        # Slice the combined weight tensor
        i, o, _ = self.weight.shape
        
        weights_pole1 = self.weight[:, :, :modes1].view(i, o, modes1)
        weights_pole2 = self.weight[:, :, modes1:(modes1+modes2)].view(i, o, modes2)
        weights_pole3 = self.weight[:, :, (modes1+modes2):(modes1+modes2+modes3)].view(i, o, modes3)
        weights_residue = self.weight[:, :, (modes1+modes2+modes3):(modes1+modes2+modes3+(modes1*modes2*modes3))].view(i, o, modes1, modes2, modes3)

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2 = self.output_PR(lambda1, 
                                                         lambda2, 
                                                         lambda3, 
                                                         alpha, 
                                                         weights_pole1, 
                                                         weights_pole2, 
                                                         weights_pole3, 
                                                         weights_residue
                                                         )
 
      
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifftn(output_residue1, 
                             s=(x.size(-3),
                                x.size(-2),
                                x.size(-1))
                             )
        x1 = torch.real(x1)
        
        term1=torch.einsum("bip,kz->bipz", 
                           weights_pole1, 
                           tz.type(torch.complex64).reshape(1,-1))
        term2=torch.einsum("biq,kx->biqx", 
                           weights_pole2, 
                           tx.type(torch.complex64).reshape(1,-1))
        term3=torch.einsum("bim,ky->bimy", 
                           weights_pole3, 
                           ty.type(torch.complex64).reshape(1,-1))
        term4=torch.einsum("bipz,biqx,bimy->bipqmzxy", 
                           torch.exp(term1),
                           torch.exp(term2),
                           torch.exp(term3))
        x2=torch.einsum("kbpqm,bipqmzxy->kizxy", 
                        output_residue2,
                        term4)
        x2=torch.real(x2)
        x2=x2/x.size(-1)/x.size(-2)/x.size(-3)
        
        x = x1+x2  
        
        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])
            
        if output_shape is not None:
            mode_sizes = output_shape
            
        # Ensuring the ouputshape is matched with desired, if it's specified
        if list(x.shape[2:]) != mode_sizes:
            x = self.shape_enforcer(x, output_shape=mode_sizes)
            
        return x
