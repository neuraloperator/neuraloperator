import torch
import logging
from torch import nn
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)

class FieldwiseAggregatorLoss(object):
    """
    AggregatorLoss takes a dict of losses, keyed to correspond 
        to different properties or fields of a model's output.
        It then returns an aggregate of all losses weighted by
        an optional weight dict.

    params:
        losses: dict[Loss]
            a dictionary of loss functions, each of which
            takes in some truth_field and pred_field
        mappings: dict[tuple(Slice)]
            a dictionary of mapping indices corresponding to 
            the output fields above. keyed 'field': indices, 
            so that pred[indices] contains output for specified field
        logging: bool
            whether to track error for each output field of the model separately 

    """ 
    def __init__(self, losses: dict, mappings: dict, logging=False):
        # AggregatorLoss should only be instantiated with more than one loss.
        assert mappings.keys() == losses.keys(), 'Mappings and losses must use the same keying'

        self.losses = losses
        self.mappings = mappings
        self.logging = logging

    def __call__(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):
        """
        Calculate aggregate loss across model inputs and outputs.

        parameters
        ----------
        pred: tensor
            contains predictions output by a model, indexed for various output fields
        y: tensor
            contains ground truth. Indexed the same way as pred.     
        **kwargs: dict
            bonus args to pass to each fieldwise loss
        """

        loss = 0.
        if self.logging: 
            loss_record = {}
        # sum losses over output fields
        for field, indices in self.mappings.items():
            pred_field = pred[indices].view(-1,1)
            truth_field = truth[indices]
            field_loss = self.losses[field](pred_field, truth_field, **kwargs)
            loss += field_loss
            if self.logging: 
                loss_record['field'] = field_loss
        loss = (1.0/len(self.mappings))*loss

        if self.logging:
            return loss, field_loss
        else:
            return loss


class WeightedSumLoss(object):
    """
    Computes an average or weighted sum of given losses.
    """

    def __init__(self, losses, weights=None):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if not len(weights) == len(losses):
            raise ValueError("Each loss must have a weight.")
        self.losses = list(zip(losses, weights))

    def __call__(self, *args, **kwargs):
        weighted_loss = 0.0
        for loss, weight in self.losses:
            weighted_loss += weight * loss(*args, **kwargs)
        return weighted_loss

    def __str__(self):
        description = "Combined loss: "
        for loss, weight in self.losses:
            description += f"{loss} (weight: {weight}) "
        return description


# Below are the implementations of the meta-losses from NVIDIA PhysicsNeMo (with almost no changes).
# These are more convenient to use when training without the Trainer/PINOTrainer class.
# NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering
# https://github.com/NVIDIA/physicsnemo


class Aggregator(nn.Module):
    """
    Base class for loss aggregators that dynamically balance multiple loss components.
    
    This class provides the foundation for adaptive loss weighting schemes that can
    automatically adjust the importance of different loss terms during training.
    
    Parameters
    ----------
    params : List[torch.Tensor]
        List of model parameters used to determine the device for tensor operations.
        All parameters should be on the same device.
    num_losses : int
        Number of loss components that will be aggregated. This determines the
        size of internal buffers for tracking loss history.
    weights : Optional[Dict[str, float]], optional
        Optional static weights for each loss component. If provided, these weights
        will be applied to the corresponding losses before aggregation.
        If a loss key is not present in the weights dict, a default weight of 1.0
        will be used, by default None

    Notes
    -----
    This is an abstract base class that should be inherited by specific loss
    aggregation algorithms like SoftAdapt or ReLoBRaLo. The forward method
    should be implemented by subclasses to define the specific aggregation strategy.
    
    The device is automatically determined from the first parameter in the params list.
    All parameters should be on the same device to avoid device mismatch errors.
    """

    def __init__(self, params, num_losses, weights):
        super().__init__()
        self.params: List[torch.Tensor] = list(params)
        self.num_losses: int = num_losses
        self.weights: Optional[Dict[str, float]] = weights
        self.device: torch.device
        self.device = list(set(p.device for p in self.params))[0]
        self.init_loss: torch.Tensor = torch.tensor(0.0, device=self.device)

        def weigh_losses_initialize(
            weights: Optional[Dict[str, float]]
        ) -> Callable[
            [Dict[str, torch.Tensor], Optional[Dict[str, float]]],
            Dict[str, torch.Tensor],
        ]:
            if weights is None:

                def weigh_losses(
                    losses: Dict[str, torch.Tensor], weights: None
                ) -> Dict[str, torch.Tensor]:
                    return losses

            else:

                def weigh_losses(
                    losses: Dict[str, torch.Tensor], weights: Dict[str, float]
                ) -> Dict[str, torch.Tensor]:
                    for key in losses.keys():
                        if key not in weights.keys():
                            weights.update({key: 1.0})
                    losses = {key: weights[key] * losses[key] for key in losses.keys()}
                    return losses

            return weigh_losses

        self.weigh_losses = weigh_losses_initialize(self.weights)



class SoftAdapt(Aggregator):
    """
    SoftAdapt algorithm for adaptive loss weighting and aggregation.
    
    SoftAdapt automatically adjusts the weights of multiple loss components based on
    their relative magnitudes and rates of change. The algorithm uses exponential
    weighting to give higher importance to losses that are not decreasing or are 
    relatively larger, based on their ratio to previous losses, helping to balance 
    the training of different objectives.
    
    Reference: "Heydari, A.A., Thompson, C.A. and Mehmood, A., 2019.
    Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions.
    arXiv preprint arXiv: 1912.12355."
    
    Parameters
    ----------
    params : List[torch.Tensor]
        List of model parameters used to determine the device for tensor operations.
    num_losses : int
        Number of loss components that will be aggregated.
    eps : float, optional
        Small constant added to denominators to prevent division by zero.
        Should be positive and small (e.g., 1e-8), by default 1e-8
    weights : Optional[Dict[str, float]], optional
        Optional static weights for each loss component, by default None
        
    Notes
    -----
    At step 0, all losses are simply summed with equal weights. For subsequent steps,
    the algorithm computes adaptive weights based on the ratio of current to previous
    loss values, with higher weights given to losses that are increasing or have
    higher relative magnitudes.
    
    Warnings
    --------
    The algorithm assumes that all loss components should be minimized. If some
    losses should be maximized, they should be negated before passing to SoftAdapt.
    """

    def __init__(self, params, num_losses, eps=1e-8, weights=None):
        super().__init__(params, num_losses, weights)
        self.eps: float = eps
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the original variant of the SoftAdapt algorithm.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of loss components to be aggregated. 
            For instance, the keys could be 'data_loss' and 'physics_loss'.
        step : int
            Current optimization step. Used to determine whether to initialize
            the algorithm (step 0) or apply adaptive weighting (step > 0).

        Returns
        -------
        loss : torch.Tensor
            The aggregated loss value.
        lmbda : torch.Tensor
            The computed adaptive weights for each loss component.
            
        Notes
        -----
        At step 0, all losses are summed with equal weights.
        For subsequent steps, the algorithm computes adaptive weights based on
        the ratio of current to previous loss values.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Aggregate losses by summation at step 0
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
            lmbda = torch.ones_like(self.prev_losses)

        # Aggregate losses using SoftAdapt for step > 0
        else:
            lmbda: torch.Tensor = torch.ones_like(self.prev_losses)
            lmbda_sum: torch.Tensor = torch.zeros_like(self.init_loss)
            losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
            normalizer: torch.Tensor = (losses_stacked / self.prev_losses).max()
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    lmbda[i] = torch.exp(
                        losses[key] / (self.prev_losses[i] + self.eps) - normalizer
                    )
                    lmbda_sum += lmbda[i]
                loss += lmbda[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
            loss *= self.num_losses / (lmbda_sum + self.eps)
            
        return loss, lmbda


class Relobralo(Aggregator):
    """
    Relative Loss Balancing with Random Lookback (ReLoBRaLo) algorithm for adaptive loss weighting.
    
    ReLoBRaLo combines relative loss balancing with random lookback to provide more
    stable and effective multi-objective optimization. The algorithm uses both
    initial loss values and previous loss values to compute adaptive weights,
    with a random lookback mechanism. This random lookback mechanism introduces 
    stochasticity in the weighting by sometimes reverting to initial loss values, 
    encouraging robustness across loss scales over time.
    
    Reference: "Bischof, R. and Kraus, M., 2021.
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
    arXiv preprint arXiv:2110.09813."
    
    Parameters
    ----------
    params : List[torch.Tensor]
        List of model parameters used to determine the device for tensor operations.
    num_losses : int
        Number of loss components that will be aggregated.
    alpha : float, optional
        Controls the smoothing factor for the exponential moving average of the adaptive weights.
        High alpha gives more inertia to past weights (more stability).
        Low alpha allows faster adaptation.
        By default 0.95
    beta : float, optional
        Probability for the random lookback mechanism, i.e. probability of sampling from 
        initial loss values instead of previous loss values during adaptive weight computation.
        By default 0.99
    tau : float, optional
        Temperature parameter that controls the sharpness of the exponential
        weighting. Higher values make the weights more uniform, by default 1.0
    eps : float, optional
        Small constant added to denominators to prevent division by zero.
        Should be positive and small (e.g., 1e-8), by default 1e-8.
    weights : Optional[Dict[str, float]], optional
        Optional static weights for each loss component, by default None

    Notes
    -----
    At step 0, all losses are simply summed with equal weights, and both initial
    and previous loss buffers are initialized with the current loss values.
    
    The random lookback mechanism (controlled by beta) helps the algorithm escape
    local minima by occasionally using initial loss values instead of recent ones.
    This is particularly useful in physics-informed neural networks where different
    loss components may have different convergence characteristics.
    
    The temperature parameter tau controls how aggressive the weighting is. Lower
    values make the algorithm more selective in which losses to prioritize,
    while higher values make the weights more uniform.
    
    Warnings
    --------
    The algorithm assumes that all loss components should be minimized. If some
    losses should be maximized, they should be negated before passing to ReLoBRaLo.
    
    The random component means that results may vary between runs, 
    due to the Bernoulli sampling in the weight computation.
    """

    def __init__(
        self, params, num_losses, alpha=0.95, beta=0.99, tau=1.0, eps=1e-8, weights=None
    ):
        super().__init__(params, num_losses, weights)
        self.alpha: float = alpha
        self.beta: float = beta
        self.tau: float = tau
        self.eps: float = eps
        self.register_buffer(
            "init_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )
        self.register_buffer(
            "lmbda_ema", torch.ones(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the ReLoBRaLo algorithm.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of loss components to be aggregated. 
            For instance, the keys could be 'data_loss' and 'physics_loss'.
        step : int
            Current optimization step. Used to determine whether to initialize
            the algorithm (step 0) or apply adaptive weighting (step > 0).

        Returns
        -------
        loss : torch.Tensor
            The aggregated loss value.
        lmbda_ema : torch.Tensor
            The exponential moving average of adaptive weights for each loss component.
            
        Notes
        -----
        At step 0, all losses are summed with equal weights, and both initial
        and previous loss buffers are initialized with the current loss values.
        
        For subsequent steps, the algorithm computes weights using both initial
        and previous loss values, with a random component that helps escape
        local minima. The final weights are computed as an exponential moving average.
        """

        # weigh losses
        losses = self.weigh_losses(losses, self.weights)

        # Initialize loss
        loss: torch.Tensor = torch.zeros_like(self.init_loss)

        # Aggregate losses by summation at step 0
        if step == 0:
            for i, key in enumerate(losses.keys()):
                loss += losses[key]
                self.init_losses[i] = losses[key].clone().detach()
                self.prev_losses[i] = losses[key].clone().detach()

        # Aggregate losses using ReLoBRaLo for step > 0
        else:
            losses_stacked: torch.Tensor = torch.stack(list(losses.values()))
            normalizer_prev: torch.Tensor = (
                losses_stacked / (self.tau * self.prev_losses)
            ).max()
            normalizer_init: torch.Tensor = (
                losses_stacked / (self.tau * self.init_losses)
            ).max()
            
            # define a Bernoulli-sampled variable that determines whether to use initial 
            # or previous loss values in computing the adaptive weights
            rho: torch.Tensor = torch.bernoulli(torch.tensor(self.beta))
            
            with torch.no_grad():
                lmbda_prev: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.prev_losses + self.eps)
                    - normalizer_prev
                )
                lmbda_init: torch.Tensor = torch.exp(
                    losses_stacked / (self.tau * self.init_losses + self.eps)
                    - normalizer_init
                )
                lmbda_prev *= self.num_losses / (lmbda_prev.sum() + self.eps)
                lmbda_init *= self.num_losses / (lmbda_init.sum() + self.eps)

            # Compute the exponential moving average of weights and aggregate losses
            for i, key in enumerate(losses.keys()):
                with torch.no_grad():
                    self.lmbda_ema[i] = self.alpha * (
                        rho * self.lmbda_ema[i].clone() + (1.0 - rho) * lmbda_init[i]
                    )
                    self.lmbda_ema[i] += (1.0 - self.alpha) * lmbda_prev[i]
                loss += self.lmbda_ema[i].clone() * losses[key]
                self.prev_losses[i] = losses[key].clone().detach()
                
        return loss, self.lmbda_ema