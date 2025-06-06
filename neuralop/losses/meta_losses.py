import torch
import logging
import inspect
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
        # AggregatorLoss should only be instantiated 
        # with more than one loss.
        assert mappings.keys() == losses.keys(), 'Mappings \
               and losses must use the same keying'

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


class Relobralo_for_Trainer(FieldwiseAggregatorLoss):
    """
    Implements the Relative Loss Balancing with Random Lookback (ReLoBRaLo) algorithm.
    
    Inherits from FieldwiseAggregatorLoss to maintain compatibility with the Trainer class.
    
    References
    ----------
    Bischoff et al., "Multi-objective loss balancing for physics-informed deep learning" (2021)
    
    Adapted from NVIDIA PhysicsNeMo
    NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering
    https://github.com/NVIDIA/physicsnemo
    
    
    Parameters
    ----------
    losses : dict
        Dictionary of loss functions to aggregate
    mappings : dict
        Dictionary mapping loss names to indices in the output tensor
    alpha : float, default=0.95
        Smoothing factor for exponential moving average
    beta : float, default=0.99
        Probability for random lookback
    tau : float, default=1.0
        Temperature parameter for softmax normalization
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    def __init__(self, losses, mappings, alpha=0.95, beta=0.99, tau=1.0, epsilon=1e-8):
        super().__init__(losses=losses, mappings=mappings, logging=True)
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.epsilon = epsilon
        self.weights = {field: 1.0 for field in losses}
        self.latest_loss_record = {}
        self.latest_weight_record = {}
        self.latest_total_loss = None
        self.init_losses = {field: None for field in losses}
        self.prev_losses = {field: None for field in losses}
        self.lmbda_ema = {field: 1.0 for field in losses}

    def __call__(self, pred: torch.Tensor, y: torch.Tensor, **kwargs):
        total_loss = 0.0
        self.latest_loss_record = {}
        self.latest_weight_record = {}
        
        # Compute current losses
        current_losses = {}
        for field, indices in self.mappings.items():
            pred_field = pred[indices]
            truth_field = y[indices]
            loss_fn = self.losses[field]
            sig = inspect.signature(loss_fn.__call__)
            
            # Prepare arguments for loss function
            call_args = {}
            if 'pred' in sig.parameters or 'y_pred' in sig.parameters:
                call_args['pred' if 'pred' in sig.parameters else 'y_pred'] = pred_field
            if 'y' in sig.parameters or 'truth' in sig.parameters:
                call_args['y' if 'y' in sig.parameters else 'truth'] = truth_field
            call_args.update(kwargs)
            
            # Compute loss
            field_loss = loss_fn(**call_args)
            current_losses[field] = field_loss.item()
            self.latest_loss_record[field] = field_loss

        # Initialize losses at first step
        if all(self.init_losses[field] is None for field in self.mappings):
            for field in self.mappings:
                self.init_losses[field] = current_losses[field]
                self.prev_losses[field] = current_losses[field]
                self.weights[field] = 1.0 / len(self.mappings)
                self.latest_weight_record[field] = self.weights[field]
        else:
            # Stack losses for vectorized computation
            losses_stacked = torch.tensor([current_losses[field] for field in self.mappings])
            prev_losses_stacked = torch.tensor([self.prev_losses[field] for field in self.mappings])
            init_losses_stacked = torch.tensor([self.init_losses[field] for field in self.mappings])
            
            # Compute weights using ReLoBRaLo formula
            with torch.no_grad():
                # Compute normalizers
                normalizer_prev = (losses_stacked / (self.tau * prev_losses_stacked)).max()
                normalizer_init = (losses_stacked / (self.tau * init_losses_stacked)).max()
                
                # Random lookback
                rho = torch.bernoulli(torch.tensor(self.beta))
                
                # Compute and normalize weights
                lmbda_prev = torch.exp(losses_stacked / (self.tau * prev_losses_stacked + self.epsilon) - normalizer_prev)
                lmbda_init = torch.exp(losses_stacked / (self.tau * init_losses_stacked + self.epsilon) - normalizer_init)
                lmbda_prev = lmbda_prev / (lmbda_prev.sum() + self.epsilon)
                lmbda_init = lmbda_init / (lmbda_init.sum() + self.epsilon)
                
                # Update weights using exponential moving average
                for i, field in enumerate(self.mappings):
                    self.lmbda_ema[field] = self.alpha * (rho * self.lmbda_ema[field] + (1.0 - rho) * lmbda_init[i].item())
                    self.lmbda_ema[field] += (1.0 - self.alpha) * lmbda_prev[i].item()
                    self.weights[field] = self.lmbda_ema[field]
                    self.latest_weight_record[field] = self.weights[field]
            
            # Update previous losses
            for field in self.mappings:
                self.prev_losses[field] = current_losses[field]

        # Compute weighted sum
        for field, indices in self.mappings.items():
            total_loss += self.weights[field] * self.latest_loss_record[field]

        # Average the total loss by number of fields
        total_loss = total_loss / len(self.mappings)
        self.latest_total_loss = total_loss
        return total_loss


class SoftAdapt_for_Trainer(FieldwiseAggregatorLoss):
    """
    SoftAdapt loss aggregator that implements the SoftAdapt algorithm for loss weighting

    Inherits from FieldwiseAggregatorLoss to maintain compatibility with the Trainer class.
        
    References
    ----------
    Heydari et al.,  "SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks 
    with Multi-Part Loss Functions" (2019)
    
    Adapted from NVIDIA PhysicsNeMo
    NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering
    https://github.com/NVIDIA/physicsnemo
    
    
    Parameters
    ----------
    losses : dict
        Dictionary of loss functions to aggregate
    mappings : dict
        Dictionary mapping loss names to indices in the output tensor
    epsilon : float, default=1e-8
        Small constant for numerical stability
    """
    def __init__(self, losses, mappings, epsilon=1e-8):
        super().__init__(losses=losses, mappings=mappings, logging=True)
        self.epsilon = epsilon
        self.weights = {field: 1.0 for field in losses}
        self.latest_loss_record = {}
        self.latest_weight_record = {}
        self.latest_total_loss = None
        self.prev_losses = {field: None for field in losses}

    def __call__(self, pred: torch.Tensor, y: torch.Tensor, **kwargs):
        total_loss = 0.0
        self.latest_loss_record = {}
        self.latest_weight_record = {}
        
        # Compute current losses
        current_losses = {}
        for field, indices in self.mappings.items():
            pred_field = pred[indices]
            truth_field = y[indices]
            loss_fn = self.losses[field]
            sig = inspect.signature(loss_fn.__call__)
            
            # Prepare arguments for loss function
            call_args = {}
            if 'pred' in sig.parameters or 'y_pred' in sig.parameters:
                call_args['pred' if 'pred' in sig.parameters else 'y_pred'] = pred_field
            if 'y' in sig.parameters or 'truth' in sig.parameters:
                call_args['y' if 'y' in sig.parameters else 'truth'] = truth_field
            call_args.update(kwargs)
            
            # Compute loss
            field_loss = loss_fn(**call_args)
            current_losses[field] = field_loss.item()

            # Store for logging
            self.latest_loss_record[field] = field_loss

        # Update weights using SoftAdapt formula if we have previous losses
        if all(self.prev_losses[field] is not None for field in self.mappings):
            # Stack losses for vectorized computation
            losses_stacked = torch.tensor([current_losses[field] for field in self.mappings])
            prev_losses_stacked = torch.tensor([self.prev_losses[field] for field in self.mappings])
            
            # Compute weights using softmax formula
            with torch.no_grad():
                # Compute normalizer as max of loss ratios
                normalizer = (losses_stacked / (prev_losses_stacked + self.epsilon)).max()
                
                # Compute and normalize weights
                weights = torch.exp(losses_stacked / (prev_losses_stacked + self.epsilon) - normalizer)
                weights = weights / (weights.sum() + self.epsilon)
                
                # Update weights dictionary
                for i, field in enumerate(self.mappings):
                    self.weights[field] = weights[i].item()
                    self.latest_weight_record[field] = self.weights[field]

        # Update previous losses for next iteration
        for field in self.mappings:
            self.prev_losses[field] = current_losses[field]

        # Compute weighted sum
        for field, indices in self.mappings.items():
            total_loss += self.weights[field] * self.latest_loss_record[field]

        # Average the total loss by number of fields
        total_loss = total_loss / len(self.mappings)
        self.latest_total_loss = total_loss
        return total_loss






# Below are the implementations of the meta-losses from NVIDIA PhysicsNeMo (with almost no changes).
# These are more convenient to use when training without the Trainer/PINOTrainer class.
# NVIDIA PhysicsNeMo: An open-source framework for physics-based deep learning in science and engineering
# https://github.com/NVIDIA/physicsnemo


class Aggregator(nn.Module):
    """
    Base class for loss aggregators
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
    SoftAdapt for loss aggregation
    Reference: "Heydari, A.A., Thompson, C.A. and Mehmood, A., 2019.
    Softadapt: Techniques for adaptive loss weighting of neural networks with multi-part loss functions.
    arXiv preprint arXiv: 1912.12355."
    """

    def __init__(self, params, num_losses, eps=1e-8, weights=None):
        super().__init__(params, num_losses, weights)
        self.eps: float = eps
        self.register_buffer(
            "prev_losses", torch.zeros(self.num_losses, device=self.device)
        )

    def forward(self, losses: Dict[str, torch.Tensor], step: int) -> torch.Tensor:
        """
        Weights and aggregates the losses using the original variant of the softadapt algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
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
    Relative loss balancing with random lookback
    Reference: "Bischof, R. and Kraus, M., 2021.
    Multi-Objective Loss Balancing for Physics-Informed Deep Learning.
    arXiv preprint arXiv:2110.09813."
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
        Weights and aggregates the losses using the ReLoBRaLo algorithm

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            A dictionary of losses.
        step : int
            Optimizer step.

        Returns
        -------
        loss : torch.Tensor
            Aggregated loss.
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
