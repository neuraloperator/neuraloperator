import numpy as np
import torch
from .data_losses import H1Loss, LpLoss

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

class ShakeShakeLoss(object):
    def __init__(self, losses, weights=None, 
                 weight_update_start=30,
                 weight_update_type='linear',
                 weight_update_slope=100):
        """
        Performs shake-shake regularization on a sum 

        Parameters
        ----------
        losses: List[loss]
            list of loss objects, all must implement __call__
        weights: torch.Tensor, optional
            optional weights to multiply with losses before shake-shake is applied
        """
        self.losses = losses
        if weights:
            self.weights = weights
        else:
            self.weights = torch.tensor([1.] * len(self.losses))
        
        self.weight_update_start = weight_update_start
        self.weight_update_slope = weight_update_slope
        self.weight_update_type = weight_update_type
    
    def sample_from_simplex(self, d: int):
        '''
        Draw a sample uniformly from the d-dim probability simplex as in [1]_.

        References
        -----------
        _[1]. Rubin, The Bayesian bootstrap. Ann. Statist. 9, 1981, 130-134.
        '''

        return torch.distributions.dirichlet.Dirichlet(torch.tensor([1.] * d)).sample()
    
    def __call__(self, *args, **kwargs):
        """ take a random interpolation of all losses in self.losses
            from the probability simplex"""
        weighted_loss = 0.
        shake_shake_weights = self.sample_from_simplex(len(self.losses))

        for i, loss in enumerate(self.losses):
            weighted_loss += shake_shake_weights[i] * loss(*args, **kwargs) * self.weights[i]
        
        return weighted_loss
    
    def update_weights(self, epoch, threshold=0.001):
        '''
        Update shake-shake relative weighting 
        '''

        if (not self.weight_update_type) or epoch < self.weight_update_start:
            return

        assert self.weight_update_type == 'linear'
        losses = []
        updated_weights = []
        # update weights
        for loss, weight in self.losses:
            losses.append(loss)
            if isinstance(loss, LpLoss) or isinstance(loss, H1Loss):
                new_weight = weight - (weight / self.weight_update_slope)
                if new_weight < threshold:
                    new_weight = 0.
                    print('loss no longer used:', loss)
                updated_weights.append(new_weight)
            else:
                updated_weights.append(weight)

        # renormalize weights
        updated_weights = np.array(updated_weights) / np.sum(updated_weights)
        print('weights are', updated_weights)
        self.losses = list(zip(losses, updated_weights))