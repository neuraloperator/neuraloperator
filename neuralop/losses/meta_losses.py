'''
meta_losses.py contains losses that compose multiple other losses.
'''

import torch

from .loss import Loss

class FieldwiseAggregatorLoss(Loss):
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

    def forward(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):
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

    def __init__(self, losses, weights=None, return_individual=True, compute_grads=False):
        super().__init__()
        if weights is None:
            weights = [1.0 / len(losses)] * len(losses)
        if not len(weights) == len(losses):
            raise ValueError("Each loss must have a weight.")
        #self.losses = list(zip(losses, weights))
        self.losses = {x.__name__: (x,y) for x,y in zip(losses,weights)}
        self.compute_grads = compute_grads

        self.return_individual = return_individual

    def __call__(self, *args, **kwargs):
        weighted_loss = 0.0
        wrapper = {}
        for name, (loss,weight) in self.losses.items():
            loss_value = loss(*args, **kwargs)
            if self.return_individual:
                wrapper[name] = weight * loss_value
            else:
                weighted_loss += weight * loss_value
        if self.return_individual:
            return wrapper
        else:
            return weighted_loss

    def __str__(self):
        description = "Combined loss: "
        for name, (loss, weight) in self.losses.items():
            description += f"{name} (weight: {weight}) "
        return description