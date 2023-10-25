
import torch

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

class SumAggregatorLoss(object):
    """General class to sum over a series of losses on the same input"""
    def __init__(self, *losses):
        self.losses = losses
    
    def __call__(self, **model_outputs):
        loss = 0
        for loss_fn in self.losses:
            loss += loss_fn(**model_outputs)
        return loss