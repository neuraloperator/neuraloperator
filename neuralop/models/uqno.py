from functools import partialmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses import PointwiseQuantileLoss
from .base_model import BaseModel
import numpy as np

class UQNO():
    """a (alpha,delta) Risk-Controlling Neural Operator based on https://arxiv.org/abs/2402.01960

    Parameters
    ----------
    alpha : float
        target domain coverage rate is 1-alpha
    delta : float
        target 1-delta of functions satisfy ">=1-alpha points in band"        
    base_model : BaseModel
        base model for point prediction
    residual_model_config: Dict
        configs for the residual model, can be same as base model
    """

    def __init__(
        self,
        alpha,
        delta,
        base_model,
        base_model_encoder,
        residual_model_config
    ):
        self.alpha = alpha
        self.delta = delta
        self.base_model = base_model
        self.base_model_encoder = base_model_encoder
        self.residual_model_config = residual_model_config
        self.scale_factor = 0
        self.quantile_model = None
        self.quantile_model_encoder = None


    def _train_quantile_model(self, train_loader, encoder, val_loader, save_path, device="cuda"):
        model_new = TFNO(n_modes=config["n_modes"], hidden_channels=config["hidden_channels"], projection_channels=config["projection_channels"], factorization="tucker", rank=0.42)
        model_new = model_new.to(device)
        model_new.train()
        optimizer_quantile = torch.optim.Adam(model_new.parameters(),
                                    lr=config["learning_rate"], 
                                    weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_quantile, config["T_max"])
        quantile_loss = PointwiseQuantileLoss(quantile=config["quantile"])
        train_loss = quantile_loss
        eval_losses={'quantile loss': quantile_loss}
        trainer = Trainer(model_new, n_epochs=config["epochs"],
                    device=device,
                    mg_patching_levels=0,
                    wandb_log=True,
                    log_test_interval=5,
                    use_distributed=False,
                    verbose=True)
        trainer.train_pointwise_err(train_loader, {"val":val_loader},
                encoder,
                model_new, 
                optimizer_quantile,
                scheduler, 
                regularizer=False, 
                quantile=config["quantile"],
                training_loss=train_loss,
                eval_losses=eval_losses)
        torch.save(model_new.state_dict(), save_path)
        return model_new

    def get_residual(self, model, encoder, loader, device="cuda"):
        error_list = []
        x_list = []
        model = model.to(device)
        model.eval()
        encoder = encoder.to(device)
        with torch.no_grad():
            for idx, sample in enumerate(loader):
                x, y = sample['x'], sample['y']
                x = x.to(device)
                y = y.to(device)
                pred_unscaled = model(x)
                pred = encoder.inverse_transform(pred_unscaled)
                error = (pred-y).detach().to("cpu") # detach, otherwise residual carries gradient of model weight
                error_list.append(error)
                x_list.append(x.to("cpu"))
                del x,y, pred
        errors = torch.cat(error_list, axis=0)
        xs = torch.cat(x_list, axis=0)
        return xs, errors

    def _get_coeff_quantile_idx(self, delta, n_samples, n_gridpts, alpha):
        """
        get the index of (ranked) sigma's for given delta and t
        we take the min alpha for given delta
        delta is percentage of functions that satisfy alpha threshold in domain
        alpha is percentage of points in ball on domain
        return 2 idxs
        domain_idx is the k for which kth (ranked descending by ptwise |err|/quantile_model_pred_err)
        value we take per function
        func_idx is the j for which jth (ranked descending) value we take among n_sample functions
        Note: there is a min alpha we can take based on number of gridpoints, n and delta, we specify lower bounds lb1 and lb2
        t needs to be between the lower bound and alpha
        """
        lb = np.sqrt(-np.log(delta)/2/n_gridpts)
        t = (alpha-lb)/3+lb # if t too small, will make the in-domain estimate conservative
        # too large will make the across-function estimate conservative. so we find a moderate t value
        print(f"we set alpha (on domain): {alpha}, t={t}")
        percentile = alpha-t
        domain_idx = int(np.ceil(percentile*n_gridpts))
        print(f"domain index: {domain_idx}'th largest of {n_gridpts}")

        # get function idx
        function_percentile= np.ceil((n_samples+1)*(delta-np.exp(-2*n_gridpts*t*t)))/n_samples
        function_idx = int(np.ceil(function_percentile*n_samples))
        print(f"function index: {function_idx}'th lagrest of {n_samples}")
        return domain_idx, function_idx

    def _calibrate_quantile_model(self, model, model_encoder, calib_loader, domain_idx, function_idx, device="cuda"):
        val_ratio_list = []
        model = model.to(device)
        model_encoder = model_encoder.to(device)
        with torch.no_grad():
            for idx, sample in enumerate(calib_loader):
                x, y = sample['x'].to(device), sample['y'].to(device)
                pred = model_encoder.inverse_transform(model(x))#.squeeze()
                ratio = torch.abs(y)/pred
                val_ratio_list.append(ratio.squeeze().to("cpu"))
                del x,y, pred
        val_ratios = torch.cat(val_ratio_list, axis=0)
        val_ratios_pointwise_quantile = torch.topk(val_ratios.view(val_ratios.shape[0], -1),domain_idx+1, dim=1).values[:,-1]
        scale_factor = torch.topk(val_ratios_pointwise_quantile, function_idx+1, dim=0).values[-1]
        print(f"scale factor: {scale_factor}")
        return scale_factor

    def train_calibrated_uqno(self, train_residual_loader, calib_residual_loader, residual_encoder, n_calib_samples, discretization, use_pretrained_quantile_model=False):
        domain_idx, func_idx = self._get_coeff_quantile_idx(self.delta, n_calib_samples, discretization, self.alpha)
        if not use_pretrained_quantile_model:
            quantile_model = self._train_quantile_model(self.residual_model_config, train_residual_loader, residual_encoder, calib_residual_loader)
            self.quantile_model = quantile_model
        scale_factor = self._calibrate_quantile_model(self.quantile_model, residual_encoder, calib_residual_loader, domain_idx, func_idx)
        self.scale_factor = scale_factor
        self.quantile_model_encoder = residual_encoder
        return self.quantile_model, scale_factor

    def predict_with_uncertainty(self, test_loader, device="cuda"):
        point_preds = []
        uq_preds = []
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                x, y = sample['x'], sample['y']
                x = x.to(device)
                y = y.to(device)
                point_pred = self.base_model_encoder.inverse_transform(self.base_model(x))
                quantile_model_pred = self.quantile_model_encoder.inverse_transform(self.quantile_model(x))
                uncertainty_pred = quantile_model_pred * self.scale_factor
                point_preds.append(point_pred)
                uq_preds.append(uncertainty_pred)

        point_pred = torch.cat(point_preds, axis=0)
        uq_pred = torch.cat(uq_preds, axis=0)
        return point_pred, uq_pred

    def eval_coverage_bandwidth(self, test_loader, device="cuda"):
        """
        Get percentage of instances hitting target-percentage pointwise coverage
        (e.g. percenetage of instances with >1-alpha points being covered by quantile model)
        as well as avg band length
        """
        in_pred_list = []
        avg_interval_list = []
        base_model = self.base_model.to(device)
        base_encoder = self.base_model_encoder.to(device)
        quantile_model = self.quantile_model.to(device)
        quantile_encoder = self.quantile_model_encoder.to(device)

        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                x, y = sample['x'], sample['y']
                x = x.to(device)
                y = y.to(device)
                point_pred = self.base_model_encoder.inverse_transform(self.base_model(x))
                pointwise_true_err = point_pred - y
                quantile_model_pred = self.quantile_model_encoder.inverse_transform(self.quantile_model(x))
                uncertainty_pred = quantile_model_pred * self.scale_factor

                in_pred = (torch.abs(pointwise_true_err) < uncertainty_pred).float().squeeze()
                avg_interval = torch.abs(uncertainty_pred.squeeze()).view(uncertainty_pred.shape[0],-1).mean(dim=1)
                avg_interval_list.append(avg_interval.to("cpu"))

                in_pred_flattened = in_pred.view(in_pred.shape[0], -1)
                in_pred_instancewise = torch.mean(in_pred_flattened,dim=1) >= 1-self.alpha # expected shape (batchsize, 1)
                in_pred_list.append(in_pred_instancewise.float().to("cpu"))
                #del x, y, pred, point_pred, in_pred_flattened
                #torch.cuda.empty_cache()

        in_pred = torch.cat(in_pred_list, axis=0)
        intervals = torch.cat(avg_interval_list, axis=0)
        mean_interval = torch.mean(intervals, dim=0)
        in_pred_percentage = torch.mean(in_pred, dim=0)
        print(f"{in_pred_percentage} of instances satisfy that >= {1-self.alpha} pts drawn are inside the predicted quantile")
        print(f"Mean interval width is {mean_interval}")
        return mean_interval, in_pred_percentage


            

        