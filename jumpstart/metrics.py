from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly
import torch
import torch.nn.functional as F
from joblib import delayed, Parallel
from torchmetrics import Metric
from tqdm import tqdm



class JRMetricManager(Metric):
    def __init__(self, debug=False, linear_flag_value=0, dead_flag_value=-1, nonlinear_flag_value=1, skip_last=False,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.skip_last = skip_last
        self.nonlinear_flag_value = nonlinear_flag_value
        self.dead_flag_value = dead_flag_value
        self.linear_flag_value = linear_flag_value
        self.debug = debug
        self.add_state("_unit_status", default=[], dist_reduce_fx=update_unit_status)
        self.add_state("_point_status", default=[], dist_reduce_fx='cat')
        self.add_state("_positive_unit_losses", default=[], dist_reduce_fx='mean')
        self.add_state("_positive_point_losses", default=[], dist_reduce_fx='cat')
        self.add_state("_negative_unit_losses", default=[], dist_reduce_fx='mean')
        self.add_state("_negative_point_losses", default=[], dist_reduce_fx='cat')
        self.names = None

    @staticmethod
    def from_plots(positive_unit_losses, negative_unit_losses, positive_point_losses, negative_point_losses,
                   unit_status, point_status):
        metric = JRMetricManager()
        metric._positive_unit_losses = positive_unit_losses
        metric._negative_unit_losses = negative_unit_losses
        metric._positive_point_losses = positive_point_losses
        metric._negative_point_losses = negative_point_losses
        metric._unit_status = unit_status
        metric._point_status = point_status
        return metric

    def reset(self) -> None:
        super().reset()

    def update(self, positive_unit_losses, negative_unit_losses, positive_point_losses, negative_point_losses):
        pos_unit_losses, neg_unit_losses = compute_loss_table(positive_unit_losses,
                                                              negative_unit_losses)
        self._positive_unit_losses.append(pos_unit_losses)
        self._negative_unit_losses.append(neg_unit_losses)

        pos_point_losses, neg_point_losses = compute_loss_table(positive_point_losses,
                                                                negative_point_losses)
        self._positive_point_losses.append(pos_point_losses)
        self._negative_point_losses.append(neg_point_losses)

        unit_status = compute_status_table(positive_unit_losses,
                                           negative_unit_losses,
                                           linear_flag_value=self.linear_flag_value,
                                           dead_flag_value=self.dead_flag_value,
                                           nonlinear_flag_value=self.nonlinear_flag_value)
        point_status = compute_status_table(positive_point_losses,
                                            negative_point_losses,
                                            linear_flag_value=self.linear_flag_value,
                                            dead_flag_value=self.dead_flag_value,
                                            nonlinear_flag_value=self.nonlinear_flag_value)

        if self.names is None:
            self.names = [f'[{i}]:{name}' for i, name in enumerate(positive_unit_losses.keys())]
        self.update_status(unit_status, point_status)

    def update_status(self, unit_status, point_status):
        self._unit_status.append(unit_status)
        self._unit_status = [dist_reduce_unit_status(self._unit_status)]
        self._point_status.append(point_status)

    @property
    def unit_status(self):
        unit_status = dist_reduce_unit_status(self._unit_status)
        return unit_status[:, :-1] if self.skip_last else unit_status

    @property
    def point_status(self):
        point_status = torch.concat(self._point_status)
        return point_status[:, :-1] if self.skip_last else point_status

    @property
    def point_network_status(self):
        point_network_status = self.point_status
        # for i, point in enumerate(point_network_status):
        #     last_dead_layer = torch.nonzero(point).max()
        #     point_network_status[i, :last_dead_layer] = -1
        dead =  point_network_status == -1
        dead = (dead +  torch.sum(dead, dim=1, keepdims=True) - torch.cumsum(dead, dim=1))
        point_network_status[dead > 0 ] = -1
        return point_network_status

    @property
    def positive_unit_losses(self):
        positive_unit_losses = torch.stack(self._positive_unit_losses).mean(axis=0)
        return positive_unit_losses[:, :-1] if self.skip_last else positive_unit_losses

    @property
    def negative_unit_losses(self):
        negative_unit_losses = torch.stack(self._negative_unit_losses).mean(axis=0)
        return negative_unit_losses[:, :-1] if self.skip_last else negative_unit_losses

    @property
    def unit_losses(self):
        return (self.positive_unit_losses + self.negative_unit_losses) / 2

    @property
    def positive_point_losses(self):
        positive_point_losses = torch.concat(self._positive_point_losses)
        return positive_point_losses[:, :-1] if self.skip_last else positive_point_losses

    @property
    def negative_point_losses(self):
        negative_point_losses = torch.concat(self._negative_point_losses)
        return negative_point_losses[:, :-1] if self.skip_last else negative_point_losses

    @property
    def point_losses(self):
        return (torch.cat(self._positive_point_losses) + torch.cat(self._negative_point_losses)) / 2

    @property
    def unit_nonlinear_ratio(self):
        unit_total = ~self.unit_status.isnan()
        unit_total = unit_total.sum()
        unit_nonlinear = (self.unit_status == 1).sum() / unit_total
        return unit_nonlinear

    @property
    def unit_linear_ratio(self):
        unit_total = ~self.unit_status.isnan()
        unit_total = unit_total.sum()
        unit_linear = (self.unit_status == 0).sum() / unit_total
        return unit_linear

    @property
    def unit_dead_ratio(self):
        unit_total = ~self.unit_status.isnan()
        unit_total = unit_total.sum()
        unit_dead = (self.unit_status == -1).sum() / unit_total
        return unit_dead

    @property
    def point_linear_ratio(self):
        point_total = self.point_status.shape[0]
        point_linear = (self.point_status == 0).all(axis=1).sum() / point_total
        return point_linear

    @property
    def point_nonlinear_ratio(self):
        point_total = self.point_status.shape[0]
        point_nonlinear = (self.point_status == 1).all(axis=1).sum() / point_total
        return point_nonlinear

    @property
    def point_dead_ratio(self):
        point_total = self.point_status.shape[0]
        point_dead = (self.point_status == -1).any(axis=1).sum() / point_total
        return point_dead

    @property
    def nonlinearity_ratio(self):
        point_total = self.point_status.numel()
        point_nonlinearity = (self.point_status == 1).sum() / point_total
        return point_nonlinearity

    @property
    def linearity_ratio(self):
        point_total = self.point_status.numel()
        point_linearity = (self.point_status == 0).sum() / point_total
        return point_linearity

    @property
    def deathness_ratio(self):
        point_total = self.point_status.numel()
        point_deathness = (self.point_status == -1).sum() / point_total
        return point_deathness

    @property
    def point_efficiency(self):
        point_network_status = self.point_network_status
        point_total = point_network_status.numel()
        point_efficiency = (point_network_status != -1).sum() / point_total
        return point_efficiency

    @property
    def unit_loss(self):
        return self.unit_losses.nan_to_num(0).mean()

    @property
    def point_loss(self):
        return self.point_losses.nan_to_num(0).mean()

    @property
    def positive_unit_loss(self):
        return self.positive_unit_losses.nan_to_num(0).mean()

    @property
    def negative_unit_loss(self):
        return self.negative_unit_losses.nan_to_num(0).mean()

    @property
    def positive_point_loss(self):
        return self.positive_point_losses.nan_to_num(0).mean()

    @property
    def negative_point_loss(self):
        return self.negative_point_losses.nan_to_num(0).mean()

    def compute(self):
        return {
            'unit_loss': self.unit_loss,
            'point_loss': self.point_loss,
            'positive_unit_loss': self.positive_unit_loss,
            'negative_unit_loss': self.negative_unit_loss,
            'positive_point_loss': self.positive_point_loss,
            'negative_point_loss': self.negative_point_loss,
            'unit_nonlinear_ratio': self.unit_nonlinear_ratio,
            'unit_linear_ratio': self.unit_linear_ratio,
            'unit_dead_ratio': self.unit_dead_ratio,
            'point_linear_ratio': self.point_linear_ratio,
            'point_nonlinear_ratio': self.point_nonlinear_ratio,
            'point_dead_ratio': self.point_dead_ratio,
            'nonlinearity_ratio': self.nonlinearity_ratio,
            'linearity_ratio': self.linearity_ratio,
            'deathness_ratio': self.deathness_ratio,
            'point_efficiency': self.point_efficiency,
        }


def pad_and_concat_table(tensors):
    max_width = max(t.shape[0] for t in tensors.values())
    padded = []
    for names, tensor in tensors.items():
        padded_tensor = F.pad(tensor.type(torch.float), (0, max_width - tensor.shape[0]),
                              mode='constant',
                              value=torch.nan)
        padded.append(padded_tensor)
    return torch.stack(padded).T.detach().cpu()


def compute_dead_linear_nonlinear_tables(positive_losses, negative_losses):
    linear = {}
    dead = {}
    nonlinear = {}
    with torch.no_grad():
        for i, ((pos_name, pos_losses), (neg_name, neg_losses)) in enumerate(
                zip(positive_losses.items(), negative_losses.items())):
            assert pos_name == neg_name
            name = f'[{i}]:{pos_name}'
            linear[name] = (neg_losses > 1).detach()
            dead[name] = (pos_losses >= 1).detach()
            nonlinear[name] = torch.logical_not(linear[name] | dead[name])

        linear = pad_and_concat_table(linear)
        dead = pad_and_concat_table(dead)
        nonlinear = pad_and_concat_table(nonlinear)

    return dead, linear, nonlinear


def compute_status_table(positive_losses, negative_losses, linear_flag_value=0, dead_flag_value=-1,
                         nonlinear_flag_value=1):
    dead, linear, nonlinear = compute_dead_linear_nonlinear_tables(positive_losses, negative_losses)
    return linear * linear_flag_value + dead * dead_flag_value + nonlinear * nonlinear_flag_value


def compute_loss_table(positive_losses, negative_losses):
    positive_table = {}
    negative_table = {}
    with torch.no_grad():
        for i, ((pos_name, pos_losses), (neg_name, neg_losses)) in enumerate(
                zip(positive_losses.items(), negative_losses.items())):
            assert pos_name == neg_name
            name = f'[{i}]:{pos_name}'
            positive_table[name] = pos_losses
            negative_table[name] = neg_losses

        positive_table = pad_and_concat_table(positive_table)
        negative_table = pad_and_concat_table(negative_table)
    return positive_table, negative_table


def dist_reduce_unit_status(unit_statuses, nonlinear_flag_value=1):
    first = unit_statuses[0]
    for other in unit_statuses[1:]:
        first[(first != other) & ~first.isnan()] = nonlinear_flag_value
    return first


def update_unit_status(unit_status, other, nonlinear_flag_value=1):
    unit_status[(unit_status != other) & ~unit_status.isnan()] = nonlinear_flag_value
    return unit_status


def update_unit_losses(unit_status, other, nonlinear_flag_value=1):
    unit_status[(unit_status != other) & ~unit_status.isnan()] = nonlinear_flag_value
    return unit_status
