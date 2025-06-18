from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning import seed_everything
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, ChainedScheduler, CosineAnnealingLR, \
    ReduceLROnPlateau, CosineAnnealingWarmRestarts, CyclicLR

from jumpstart.jumpstart import JumpstartRegularization
from jumpstart.metrics import JRMetricManager
from model.lambda_schedulers import CosineAnnealingLambda
from model.resnet_conv import resnet50_conv, resnet34_conv, resnet18_conv, resnet101_conv, resnet152_conv
from model.resnet_dense import resnet50_dense, resnet34_dense, resnet18_dense, resnet101_dense, resnet152_dense

available_resnets = {(18, 'conv'): resnet18_conv,
                     (34, 'conv'): resnet34_conv,
                     (50, 'conv'): resnet50_conv,
                     (101, 'conv'): resnet101_conv,
                     (152, 'conv'): resnet152_conv,
                     (18, 'dense'): resnet18_dense,
                     (34, 'dense'): resnet34_dense,
                     (50, 'dense'): resnet50_dense,
                     (101, 'dense'): resnet101_dense,
                     (152, 'dense'): resnet152_dense
                     }


def _get_resnet_version(depth, unit_type):
    try:
        resnet_version = available_resnets.get((depth, unit_type))
    except KeyError:
        raise ValueError(f'Unknown resnet version with depth {depth},  and unit type {unit_type}')

    return resnet_version


class ResnetExperiment(pl.LightningModule):
    def __init__(self, depth, output_shape, lr, lambda_,
                 use_loss, aggr, optimizer, seed, jr_mode,
                 sign_balance, tie_breaking, activation,
                 use_batchnorm, use_skip_connections, skip_last,
                 width_multiplier=2,
                 skip_batchnorm=True, weights=None, in_channels=3,
                 unit_type='conv', dropout=0.0, weight_decay=0.0,
                 label_smoothing=0.0, lr_scheduler=None,
                 lr_warmup=0, lr_gamma=0.1, lr_milestones=None,
                 log_jr_metrics=False, lambda_scheduler=None,
                 lambda_cycle_length=None,
                 lambda_clip=None, lr_cycle_length=None, lr_cycle_mult=1,
                 lr_cycle_mode='triangular2',
                 *args:Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lambda_cycle_length = lambda_cycle_length
        self.cycle_mode = lr_cycle_mode
        self.cycle_mult = lr_cycle_mult
        self.cycle_length = lr_cycle_length
        self.lambda_clip = lambda_clip
        self.lambda_scheduler = None
        self.lambda_scheduler_name = lambda_scheduler
        self.log_jr_metrics = log_jr_metrics
        self.depth = depth
        self.width_multiplier = width_multiplier
        self.lr_scheduler = lr_scheduler
        self.lr_milestones = lr_milestones
        self.lr_gamma = lr_gamma
        self.lr_warmup = lr_warmup
        self.label_smoothing = label_smoothing
        self.weight_decay = weight_decay
        resnet_version = _get_resnet_version(depth, unit_type)

        self.use_batchnorm = use_batchnorm
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        if use_batchnorm:
            batchnorm = nn.BatchNorm2d if unit_type == 'conv' else nn.BatchNorm1d
        else:
            batchnorm = None
        if activation == 'relu':
            activation_layer = nn.ReLU
        elif activation == 'gelu':
            activation_layer = nn.GELU
        elif activation is None:
            activation_layer = None
        else:
            raise ValueError(f'Unknown activation {activation}')
        self.dropout = dropout
        if unit_type == 'conv':
            self.model = resnet_version(num_classes=output_shape,
                                        in_channels=in_channels,
                                        use_skip_connections=use_skip_connections,
                                        norm_layer=batchnorm,
                                        activation=activation_layer,
                                        weights=weights,
                                        dropout_prob=dropout,
                                        width_multiplier=width_multiplier)
        else:
            print('Weight multiplier not supported for dense unit type. Ignoring')
            self.model = resnet_version(num_classes=output_shape,
                                        in_channels=in_channels,
                                        use_skip_connections=use_skip_connections,
                                        norm_layer=batchnorm,
                                        activation=activation_layer,
                                        weights=weights,
                                        dropout_prob=dropout)

        self.lambda_ = lambda_
        self.optimizer = optimizer
        self.use_loss = use_loss
        self.in_channels = in_channels
        self.output_shape = output_shape

        self.jumpstart = JumpstartRegularization(model=self.model, sign_balance=sign_balance,
                                                 jr_mode=jr_mode, aggr=aggr, tie_breaking=tie_breaking,
                                                 skip_last=skip_last, skip_batchnorm=skip_batchnorm)

        self.lr = lr

        # Metrics
        self._setup_metrics()

        if use_loss:
            if self.output_shape == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            def criterion(outputs, targets):
                return torch.tensor(0)

            self.criterion = criterion

        self.seed = seed
        if self.seed:
            seed_everything(self.seed)

        self._setup_lambda_scheduler()

        self.save_hyperparameters()

    def forward(self, x) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer is None or self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer {self.optimizer}')
        lr_scheduler = None
        if self.lr_scheduler == 'multistep':
            lr_scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        elif self.lr_scheduler == 'cosine':
            if self.lr_warmup > 0:
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.cycle_length + self.lr_warmup)
            else:
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.cycle_length)

        elif self.lr_scheduler == 'cosine_with_restarts':
            lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.cycle_length, T_mult=self.cycle_mult,
                                                       eta_min=0.00001)
        elif self.lr_scheduler == 'cyclic':
            lr_scheduler = CyclicLR(optimizer, base_lr=0.00001, max_lr=self.lr, step_size_up=self.cycle_length,
                                    mode=self.cycle_mode, gamma=self.lr_gamma, cycle_momentum=self.optimizer != 'adam')
        elif self.lr_scheduler == 'reduce_on_plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, factor=self.lr_factor)
        elif self.lr_scheduler is None:
            pass
        else:
            raise ValueError(f'Unknown lr_scheduler {self.lr_scheduler}')

        if self.lr_warmup > 0:
            linear_warmup = LinearLR(optimizer, start_factor=0.00001, total_iters=self.lr_warmup)
            if lr_scheduler is None:
                lr_scheduler = linear_warmup
            else:
                lr_scheduler = ChainedScheduler([linear_warmup, lr_scheduler])

        if lr_scheduler:
            optimizer_config = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch',
                    "monitor": "val/loss",
                    'strict': True,
                    'name': 'lr'
                }
            }
        else:
            optimizer_config = optimizer

        return optimizer_config

    def _perform_step(self, stage, batch, batch_idx):
        data, targets = batch
        output = self(data)
        loss = self.criterion(output, targets)
        jumpstart_loss = self.jumpstart.loss
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/JR_loss', jumpstart_loss, prog_bar=True)

        if self.lambda_ > 0:
            if self.lambda_clip is not None:
                jumpstart_loss = torch.clamp(jumpstart_loss,
                                             max=self.lambda_clip,
                                             min=-self.lambda_clip)
            scaled_jumpstart_loss = self.get_lambda() * jumpstart_loss

            self.log(f'{stage}/scaled_JR_loss', scaled_jumpstart_loss)

            loss = loss + scaled_jumpstart_loss

        self._log_step_metrics(stage, output, targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self._perform_step('train', batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._perform_step('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._perform_step('test', batch, batch_idx)

    def _log_step_metrics(self, stage, output, targets):
        for metric_name, metric in self.metrics[stage].items():
            metric(output, targets.to(int))
            self.log(f'{stage}/{metric_name}', metric, prog_bar=True)

        if self.log_jr_metrics:
            self.jr_metric_managers[stage].update(self.jumpstart.positive_unit_losses,
                                                  self.jumpstart.negative_unit_losses,
                                                  self.jumpstart.positive_point_losses,
                                                  self.jumpstart.negative_point_losses)

    def _setup_metrics(self):
        self.metrics = {}
        self.jr_metric_managers = {}
        for stage in ['train', 'val', 'test']:
            self.metrics[stage] = {}
            self.metrics[stage] = self._get_normal_metrics()
            if self.log_jr_metrics:
                self.jr_metric_managers[stage] = JRMetricManager()
                setattr(self, f'{stage}_jr', self.jr_metric_managers[stage])
            for metric_name, metric in self.metrics[stage].items():
                setattr(self, f'{stage}_{metric_name}', metric)

    def _get_normal_metrics(self):
        metrics = {'acc': torchmetrics.Accuracy(task='multiclass', num_classes=self.output_shape)}
        return metrics

    def _jr_on_epoch_end_log(self, stage):
        if self.log_jr_metrics:
            value = self.jr_metric_managers[stage].compute()
            self.log_dict({f'{stage}/jr/{k}': v for k, v in value.items()})
            self.jr_metric_managers[stage].reset()

    def on_train_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('train')

    def on_validation_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('val')

    def on_test_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('test')

    def get_lambda(self):
        if self.lambda_scheduler is not None:
            lambda_ = self.lambda_scheduler.get_lambda()
            self.log('lambda', lambda_)
            return lambda_
        else:
            return self.lambda_

    def _setup_lambda_scheduler(self):
        if self.lambda_scheduler_name is not None:
            if self.lambda_scheduler_name == 'cosine':
                lambda_scheduler = CosineAnnealingLambda(base_lambda=self.lambda_,
                                                         T_max=self.lambda_cycle_length)
            else:
                raise ValueError(f'Unknown lambda scheduler {self.lambda_scheduler_name}')

            lambda_scheduler.model = self
            self.lambda_scheduler = lambda_scheduler

        else:
            self.lambda_scheduler = None
