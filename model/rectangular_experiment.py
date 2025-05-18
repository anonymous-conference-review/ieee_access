from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torchmetrics
from lightning import seed_everything

from jumpstart.jumpstart import JumpstartRegularization
from jumpstart.metrics import JRMetricManager
from model.rectangular import Rectangular


class RectangularExperiment(pl.LightningModule):
    def __init__(self, depth, width, input_shape, output_shape, lr, lambda_,
                 use_loss, aggr, optimizer, seed, mode,
                 sign_balance, tie_breaking, activation, kernel_size, use_flattening,
                 use_batchnorm, dropout_rate, n_maxpool, maxpool_mode, skip_connections,
                 init, skip_last, skip_batchnorm=True, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.model = Rectangular(depth=depth, width=width, input_shape=input_shape,
                                 output_shape=output_shape,
                                 activation=activation, kernel_size=kernel_size,
                                 use_flattening=use_flattening, use_batchnorm=use_batchnorm,
                                 dropout_rate=dropout_rate,
                                 n_maxpool=n_maxpool, maxpool_mode=maxpool_mode,
                                 skip_connections=skip_connections,
                                 init=init)
        self.lambda_ = lambda_
        self.optimizer = optimizer
        self.use_loss = use_loss
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.jumpstart = JumpstartRegularization(model=self.model, sign_balance=sign_balance,
                                                 jr_mode=mode, aggr=aggr, tie_breaking=tie_breaking,
                                                 skip_last=skip_last, skip_batchnorm=skip_batchnorm)

        self.lr = lr

        # Metrics
        self._setup_metrics()

        if use_loss:
            if self.output_shape == 1:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            def criterion(outputs, targets):
                return torch.tensor(0)

            self.criterion = criterion

        self.seed = seed
        if self.seed:
            seed_everything(self.seed)

        self.save_hyperparameters()

    def forward(self, x) -> Any:
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer is None or self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Unknown optimizer {self.optimizer}')

        return optimizer

    def _perform_step(self, stage, batch, batch_idx):
        data, targets = batch
        output = self(data)
        loss = self.criterion(output, targets)
        jumpstart_loss = self.jumpstart.loss
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/JR_loss', jumpstart_loss, prog_bar=True)

        if self.lambda_ > 0:
            scaled_jumpstart_loss = self.lambda_ * jumpstart_loss
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
        for metric_name, metric in self.metrics[stage]['output'].items():
            metric(output, targets.to(int))
            self.log(f'{stage}/{metric_name}', metric, prog_bar=True)

        for metric_name, metric in self.metrics[stage]['jr'].items():
            metric.update(self.jumpstart.positive_unit_losses, self.jumpstart.negative_unit_losses,
                          self.jumpstart.positive_point_losses, self.jumpstart.negative_point_losses)

    def _setup_metrics(self):
        self.metrics = {'train': self._get_metrics(),
                        'val': self._get_metrics(),
                        'test': self._get_metrics()
                        }
        for stage, stage_metrics in self.metrics.items():
            for metric_input, metrics_input in stage_metrics.items():
                for metric_name, metric in metrics_input.items():
                    setattr(self, f'{stage}_{metric_name}', metric)

    def _get_metrics(self):
        task = 'binary' if self.output_shape == 1 else 'multiclass'
        output_metrics = {'acc': torchmetrics.Accuracy(task=task, num_classes=self.output_shape),
                          }
        jr_metrics = {'jr': JRMetricManager()}
        metrics = {'output': output_metrics, 'jr': jr_metrics}
        return metrics

    def _jr_on_epoch_end_log(self, stage):
        for metric_name, metric in self.metrics[stage]['jr'].items():
            metric.update(self.jumpstart.positive_unit_losses, self.jumpstart.negative_unit_losses,
                          self.jumpstart.positive_point_losses, self.jumpstart.negative_point_losses)
            value = metric.compute()
            self.log_dict({f'{stage}/jr/{k}': v for k, v in value.items()})

    def on_train_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('train')

    def on_validation_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('val')

    def on_test_epoch_end(self) -> None:
        self._jr_on_epoch_end_log('test')
