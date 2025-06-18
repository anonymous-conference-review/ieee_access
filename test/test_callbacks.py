from unittest import TestCase
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from plotly.graph_objs import Figure

from jumpstart.callbacks import NetworkStatusPlot
from model.resnet_experiment import ResnetExperiment
from test.utils import DebugDataModule


class TestCallbacks(TestCase):

    def test_status_callback(self):
        unit_status = torch.tensor([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [np.nan, np.nan, -1],

        ]).T

        point_status = torch.tensor([
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],

        ]).T

        run = wandb.init(project='debug', mode='disabled')
        callback = NetworkStatusPlot()
        mock = MagicMock()
        mock.val_jr.unit_status = unit_status
        mock.val_jr.point_status = point_status
        mock.val_jr.names = ['layer1', 'layer2', 'layer3', 'output']
        trainer_mock = MagicMock()
        trainer_mock.current_epoch = 1
        callback.sync = MagicMock()
        callback.on_validation_epoch_end(trainer=trainer_mock, pl_module=mock)
        self.assertEqual('val', callback.sync.call_args_list[0][0][0])
        self.assertIsInstance(callback.sync.call_args_list[0][0][1], Figure)
        self.assertIsInstance(callback.sync.call_args_list[0][0][2], Figure)

    def test_callback_during_training(self):
        data_module = DebugDataModule(batch_size=128, num_samples=256)
        wandb_logger = WandbLogger(project='debug')

        experiment = ResnetExperiment(50, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True, log_jr_metrics=True)
        status_plot = NetworkStatusPlot()
        status_plot.sync = MagicMock()
        callbacks = [status_plot]
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=2,
            callbacks=callbacks
        )

        trainer.validate(experiment, data_module)
        self.assertEqual('val', status_plot.sync.call_args_list[0][0][0])
        self.assertIsInstance(status_plot.sync.call_args_list[0][0][1], Figure)
        self.assertIsInstance(status_plot.sync.call_args_list[0][0][2], Figure)

    def test_callback_show(self):
        data_module = DebugDataModule(batch_size=128, num_samples=256)
        wandb_logger = WandbLogger(project='debug')

        experiment = ResnetExperiment(50, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True, log_jr_metrics=True)
        status_plot = NetworkStatusPlot()
        status_plot.sync = MagicMock()
        callbacks = [status_plot]
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=2,
            callbacks=callbacks
        )

        trainer.validate(experiment, data_module)

        status_plot.sync.call_args_list[0][0][1].show()
        # status_plot.sync.call_args_list[0][0][2].show()

    def test_callback_during_training_num_points(self):
        num_points = 52
        data_module = DebugDataModule(batch_size=128, num_samples=256)
        wandb_logger = WandbLogger(project='debug')

        experiment = ResnetExperiment(50, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True, log_jr_metrics=True)
        status_plot = NetworkStatusPlot(num_points=num_points)
        status_plot.plot_and_sync = MagicMock()
        callbacks = [status_plot]
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=2,
            callbacks=callbacks
        )

        trainer.validate(experiment, data_module)
        self.assertEqual(num_points, status_plot.plot_and_sync.call_args_list[0].args[2].shape[0])

    def test_callback_show_freq(self):
        data_module = DebugDataModule(batch_size=128, num_samples=256)
        data_module.setup()
        wandb_logger = WandbLogger(project='debug')

        experiment = ResnetExperiment(50, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True, log_jr_metrics=True)
        status_plot = NetworkStatusPlot(epoch_freq=2)
        status_plot.sync = MagicMock()
        callbacks = [status_plot]
        trainer = Trainer(
            logger=wandb_logger,
            max_epochs=2,
            callbacks=callbacks
        )

        trainer.fit(experiment, data_module.val_dataloader())

        self.assertEqual(status_plot.sync.call_count, 2)
        # status_plot.sync.call_args_list[0][0][2].show()
