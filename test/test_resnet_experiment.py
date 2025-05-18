import tempfile
import tracemalloc
from pathlib import Path
from unittest import TestCase

import pandas as pd
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, ChainedScheduler

from data import ImageNetDataModule, CIFAR10DataModule, MNISTDataModule, SklearnDataModule
from jumpstart.plotting import plot_lr_schedule
from model.resnet_experiment import ResnetExperiment
from run_resnet_experiment_sweep import run
from test.utils import DebugDataModule
import matplotlib.pyplot as plt


class Test(TestCase):

    def test_resnet_ablation_mnist(self):
        data_module = MNISTDataModule(batch_size=128, normalize=True)

        experiment = ResnetExperiment(50, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True)
        trainer = Trainer(
            fast_dev_run=True
        )

        trainer.fit(experiment, data_module)

    def test_resnet_gelu_mnist(self):
        data_module = MNISTDataModule(batch_size=128, normalize=True)

        experiment = ResnetExperiment(18, in_channels=1, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='gelu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True)
        gpus = 1 if torch.cuda.is_available() else None
        if gpus:
            trainer = Trainer(
                devices=gpus,
                max_steps=30
            )
        else:
            trainer = Trainer(
                max_steps=30
            )

        trainer.fit(experiment, data_module)
        self.assertGreater(trainer.logged_metrics['train/acc'], 0.8)

    def test_resnet_ablation_cifar(self):
        data_module = CIFAR10DataModule(batch_size=128)

        experiment = ResnetExperiment(50, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True)
        trainer = Trainer(
            fast_dev_run=True
        )

        trainer.fit(experiment, data_module)

    def test_resnet_dense(self):
        num_features = 100
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(50, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features
                                      )
        trainer = Trainer(
            fast_dev_run=True
        )

        trainer.fit(experiment, data_module)

    def test_resnet_wine(self):
        from sklearn.datasets import load_wine

        X, y = load_wine(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=32)
        num_features = X.shape[1]
        experiment = ResnetExperiment(50, output_shape=3, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features
                                      )
        trainer = Trainer(
            fast_dev_run=True
        )

        trainer.fit(experiment, data_module)

    def test_resnet_iris(self):
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=32)
        num_features = X.shape[1]
        experiment = ResnetExperiment(50, output_shape=3, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features
                                      )
        trainer = Trainer(
            fast_dev_run=True
        )

        trainer.fit(experiment, data_module)

    def test_resnet_ablation_sweep_fast(self):
        gpus = 1 if torch.cuda.is_available() else 'auto'
        run(epochs=2, project='debug', entity='blauigris', fast_dev_run=True, gpus=gpus,
            lambda_=0.1, use_batchnorm=False, use_skip_connections=False)

    def test_resnet_point_status_num_points(self):
        gpus = 1 if torch.cuda.is_available() else 'auto'
        run(epochs=2, project='debug', entity='blauigris', fast_dev_run=True, gpus=gpus, plot_status=True,
            plot_num_points=50, log_jr_metrics=True, lambda_=0.1, use_batchnorm=False, use_skip_connections=False)


    def test_resnet_ablation_imagenet(self):
        data_path = Path('~/data/imagenet/imagenet_class_index.json').expanduser()
        if data_path.exists():
            data_module = ImageNetDataModule(data_path, batch_size=128)

            experiment = ResnetExperiment(50, output_shape=1000, lr=0.0001,
                                          use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                          sign_balance=0.5, tie_breaking='single', activation='relu',
                                          use_batchnorm=True, lambda_=0, aggr='balanced',
                                          use_skip_connections=True, skip_last=True)
            trainer = Trainer(
                fast_dev_run=True
            )

            trainer.fit(experiment, data_module)
        else:
            print('ImageNet data not found, skipping test')


    def test_resnet_label_smoothing(self):
        label_smoothing = 0.1
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3,
                                 unit_type='conv', dropout=0.0, weight_decay=0.0,
                                 label_smoothing=label_smoothing, )

        assert model.criterion.label_smoothing == label_smoothing

    def test_resnet_wd(self):
        weight_decay = 0.1
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3,
                                 unit_type='conv', dropout=0.0, weight_decay=weight_decay, )
        optimizer = model.configure_optimizers()
        assert optimizer.param_groups[0]['weight_decay'] == weight_decay

    def test_resnet_dropout(self):
        dropout = 0.1
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3,
                                 unit_type='conv', dropout=dropout)
        assert model.dropout == dropout
        assert model.model.dropout_prob == dropout
        assert model.model.dropout.p == dropout

    def test_resnet_width_multiplier(self):
        width_multiplier = 1.5
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3,
                                 unit_type='conv', width_multiplier=width_multiplier)

        expected_widths = {'layer1': 64, 'layer2': 96, 'layer3': 144, 'layer4': 216}
        for name, layer in model.model.named_children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                if 'layer' in name:
                    expected_width = expected_widths[name.split('.')[0]]
                    actual_width = layer.weight.shape[0]
                    self.assertEqual(expected_width, actual_width)

    def test_setup_metrics_normal(self):
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3,
                                 unit_type='conv')
        actual = {}
        for stage, metrics in model.metrics.items():
            actual[stage] = {name: metric.__class__.__name__ for name, metric in metrics.items()}
        expected = {'test': {'acc': 'MulticlassAccuracy'}, 'train': {'acc': 'MulticlassAccuracy'},
                    'val': {'acc': 'MulticlassAccuracy'}}
        self.assertEqual(model.metrics, expected)
        for stage, metrics in model.metrics.items():
            for name, metric in metrics.items():
                self.assertTrue(hasattr(model, f'{stage}_{name}'))

    def test_setup_metrics_normal_jr(self):
        model = ResnetExperiment(depth=50, output_shape=1000, lr=0.1, lambda_=0.0001,
                                 use_loss=True, aggr='balanced', optimizer='adam', seed=None, jr_mode='full',
                                 sign_balance=0.5, tie_breaking='single', activation='relu',
                                 use_batchnorm=False, use_skip_connections=False, skip_last=True,
                                 skip_batchnorm=True, weights=None, in_channels=3, log_jr_metrics=True,
                                 unit_type='conv')
        actual = {k: v.__class__.__name__ for k, v in model.jr_metric_managers.items()}
        expected = {'test': 'JRMetricManager', 'train': 'JRMetricManager', 'val': 'JRMetricManager'}
        self.assertEqual(actual, expected)
        for stage, metrics in model.jr_metric_managers.items():
            self.assertTrue(hasattr(model, f'{stage}_jr'))






    def test_cosine_scheduler(self):
        num_features = 100
        max_epochs = 20
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True, lambda_scheduler='cosine',
                                      lr_scheduler='cosine', lr_cycle_length=5
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lr.plot()
            plt.show()
            assert lr.value_counts().shape[0] > 10

    def test_cosine_restarts_scheduler(self):
        num_features = 10
        max_epochs = 50
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True, lambda_scheduler='cosine',
                                      lr_scheduler='cosine_with_restarts',
                                      lr_cycle_length=5, lr_cycle_mult=2
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lr.plot()
            plt.show()
            assert lr.value_counts().shape[0] > 1

    def test_cycle_triangle_scheduler(self):
        num_features = 10
        max_epochs = 50
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True, lambda_scheduler='cosine',
                                      lr_scheduler='cyclic',
                                      lr_cycle_length=5, lr_cycle_mult=2
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lr.plot()
            plt.show()
            assert lr.value_counts().shape[0] > 1

    def test_cycle_triangle_2_scheduler(self):
        num_features = 10
        max_epochs = 50
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True, lambda_scheduler='cosine',
                                      lr_scheduler='cyclic',
                                      lr_cycle_length=5, lr_cycle_mult=2, lr_cycle_mode='triangular2'
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lr.plot()
            plt.show()
            assert lr.value_counts().shape[0] > 1

    def test_cycle_exponential_scheduler(self):
        num_features = 10
        max_epochs = 50
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.0001,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True, lambda_scheduler='cosine',
                                      lr_scheduler='cyclic',
                                      lr_cycle_length=5, lr_cycle_mult=2, lr_cycle_mode='exp_range',
                                      lr_gamma=0.9
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lr.plot()
            plt.show()
            assert lr.value_counts().shape[0] > 1


    def test_lambda_cosine_scheduler(self):
        num_features = 100
        max_epochs = 20
        data_module = DebugDataModule(input_shape=(num_features,), batch_size=128, num_samples=256)

        experiment = ResnetExperiment(18, output_shape=10, lr=0.1,
                                      use_loss=True, optimizer='adam', seed=42, jr_mode='full',
                                      sign_balance=0.5, tie_breaking='single', activation='relu',
                                      use_batchnorm=True, lambda_=0.1, aggr='balanced',
                                      use_skip_connections=True, skip_last=True,
                                      unit_type='dense', in_channels=num_features,
                                      log_jr_metrics=True,
                                      lambda_scheduler='cosine', lambda_cycle_length=max_epochs // 3,
                                      lr_scheduler='cosine', lr_cycle_length=max_epochs // 3,
                                      )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # Create tmp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            trainer = Trainer(
                max_epochs=max_epochs,
                logger=CSVLogger(tmpdirname, name='test', version='0'),
                callbacks=[lr_monitor]
            )
            trainer.fit(experiment, data_module)

            # load csv
            csv_path = Path(tmpdirname) / 'test' / '0' / 'metrics.csv'
            df = pd.read_csv(csv_path)
            lr = df['lr'].dropna().reset_index(drop=True)
            lambda_ = df['lambda'].dropna().reset_index(drop=True)
            a = pd.concat([lr, lambda_], axis=1)
            a.plot()
            plt.show()
            pd.testing.assert_series_equal(lr, lambda_, check_dtype=False, check_names=False)


