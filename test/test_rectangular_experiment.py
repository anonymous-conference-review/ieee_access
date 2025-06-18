from unittest import TestCase

import torch
from lightning import Trainer
from lightning_fabric import seed_everything
from data import SklearnDataModule
from sklearn.datasets import load_breast_cancer

from model.rectangular_experiment import RectangularExperiment
from run_rectangular_sweep import run


class Test(TestCase):
    def setUp(self) -> None:
        self.hyperparameter_defaults = dict(
            gpus=1,
            plot_status=False,
            lr=1e-3,
            lambda_=0.01,
            batch_size=32,
            use_loss=True,
            aggr='balanced',
            optimizer=None,
            seed=42,
            sign_balance=0.5,
            tie_breaking='single',
            activation='relu',
            kernel_size=3,
            use_flattening=False,
            use_batchnorm=False,
            dropout_rate=0,
            n_maxpool=None,
            maxpool_mode=None,
            skip_connections=None,
            init='kaiming',
            num_workers=0,
            n_samples=1000,
            noise=0.1,
            epochs=4000,
            entity='jumpstart',
            project='linear-regions',
            fast_dev_run=True
        )

    def test_wine(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'wine'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_moons(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'moons'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_abalone(self):
        config = self.hyperparameter_defaults
        config['depth'] = 50
        config['width'] = 4
        config['dataset'] = 'abalone'

        config['project'] = 'debug'
        run(**config)

    def test_load_credit_approval(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'credit_approval'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_breast_cancer(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'breast_cancer'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_heart_disease(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'heart_disease'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_cylinder_bands(self):

        config = self.hyperparameter_defaults
        config['dataset'] = 'cylinder_bands'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_dermatology(self):

        config = self.hyperparameter_defaults
        config['dataset'] = 'dermatology'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_diabetic(self):
    
        config = self.hyperparameter_defaults
        config['dataset'] = 'diabetic'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_iris(self):
        

        config = self.hyperparameter_defaults
        config['dataset'] = 'iris'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_dishonest_users(self):
      

        config = self.hyperparameter_defaults
        config['dataset'] = 'dishonest_users'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_early_stage_diabetes_risk(self):
    
        config = self.hyperparameter_defaults
        config['dataset'] = 'early_stage_diabetes_risk'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_fertility(self):

        config = self.hyperparameter_defaults
        config['dataset'] = 'fertility'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_haberman(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'haberman'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_hayes_roth(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'hayes_roth'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_hcv(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'hcv'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_iris_2(self):
        config = self.hyperparameter_defaults
        config['dataset'] = 'iris'
        config['depth'] = 50
        config['width'] = 4

        config['project'] = 'debug'
        run(**config)

    def test_rectangular_gelu_mnist(self):
        seed_everything(42)
        X, y = load_breast_cancer(return_X_y=True)
        data_module = SklearnDataModule(X, y, batch_size=128,
                                        num_workers=0)
        num_classes = 2
        in_channels = X.shape[1]

        experiment = RectangularExperiment(
            depth=3, width=4,
            skip_last=False,
            lr=1e-3,
            lambda_=0,
            use_loss=True,
            aggr='balanced',
            optimizer=None,
            seed=42,
            sign_balance=0.5,
            activation='relu',
            kernel_size=3,
            use_flattening=False,
            use_batchnorm=False,
            dropout_rate=0,
            n_maxpool=None,
            maxpool_mode=None,
            skip_connections=None,
            init='kaiming',
            mode='full',
            tie_breaking='single',
            input_shape=(in_channels,),
            output_shape=num_classes,
        )

        gpus = 1 if torch.cuda.is_available() else None
        if gpus:
            trainer = Trainer(
                devices=gpus,
                max_epochs=100
            )
        else:
            trainer = Trainer(
                max_epochs=100
            )

        trainer.fit(experiment, data_module)
        self.assertGreater(trainer.logged_metrics['train/acc'], 0.8)
