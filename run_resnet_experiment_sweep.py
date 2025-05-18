import os
from pathlib import Path
from typing import Union, List

import fire
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from typeguard import typechecked

import wandb
from data import ImageNetDataModule, MNISTDataModule, CIFAR10DataModule
from jumpstart.callbacks import NetworkStatusPlot
from model.resnet_experiment import ResnetExperiment


@typechecked
def run(lambda_: float,
        use_batchnorm: bool,
        use_skip_connections: bool,
        *,
        depth: int = 50,
        width_multiplier: float = 2,
        activation: str = 'relu',
        skip_last: bool = True,
        plot_status: bool = True,
        plot_num_points: int = 200,
        plot_status_epoch_freq: Union[int, None] = 100,
        lr: float = 1e-3,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        lr_scheduler: Union[str, None] = None,
        lr_warmup: int = 0,
        lr_gamma: float = 0.0,
        lr_milestones: Union[List, None] = None,
        batch_size: int = 128,
        use_loss: bool = True,
        aggr: str = 'balanced',
        optimizer: Union[str, None] = None,
        seed: Union[int, None] = 42,
        sign_balance: float = 0.5,
        num_workers: Union[None, int] = 16,
        epochs: int = -1,
        patience: Union[int, None] = None,
        jr_mode: str = 'full',
        unit_type: str = 'conv',
        tie_breaking: str = 'single',
        dataset: str = 'mnist',
        log_jr_metrics: bool = True,
        project: Union[str, None] = None,
        entity: Union[str, None] = None,
        name: Union[str, None] = None,
        fast_dev_run: bool = False,
        gpus: Union[str, int, List] = 'auto',
        precision=32,
        float32_matmul_precision: str = 'highest',
        use_compile: bool = False,
        gradient_clip_val: Union[float, None] = None,
        gradient_clip_algorithm: str = 'norm',
        lambda_clip: Union[None, float] = None,
        save_checkpoint: bool = True):
    # Set up your default hyperparameters
    hyperparameter_defaults = dict(
        depth=depth,
        use_batchnorm=use_batchnorm,
        width_multiplier=width_multiplier,
        use_skip_connections=use_skip_connections,
        activation=activation,
        skip_last=skip_last,
        lr=lr,
        dropout=dropout, weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        lr_scheduler=lr_scheduler,
        lr_warmup=lr_warmup, lr_gamma=lr_gamma, lr_milestones=lr_milestones,
        batch_size=batch_size,
        lambda_=lambda_,
        use_loss=use_loss,
        aggr=aggr,
        optimizer=optimizer,
        seed=seed,
        sign_balance=sign_balance,
        epochs=epochs,
        patience=patience,
        jr_mode=jr_mode,
        tie_breaking=tie_breaking,
        dataset=dataset,
        unit_type=unit_type,
        precision=precision,
        float32_matmul_precision=float32_matmul_precision,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        lambda_clip=lambda_clip,
    )

    # Pass your defaults to wandb.init
    with wandb.init(config=hyperparameter_defaults, entity=entity, project=project, name=name):
        # Access all hyperparameter values through wandb.config
        config = wandb.config
        print('Using config:')
        for k, v in config.items():
            print(f'\t{k}: {v}')

        print('Using defaults:')
        for k, v in hyperparameter_defaults.items():
            print(f'\t{k}: {v}')

        in_channels = 3
        data_dir = None
        if config['dataset'] == 'mnist':
            data_module = MNISTDataModule(batch_size=config['batch_size'], seed=config['seed'],
                                          num_workers=num_workers)
            in_channels = 1
            num_classes = 10
            data_module.prepare_data()
            data_module.setup()
        elif config['dataset'] == 'cifar10':
            data_module = CIFAR10DataModule(batch_size=config['batch_size'], seed=config['seed'],
                                            num_workers=num_workers)
            num_classes = 10
            data_module.prepare_data()
            data_module.setup()
        elif config['dataset'] == 'imagenet':
            data_dir = Path(os.environ.get('DATA_DIR', '~/data')).expanduser() / 'imagenet'
            data_module = ImageNetDataModule(data_dir=data_dir, batch_size=config['batch_size'],
                                             num_workers=num_workers)
            data_module.prepare_data()
            data_module.setup()
            num_classes = data_module.num_classes

        else:
            raise ValueError(f'Unknown dataset {config["dataset"]}')

        experiment = ResnetExperiment(
            depth=config['depth'], width_multiplier=config['width_multiplier'], use_batchnorm=config['use_batchnorm'],
            use_skip_connections=config['use_skip_connections'], activation=config['activation'],
            in_channels=in_channels, lr=config['lr'], lambda_=config['lambda_'],
            use_loss=config['use_loss'], aggr=config['aggr'], optimizer=config['optimizer'], seed=config['seed'],
            jr_mode=config['jr_mode'], sign_balance=config['sign_balance'], tie_breaking=config['tie_breaking'],
            skip_last=config['skip_last'], output_shape=num_classes, unit_type=unit_type,
            dropout=config['dropout'], weight_decay=config['weight_decay'],
            label_smoothing=config['label_smoothing'], lr_scheduler=config['lr_scheduler'],
            lr_warmup=config['lr_warmup'], lr_gamma=config['lr_gamma'], lr_milestones=config['lr_milestones'],
            log_jr_metrics=log_jr_metrics, lambda_clip=config['lambda_clip']

        )

        torch.set_float32_matmul_precision(config['float32_matmul_precision'])
        if use_compile:
            try:
                experiment = torch.compile(experiment)
            except RuntimeError as ex:
                if 'Python 3.11+ not yet supported for torch.compile' in str(ex):
                    print('Python 3.11+ not yet supported for torch.compile')
                else:
                    raise ex

        wandb_logger = WandbLogger(log_model=True)

        # define a metric we are interested in the minimum of
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("val/loss", summary="min")
        # define a metric we are interested in the maximum of
        for stage, metrics in experiment.metrics.items():
            for metric in metrics:
                wandb.define_metric(f'{stage}/{metric}', summary="max")

        callbacks = []
        if config['patience'] is not None:
            stopper = EarlyStopping(monitor='val/acc', stopping_threshold=1.0, mode='max', patience=config['patience'])
            callbacks.append(stopper)


        if plot_status:
            if not log_jr_metrics:
                raise ValueError('plot_status requires log_jr_metrics to be True')
            callbacks.append(NetworkStatusPlot(num_points=plot_num_points, epoch_freq=plot_status_epoch_freq))

        if save_checkpoint:
            if not data_dir:
                data_dir = Path(os.environ.get('DATA_DIR', '~/data')).expanduser()
            dir_path = data_dir / name / 'checkpoints' if name is not None else data_dir / 'checkpoints'
            callbacks.append(ModelCheckpoint(monitor='val/acc', mode='max', auto_insert_metric_name=True,
                                             filename='best_model', save_last=True, save_top_k=1,
                                             dirpath=dir_path))


        # ------------------------
        # 4 TRAINER
        # ------------------------
        trainer = Trainer(
            accelerator='auto', devices=gpus,
            logger=wandb_logger,
            max_epochs=config['epochs'],
            callbacks=callbacks,
            fast_dev_run=fast_dev_run,
            precision=config['precision'],
            gradient_clip_val=config['gradient_clip_val'],
            gradient_clip_algorithm=config['gradient_clip_algorithm']

        )

        trainer.fit(experiment, data_module)
        if hasattr(data_module, 'test_dataset'):
            trainer.test(experiment, data_module)


if __name__ == '__main__':
    fire.Fire(run)
