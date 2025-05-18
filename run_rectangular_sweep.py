import fire

import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from data import get_dataset
from jumpstart.callbacks import  NetworkStatusPlot
from model.rectangular_experiment import RectangularExperiment


def run(depth=50,
        width=4,
        skip_last=False,
        plot_status=False,
        plot_num_points=None,
        plot_epoch_freq=None,
        lr=1e-3,
        batch_size=256,
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
        num_workers=0,
        n_samples=500,
        noise=0.1,
        epochs=4000,
        mode='full',
        tie_breaking='single',

        dataset='moons', project=None, entity=None, fast_dev_run=False, gpus='auto'):
    # Set up your default hyperparameters
    hyperparameter_defaults = dict(
        depth=depth,
        width=width,
        skip_last=skip_last,
        plot_status=plot_status,
        plot_epoch_freq=plot_epoch_freq,
        lr=lr,
        batch_size=batch_size,
        lambda_=lambda_,
        use_loss=use_loss,
        aggr=aggr,
        optimizer=optimizer,
        seed=seed,
        sign_balance=sign_balance,
        activation=activation,
        kernel_size=kernel_size,
        use_flattening=use_flattening,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate,
        n_maxpool=n_maxpool,
        maxpool_mode=maxpool_mode,
        skip_connections=skip_connections,
        init=init,
        num_workers=num_workers,
        n_samples=n_samples,
        noise=noise,
        epochs=epochs,
        mode=mode,
        tie_breaking=tie_breaking,
        dataset=dataset,
        plot_status_num_points=plot_num_points,
    )

    # Pass your defaults to wandb.init
    with wandb.init(config=hyperparameter_defaults, entity=entity, project=project):
        # Access all hyperparameter values through wandb.config
        config = wandb.config

        print(config['depth'])
        data_module, num_classes, in_channels = get_dataset(config)

        experiment = RectangularExperiment(
            depth=config['depth'], width=config['width'], lr=config['lr'], lambda_=config['lambda_'],
            use_loss=config['use_loss'], aggr=config['aggr'], optimizer=config['optimizer'], seed=config['seed'],
            mode=config['mode'], sign_balance=config['sign_balance'], tie_breaking=config['tie_breaking'],
            activation=config['activation'], kernel_size=config['kernel_size'], use_flattening=config['use_flattening'],
            use_batchnorm=config['use_batchnorm'], dropout_rate=config['dropout_rate'],
            n_maxpool=config['n_maxpool'], maxpool_mode=config['maxpool_mode'], skip_connections=config['skip_connections'],
            init=config['init'], skip_last=config['skip_last'], input_shape=(in_channels,),
            output_shape=num_classes
        )

        wandb_logger = WandbLogger()
        callbacks = [EarlyStopping(monitor='val/acc', stopping_threshold=1.0, mode='max', patience=config['epochs'])]

        if config['plot_status']:
            callbacks.append(NetworkStatusPlot(num_points=plot_num_points, epoch_freq=plot_epoch_freq))


        # ------------------------
        # 4 TRAINER
        # ------------------------
        trainer = Trainer(
            accelerator='auto', devices=gpus,
            logger=wandb_logger,
            max_epochs=config['epochs'],
            callbacks=callbacks,
            fast_dev_run=fast_dev_run
        )

        trainer.fit(experiment, data_module)
        # trainer.test(experiment, data_module)


if __name__ == '__main__':
    fire.Fire(run)
