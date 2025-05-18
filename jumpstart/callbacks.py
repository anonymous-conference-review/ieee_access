import lightning.pytorch as pl
import torch
import wandb
from lightning import Callback

from jumpstart.plotting import plot_unit_status_plotly, plot_point_status_plotly


class NetworkStatusPlot(Callback):
    def __init__(self, num_points=5000, epoch_freq=None):
        self.epoch_freq = epoch_freq
        self.num_points = num_points

    def plot_and_sync(self, stage, unit_status, point_status, labels, epoch):
        unit_fig = plot_unit_status_plotly(unit_status, labels)
        point_fig = plot_point_status_plotly(point_status, labels)
        self.sync(stage, unit_fig, point_fig, epoch)

    def sync(self, stage, unit_fig, point_fig, epoch):
        wandb.log({f"{stage}/status/unit": unit_fig, 'epoch': epoch})
        wandb.log({f"{stage}/status/point": point_fig, 'epoch': epoch})

    def _plot_status(self, stage, metric, current_epoch):
        if self._should_run(current_epoch):
            with torch.no_grad():
                unit_status, point_status = metric.unit_status, metric.point_status
                if self.num_points and self.num_points < metric.point_status.shape[0]:
                    selected = torch.randint(0, metric.point_status.shape[0], (self.num_points,))
                    point_status = metric.point_status[selected]

                point_status = point_status.detach().cpu().numpy()
                unit_status = unit_status.detach().cpu().numpy()
                self.plot_and_sync(stage, unit_status, point_status, metric.names, current_epoch)

    def _jr_on_epoch_end_log(self, stage, pl_module, metric):
        with torch.no_grad():
            # value = metric.compute()
            # pl_module.log_dict({f'{stage}/jr/{k}': v for k, v in value.items()})
            self._plot_status(stage, metric, pl_module.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, 'train_jr'):
            self._jr_on_epoch_end_log('train', pl_module, pl_module.train_jr)
        else:
            raise RuntimeError('No train JR metric enabled')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, 'val_jr'):
            self._jr_on_epoch_end_log('val', pl_module, pl_module.val_jr)
        else:
            raise RuntimeError('No val JR metric enabled')

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, 'test_jr'):
            self._jr_on_epoch_end_log('test', pl_module, pl_module.test_jr)
        else:
            raise RuntimeError('No test JR metric enabled')

    def _should_run(self, current_epoch):
        is_run = bool(wandb.run)
        if self.epoch_freq is not None:
            is_freq = (current_epoch + 1) % self.epoch_freq == 0 or current_epoch == 0
        else:
            is_freq = True
        return is_run and is_freq
