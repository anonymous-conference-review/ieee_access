import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Generate a dummy dataset.

    Example:
        >>> from pl_bolts.datasets import DummyDataset
        >>> from torch.utils.data import DataLoader
        >>> # mnist dims
        >>> ds = DummyDataset((1, 28, 28), (1, ))
        >>> dl = DataLoader(ds, batch_size=7)
        >>> # get first batch
        >>> batch = next(iter(dl))
        >>> x, y = batch
        >>> x.size()
        torch.Size([7, 1, 28, 28])
        >>> y.size()
        torch.Size([7, 1])
    """

    def __init__(self, *shapes, num_samples: int = 10000):
        """
        Args:
            *shapes: list of shapes
            num_samples: how many samples to use in this dataset
        """
        super().__init__()
        self.shapes = shapes

        if num_samples < 1:
            raise ValueError("Provide an argument greater than 0 for `num_samples`")

        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        sample = []
        for shape in self.shapes:
            spl = torch.rand(*shape)
            sample.append(spl)
        return sample


class DebugDataModule(pl.LightningDataModule):
    def __init__(self, input_shape=(1, 28, 28), target_shape=(10,), batch_size: int = 32, num_samples: int = 256):
        super().__init__()
        self.target_shape = target_shape
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.test = DummyDataset(self.input_shape, self.target_shape, num_samples=self.num_samples)
        self.train = DummyDataset(self.input_shape, self.target_shape, num_samples=self.num_samples)
        self.val = DummyDataset(self.input_shape, self.target_shape, num_samples=self.num_samples)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)




class DummyModel(pl.LightningModule):
    def __init__(self, output_dim, conv_out_channels=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=conv_out_channels, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_out_channels, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
