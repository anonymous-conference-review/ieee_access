import math
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Callable, Union, Any, List

import lightning.pytorch as pl
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10

from .datasets import ImageNetKaggle, Imagenette, SklearnDataset, ImageNetNormal


class MoonsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=None, num_workers=4, n_samples=1000,
                 random_state=None, noise=0, shuffle=True, test_size=0.33):
        super().__init__()
        self.test_size = test_size
        self.save_hyperparameters()
        self.shuffle = shuffle
        self.noise = noise
        self.random_state = random_state
        self.n_samples = n_samples
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.input_shape = None
        self.output_shape = None

    def setup(self, stage: Optional[str] = None):
        X, y = make_moons(n_samples=self.n_samples, shuffle=self.shuffle, random_state=self.random_state,
                          noise=self.noise)

        self.train_ds, self.val_ds, self.test_ds = self.convert_numpy_dataset(X, y)

        if self.batch_size is None:
            self.batch_size = len(self.train_ds)

        self.input_shape = tuple(self.train_ds[0][0].shape)
        self.output_shape = self.train_ds[0][1].shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def convert_numpy_dataset(self, X, y):
        if self.test_size:
            X_train, X_valtest, y_train, y_valtest, = train_test_split(X, y,
                                                                       test_size=self.test_size,
                                                                       random_state=self.random_state)
            X_val, X_test, y_val, y_test, = train_test_split(X, y,
                                                             test_size=0.5,
                                                             random_state=self.random_state)
            X_train, y_train = torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)
            X_val, y_val = torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
            X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)

            trainset = TensorDataset(X_train, y_train.unsqueeze(1))
            valset = TensorDataset(X_val, y_val.unsqueeze(1))
            testset = TensorDataset(X_test, y_test.unsqueeze(1))
        else:
            X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
            trainset = TensorDataset(X, y.unsqueeze(1))
            valset = None
            testset = None

        return trainset, valset, testset


class VisionDataModule(pl.LightningDataModule):
    name: str = ""
    #: Dataset class to use
    dataset_cls: type
    #: A tuple describing the shape of the data
    dims: tuple

    def __init__(self, data_dir: Optional[str] = None, num_workers: int = 0, batch_size: int = 32, shuffle: bool = True,
                 pin_memory: bool = True, drop_last: bool = False, train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None, test_transform: Optional[Callable] = None) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            batch_size: How many samples per batch to load
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
            train_transform: transformations you can apply to train dataset
            val_transform: transformations you can apply to validation dataset
            test_transform: transformations you can apply to test dataset
        """
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else Path.cwd() / 'datasets' / self.name
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transform = train_transform if train_transform is not None else self.default_train_transform()
        self.val_transform = val_transform if val_transform is not None else self.default_eval_transform()
        self.test_transform = test_transform if test_transform is not None else self.default_eval_transform()

        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    @property
    def num_samples(self) -> int:
        num_samples = len(self.dataset_train) + len(self.dataset_val)
        num_samples += len(self.dataset_test) if self.dataset_test is not None else 0
        return num_samples

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:

        if stage == "fit" or stage is None:
            self.dataset_train = self.dataset_cls(self.data_dir, train=True, transform=self.train_transform)
            self.dataset_val = self.dataset_cls(self.data_dir, train=False, transform=self.val_transform)

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.data_dir, train=False, transform=self.test_transform)

    @abstractmethod
    def default_train_transform(self) -> Callable:
        raise NotImplementedError()

    @abstractmethod
    def default_eval_transform(self) -> Callable:
        raise NotImplementedError()

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )


class MNISTDataModule(VisionDataModule):
    name = "mnist"
    dataset_cls = MNIST
    dims = (1, 28, 28)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            num_workers: int = 0,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    @property
    def classes(self):
        """
        Return:
            10 classes (1 per digit)
        """
        return tuple(range(10))

    def default_train_transform(self) -> Callable:
        mnist_transforms = T.Compose(
            [T.ToTensor(), T.Normalize(mean=(0.5,), std=(0.5,))]
        )

        return mnist_transforms

    def default_eval_transform(self) -> Callable:
        return self.default_train_transform()


class CIFAR10DataModule(VisionDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-10

    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR10, train, val, test splits and transforms

    Transforms::

        transforms = T.Compose([
            T.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar10"
    dataset_cls = CIFAR10
    dims = (3, 32, 32)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            num_workers: int = 0,
            batch_size: int = 32,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def classes(self):
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_train_transform(self) -> Callable:
        normalize = T.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        return T.Compose([T.ToTensor(), normalize])

    def default_eval_transform(self) -> Callable:
        return self.default_train_transform()


class ImageNetDataModule(VisionDataModule):
    name = "imagenet"

    def __init__(
            self,
            data_dir,
            train_transform=None,
            val_transform=None,
            image_size: int = 224,
            num_workers: int = 0,
            batch_size: int = 128,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
    ) -> None:
        """
        Args:
            data_dir: path to the imagenet dataset file
            image_size: final image size
            num_workers: how many data workers
            batch_size: batch_size
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        data_dir = Path(data_dir)
        # Check if its normal imagenet or kaggle version
        # Kaggle version should contain those files
        expected_kaggle_files = {'LOC_train_solution.csv',
                                 'LOC_sample_submission.csv',
                                 'ILSVRC2012_val_labels.json',
                                 'LOC_val_solution.csv',
                                 'ILSVRC',
                                 'imagenet_class_index.json',
                                 'LOC_synset_mapping.txt'}
        # Normal imagenet should contain just train val and test folders
        expected_normal_files = {'train', 'val', 'ILSVRC2012_devkit_t12.tar.gz'}
        actual_files = set([f.name for f in data_dir.iterdir()])
        if expected_kaggle_files.issubset(actual_files):
            self.dataset_cls = ImageNetKaggle
        elif expected_normal_files.issubset(actual_files):
            self.dataset_cls = ImageNetNormal
        else:
            raise ValueError(f'Invalid imagenet directory. Expected to contain one of {expected_kaggle_files} or '
                             f'{expected_normal_files} but got {actual_files}')

        super().__init__(data_dir=str(data_dir),
                         train_transform=train_transform,
                         val_transform=val_transform,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         pin_memory=pin_memory,
                         drop_last=drop_last,
                         )

    @property
    def num_classes(self) -> int:
        return len(self.dataset_train.classes)

    @property
    def classes(self):
        return self.dataset_train.classes

    def default_train_transform(self):
        """
        From https://arxiv.org/pdf/2203.08120.pdf section D:

        For input preprocessing on ImageNet we perform a random crop of size 224 × 224 to each image, and
        apply a random horizontal flip. In all experiments, we applied L2 regularization only to the weights
        (and not the biases or batch normalization parameters).
        """
        transform = T.Compose([
            T.RandomResizedCrop(self.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform

    def default_eval_transform(self):
        """The standard imagenet transforms for validation.
        .. code-block:: python
            T.Compose([
                T.Resize(self.image_size + 32),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        transform = T.Compose(
            [
                T.Resize(self.image_size + 32),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return transform

    def __repr__(self):
        if self.dataset_train:
            repr_ = f'ImageNet Datamodule with {len(self.dataset_train)} train ' \
                    f'and {len(self.dataset_val)} val samples at root {self.data_dir}, ' \
                    f'{len(self.classes)} classes and train transform:\n\t ' \
                    f'{self.train_transform}\n and val\n\t {self.val_transform}'
        else:
            repr_ = f'ImageNet Datamodule at root {self.data_dir}'
        return repr_


class ImagenetteDataModule(ImageNetDataModule):
    name = "imagenette"
    dataset_cls = Imagenette

    def __repr__(self):
        if self.dataset_train:
            repr_ = f'Imagenette Datamodule with {len(self.dataset_train)} train ' \
                    f'and {len(self.dataset_val)} val samples at root {self.data_dir}, ' \
                    f'{len(self.classes)} classes and train transform:\n\t ' \
                    f'{self.train_transform}\n and val\n\t {self.val_transform}'
        else:
            repr_ = f'Imagenette Datamodule at root {self.data_dir}'
        return repr_


class SklearnDataModule(pl.LightningDataModule):
    name = "sklearn"

    def __init__(
            self,
            X,
            y,
            x_val=None,
            y_val=None,
            x_test=None,
            y_test=None,
            val_split=0.2,
            test_split=0,
            num_workers=0,
            random_state=42,
            shuffle=True,
            batch_size: int = 16,
            pin_memory=True,
            drop_last=False,
            *args,
            **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.batch_size = batch_size if batch_size is not None else len(X)
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # shuffle x and y
        X, y = sk_shuffle(X, y, random_state=random_state)

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[:hold_out_size], y[:hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = x_holdout[:test_i_start], y_holdout[:test_i_start]
            x_test_hold_out, y_test_holdout = x_holdout[test_i_start:], y_holdout[test_i_start:]
            X, y = X[hold_out_size:], y[hold_out_size:]

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None and val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None and test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def _init_datasets(
            self, X: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray,
            y_test: np.ndarray
    ) -> None:
        self.train_dataset = SklearnDataset(X, y) if X is not None else None
        self.val_dataset = SklearnDataset(x_val, y_val) if x_val is not None else None
        self.test_dataset = SklearnDataset(x_test, y_test) if x_test is not None else None

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader
