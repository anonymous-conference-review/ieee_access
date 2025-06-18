import json
import os
import subprocess
import urllib.request
from typing import Tuple, Any

from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import default_loader, ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, extract_archive

import numpy as np


class ImageNetNormal(ImageNet):
    def __init__(self, root: str, train=True, **kwargs: Any):
        split = 'train' if train else 'val'
        if 'download' in kwargs:
            del kwargs['download']
        super().__init__(root, split=split, **kwargs)


class ImageNetKaggle(Dataset):
    def __init__(self, root, train=True, transform=None, download=True, loader=default_loader):
        self.root = root
        self.loader = loader
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        if download:
            self.download(root)
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        self.classes = list(self.syn_to_class.values())
        self.synsets = list(self.syn_to_class.keys())
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        split = 'train' if train else 'val'
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if train:
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            else:
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def download(self, root):
        if os.path.exists(root):
            return
        os.makedirs(root, exist_ok=True)
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "imagenet-object-localization-challenge", "-p", root])
        filepath = os.path.join(root, "imagenet-object-localization-challenge.zip")
        extract_archive(filepath, root)
        # Clean up zip file
        os.remove(filepath)
        # Download imagenet_class_index.json
        url = 'https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json'
        filename = os.path.join(root, 'imagenet_class_index.json')
        urllib.request.urlretrieve(url, filename)

        # Download ILSVRC2012_val_labels.json
        url = 'https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json'
        filename = os.path.join(root, 'ILSVRC2012_val_labels.json')
        urllib.request.urlretrieve(url, filename)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.loader(self.samples[idx])
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]

    def __repr__(self):
        return f'ImageNetKaggle Dataset with {len(self)} samples at root {self.root}, ' \
               f'{len(self.classes)} classes and transform:\n\t {self.transform}'


class Imagenette(ImageFolder):
    """
    `Imagenette <https://github.com/fastai/imagenette>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``imagenette2`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from the training set,
            otherwise creates from the validation set. Default: True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        if download:
            self.download(root)

        split_dir = 'train' if train else 'val'
        super().__init__(os.path.join(root, 'imagenette2-320', split_dir), transform=transform,
                         target_transform=target_transform)

    def download(self, root):
        if os.path.exists(root):
            return f'{root} exists skipping download'
        download_and_extract_archive(self.url, root)


class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets.


    """

    def __init__(self, X: np.ndarray, y: np.ndarray, X_transform: Any = None, y_transform: Any = None) -> None:
        """
        Args:
            X: Numpy ndarray
            y: Numpy ndarray
            X_transform: Any transform that works with Numpy arrays
            y_transform: Any transform that works with Numpy arrays
        """
        super().__init__()
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]

        # Do not convert integer to float for classification data
        if not ((y.dtype == np.int32) or (y.dtype == np.int64)):
            y = y.astype(np.float32)

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y
