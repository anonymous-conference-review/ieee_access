import os
from pathlib import Path
from unittest import TestCase

import PIL.Image
import lightning.pytorch as pl
import torch
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as T
from torchvision import transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm

from data import ImageNetKaggle, MNISTDataModule, CIFAR10DataModule, ImageNetDataModule, ImagenetteDataModule, \
    Imagenette, \
    get_dataset
from test.utils import DummyModel


DATA_DIR = Path(os.environ.get('DATA_DIR', '~/data')).expanduser()
IMAGENET_PATH = DATA_DIR / 'imagenet'


class TestImageNetKaggle(TestCase):
    def test_train(self):
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        dataset = ImageNetKaggle(IMAGENET_PATH, train=True, transform=transform)
        print(dataset)
        self.assertEqual(len(dataset.classes), 1000)
        self.assertEqual(len(dataset), 1281167)

    def test_val(self):
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        dataset = ImageNetKaggle(IMAGENET_PATH, train=False, transform=transform)
        print(dataset)
        self.assertEqual(len(dataset.classes), 1000)
        self.assertEqual(len(dataset), 50000)

    def test_accuracy_pretrained_resnet(self):
        model = torchvision.models.resnet50(weights="DEFAULT")
        model.eval().cuda()  # Needs CUDA, don't bother on CPUs

        datamodule = ImageNetDataModule(IMAGENET_PATH, batch_size=1024)
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        correct = 0
        total = 0
        n_batches = 10
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(train_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.76)
                    break

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.7)
                    break


class TestImagenet(TestCase):
    def test_train(self):
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        dataset = ImageNet(IMAGENET_PATH, train=True, transform=transform)
        print(dataset)
        self.assertEqual(len(dataset.classes), 1000)
        self.assertEqual(len(dataset), 1281167)

    def test_val(self):
        transform = T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        dataset = ImageNet(IMAGENET_PATH, train=False, transform=transform)
        print(dataset)
        self.assertEqual(len(dataset.classes), 1000)
        self.assertEqual(len(dataset), 50000)

    def test_accuracy_pretrained_resnet(self):
        model = torchvision.models.resnet50(weights="DEFAULT")
        model.eval().cuda()  # Needs CUDA, don't bother on CPUs

        datamodule = ImageNetDataModule(IMAGENET_PATH, batch_size=1024)
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        correct = 0
        total = 0
        n_batches = 10
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(train_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.76)
                    break

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.7)
                    break


class TestImagenette(TestCase):
    def setUp(self):
        self.root = Path('./') / 'imagenette'
        self.train_dataset = Imagenette(root=self.root, train=True, transform=None, download=True)
        self.val_dataset = Imagenette(root=self.root, train=False, transform=None, download=True)

    # def tearDown(self):
    #     shutil.rmtree(self.root)

    def test_download(self):
        # Check that the dataset has been downloaded and extracted to the correct directory
        self.assertTrue(os.path.exists(os.path.join(self.root, 'imagenette2-320')))
        self.assertTrue(os.path.exists(os.path.join(self.root, 'imagenette2-320', 'train')))
        self.assertTrue(os.path.exists(os.path.join(self.root, 'imagenette2-320', 'val')))

    def test_len(self):
        # Check that the length of the training and validation datasets are correct
        self.assertEqual(len(self.train_dataset), 9469)
        self.assertEqual(len(self.val_dataset), 3925)

    def test_classes(self):
        # Check that the classes are the same for both the training and validation datasets
        self.assertEqual(self.train_dataset.classes, self.val_dataset.classes)

    def test_class_to_idx(self):
        # Check that the class-to-index mapping is the same for both the training and validation datasets
        self.assertEqual(self.train_dataset.class_to_idx, self.val_dataset.class_to_idx)

    def test_getitem(self):
        # Check that __getitem__ returns the correct image and label
        img, label = self.train_dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertEqual(label, 0)

    def test_transform(self):
        # Check that the transforms are correctly applied to the images
        transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        train_dataset = Imagenette(root=self.root, train=True, transform=transform, download=True)
        img, _ = train_dataset[0]
        img_flip = transform(train_dataset.loader(train_dataset.samples[0][0]))
        self.assertTrue((img == img_flip))


class TestImageNetDataModule(TestCase):
    def setUp(self) -> None:
        self.data_dir = IMAGENET_PATH

    def test_train_transform(self):
        dm = ImageNetDataModule(
            self.data_dir,
            train_transform=None,
            val_transform=None,
            image_size=224,
            num_workers=0,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        transform = dm.default_train_transform()
        expected_transform = T.Compose(
            [
                T.RandomResizedCrop(dm.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.assertEqual(str(transform.transforms), str(expected_transform.transforms))

    def test_eval_transform(self):
        dm = ImageNetDataModule(
            self.data_dir,
            train_transform=None,
            val_transform=None,
            image_size=224,
            num_workers=0,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        transform = dm.default_eval_transform()
        expected_transform = T.Compose(
            [
                T.Resize(dm.image_size + 32),
                T.CenterCrop(dm.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.assertEqual(str(transform.transforms), str(expected_transform.transforms))

    def test_accuracy_pretrained_resnet(self):
        model = torchvision.models.resnet50(weights="DEFAULT")
        model.eval().cuda()  # Needs CUDA, don't bother on CPUs

        datamodule = ImageNetDataModule(IMAGENET_PATH, batch_size=1024)
        datamodule.setup()
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        correct = 0
        total = 0
        n_batches = 10
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(train_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.76)
                    break

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_dataloader)):
                y_pred = model(x.cuda())
                correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
                total += len(y)

                if i == n_batches:
                    self.assertGreater(correct / total, 0.7)
                    break


class TestImagenetteDataModule(TestCase):
    def test_train_fast_dev_run(self):
        # set up the data module
        dm = ImagenetteDataModule(Path('./') / 'imagenette', batch_size=32, num_workers=0, image_size=64)
        # set up the dummy model
        model = DummyModel(output_dim=10)
        # set up the trainer with fast_dev_run
        trainer = pl.Trainer(fast_dev_run=True)
        # train the model for one epoch
        trainer.fit(model, dm)


class TestDataModule(TestCase):

    def test_mnist_datamodule(self):
        dm = MNISTDataModule(data_dir="./", batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup()

        # Test train dataloader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 1, 28, 28))

        # Test val dataloader
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 1, 28, 28))

        # Test test dataloader
        test_loader = dm.test_dataloader()
        batch = next(iter(test_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 1, 28, 28))

    def test_cifar10_datamodule(self):
        dm = CIFAR10DataModule(data_dir="./", batch_size=32, num_workers=0)
        dm.prepare_data()
        dm.setup()

        # Test train dataloader
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 3, 32, 32))

        # Test val dataloader
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 3, 32, 32))

        # Test test dataloader
        test_loader = dm.test_dataloader()
        batch = next(iter(test_loader))
        self.assertIsInstance(batch[0], torch.Tensor)
        self.assertEqual(batch[0].shape, (32, 3, 32, 32))


class TestLoadData(TestCase):
    def test_dishonest_users(self):
        config = {"dataset": "dishonest_users",
                  "batch_size": 32,
                  "num_workers": 0}
        dm, _, _ = get_dataset(config)
        assert len(dm.train_dataset) % config["batch_size"] != 1
        assert len(dm.val_dataset) % config["batch_size"] != 1

class TestUCI(TestCase):
    def test_hayes_roth(self):
        config = {"dataset": "hayes_roth",
                  "batch_size": 32,
                  "num_workers": 0}
        dm, _, _ = get_dataset(config)
        assert len(dm.train_dataset) % config["batch_size"] != 1
        assert len(dm.val_dataset) % config["batch_size"] != 1
