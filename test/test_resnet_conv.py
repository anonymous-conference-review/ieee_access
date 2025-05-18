from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet50_torch

from model.resnet_conv import resnet50_conv, compute_widths_from_multiplier


class TestResnet(TestCase):
    def test_resnet50_vanilla(self):
        resnet_journal = resnet50_conv()
        resnet_torch = resnet50_torch()

        for (name_torch, layer_torch), (name_journal, layer_journal) in \
                zip(resnet_torch.named_children(), resnet_journal.named_children()):
            self.assertEqual(name_torch.replace('relu', 'activation'), name_journal)
            self.assertEqual(layer_torch.__class__, layer_journal.__class__)
            if isinstance(layer_torch, nn.Sequential):
                for (name_torch, layer_torch), (name_journal, layer_journal) in \
                        zip(layer_torch.named_children(), layer_journal.named_children()):
                    self.assertEqual(name_torch.replace('relu', 'activation'), name_journal)
                    layer_class_name_torch = layer_torch.__class__.__name__.split('.')[-1]
                    layer_class_name_journal = layer_journal.__class__.__name__.split('.')[-1]
                    self.assertEqual(layer_class_name_torch, layer_class_name_journal)
                    if layer_class_name_torch == 'Bottleneck':
                        self.assertTrue(layer_journal.use_skip_connections)
                        layer_torch = [(name_torch.replace('relu', 'activation'), layer_torch) for
                                       (name_torch, layer_torch) in
                                       layer_torch.named_children()]
                        layer_torch = sorted(layer_torch, key=lambda x: x[0])
                        layer_journal = [(name_journal.replace('relu', 'activation'), layer_journal) for
                                         (name_journal, layer_journal) in
                                         layer_journal.named_children()]
                        layer_journal = sorted(layer_journal, key=lambda x: x[0])
                        for (name_torch, layer_torch), (name_journal, layer_journal) in zip(layer_torch, layer_journal):
                            self.assertEqual(name_torch.replace('relu', 'activation'), name_journal)
                            self.assertEqual(layer_torch.__class__, layer_journal.__class__)

    def test_resnet50_vanilla_2(self):
        resnet_journal = resnet50_conv(pretrained=True)
        resnet_torch = resnet50_torch(pretrained=True)
        x = torch.rand((1, 3, 224, 224))
        expected = resnet_torch(x)
        actual = resnet_journal(x)
        self.assertTrue(torch.allclose(expected, actual))

    def test_resnet50_no_bn(self):
        resnet_journal = resnet50_conv(norm_layer=None)
        for (name_journal, layer_journal) in resnet_journal.named_children():
            self.assertNotEqual(layer_journal.__class__, nn.BatchNorm2d)
            if isinstance(layer_journal, nn.Sequential):
                for (name_journal, layer_journal) in layer_journal.named_children():
                    self.assertNotEqual(layer_journal.__class__, nn.BatchNorm2d)
                    layer_class_name = layer_journal.__class__.__name__.split('.')[-1]
                    if layer_class_name == 'Bottleneck':
                        for (name_journal, layer_journal) in layer_journal.named_children():
                            self.assertNotEqual(layer_journal.__class__, nn.BatchNorm2d)

    def test_resnet50_no_bn_2(self):
        resnet_torch = resnet50_torch(pretrained=True)
        resnet_journal = resnet50_conv(pretrained=True, norm_layer=None, strict=False)
        x = torch.rand((1, 3, 224, 224))
        expected = resnet_torch(x)
        actual = resnet_journal(x)
        self.assertFalse(torch.allclose(expected, actual))

    def test_resnet50_gelu(self):
        resnet_journal = resnet50_conv(activation=nn.GELU)
        for (name_journal, layer_journal) in resnet_journal.named_children():
            self.assertNotEqual(layer_journal.__class__, nn.ReLU)
            if name_journal == 'activation':
                self.assertEqual(layer_journal.__class__, nn.GELU)
            if isinstance(layer_journal, nn.Sequential):
                for (name_journal, layer_journal) in layer_journal.named_children():
                    self.assertNotEqual(layer_journal.__class__, nn.ReLU)
                    if name_journal == 'activation':
                        self.assertEqual(layer_journal.__class__, nn.GELU)
                    layer_class_name = layer_journal.__class__.__name__.split('.')[-1]
                    if layer_class_name == 'Bottleneck':
                        for (name_journal, layer_journal) in layer_journal.named_children():
                            self.assertNotEqual(layer_journal.__class__, nn.ReLU)
                            if name_journal == 'activation':
                                self.assertEqual(layer_journal.__class__, nn.GELU)

    def test_resnet50_gelu_2(self):
        resnet_journal = resnet50_conv(pretrained=True, activation=nn.GELU)
        resnet_torch = resnet50_torch(pretrained=True)
        x = torch.rand((1, 3, 224, 224))
        expected = resnet_torch(x)
        actual = resnet_journal(x)
        self.assertFalse(torch.allclose(expected, actual))

    def test_skip_connection(self):
        with torch.no_grad():
            resnet_journal = resnet50_conv(use_skip_connections=False, pretrained=True)
            resnet_journal.eval()
            for (name_journal, layer_journal) in resnet_journal.named_children():
                if isinstance(layer_journal, nn.Sequential):
                    for (name_journal, layer_journal) in layer_journal.named_children():
                        layer_class_name = layer_journal.__class__.__name__.split('.')[-1]
                        if layer_class_name == 'Bottleneck':
                            layer_journal.activation = None
                            x = torch.ones((1, layer_journal.conv1.in_channels, 224, 224))
                            no_skip = layer_journal(x)
                            layer_journal.use_skip_connections = True
                            skip = layer_journal(x)
                            if layer_journal.downsample:
                                identity = layer_journal.downsample(x)
                            else:
                                identity = x

                            self.assertTrue(np.allclose(identity.numpy(), skip.numpy() - no_skip.numpy(), rtol=0.001))

    def test_skip_connections_2(self):
        resnet_journal = resnet50_conv(pretrained=True, use_skip_connections=False)
        resnet_torch = resnet50_torch(pretrained=True)
        x = torch.rand((1, 3, 224, 224))
        expected = resnet_torch(x)
        actual = resnet_journal(x)
        self.assertFalse(torch.allclose(expected, actual))

    def test_width_multiplier_computation(self):
        # resnet_journal = resnet50_conv(width_multiplier=1.5, pretrained=False)
        actual_original = compute_widths_from_multiplier()
        actual_reduced = compute_widths_from_multiplier(width_multiplier=1.5)
        self.assertEqual([64, 128, 256, 512], actual_original)
        self.assertEqual([64, 96, 144, 216], actual_reduced)

    def test_width_multiplier(self):
        resnet_journal = resnet50_conv(width_multiplier=2, pretrained=False)
        expected_widths = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
        for name, layer in resnet_journal.named_children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                if 'layer' in name:
                    expected_width = expected_widths[name.split('.')[0]]
                    actual_width = layer.weight.shape[0]
                    self.assertEqual(expected_width, actual_width)

    def test_width_multiplier_15(self):
        resnet_journal = resnet50_conv(width_multiplier=1.5, pretrained=False)
        expected_widths = {'layer1': 64, 'layer2': 96, 'layer3': 144, 'layer4': 216}
        for name, layer in resnet_journal.named_children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                if 'layer' in name:
                    expected_width = expected_widths[name.split('.')[0]]
                    actual_width = layer.weight.shape[0]
                    self.assertEqual(expected_width, actual_width)

    def test_width_multiplier_15_forward(self):
        resnet_journal = resnet50_conv(width_multiplier=1.5, pretrained=False)
        x = torch.rand((1, 3, 224, 224))
        actual = resnet_journal(x)
        self.assertEqual((1, 1000), actual.shape)

    def test_width_multiplier_15_backward(self):
        resnet_journal = resnet50_conv(width_multiplier=1.5, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(resnet_journal.parameters(), lr=1e-3)
        x = torch.rand((1, 3, 224, 224))
        y = torch.randint(0, 1000, (1,))
        for _ in range(10):
            optimizer.zero_grad()
            output = resnet_journal(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
