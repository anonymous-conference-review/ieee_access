from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet50_torch

from model.resnet_dense import resnet50_dense


class TestResnet(TestCase):
    def test_resnet50_vanilla(self):
        resnet_journal = resnet50_dense()
        resnet_torch = resnet50_torch()
        resnet_torch_layers = [(k, v) for k, v in resnet_torch.named_children() if 'pool' not in k]
        for (name_torch, layer_torch), (name_journal, layer_journal) in \
                zip(resnet_torch_layers, resnet_journal.named_children()):
            self.assertEqual(name_torch.replace('relu', 'activation').replace('conv', 'dense'), name_journal)
            if 'Conv2d' in layer_torch.__class__.__name__:
                self.assertEqual('Linear', layer_journal.__class__.__name__)
            if isinstance(layer_torch, nn.Sequential):
                for (name_torch, layer_torch), (name_journal, layer_journal) in \
                        zip(layer_torch.named_children(), layer_journal.named_children()):
                    self.assertEqual(name_torch.replace('relu', 'activation').replace('conv', 'dense'), name_journal)
                    layer_class_name_torch = layer_torch.__class__.__name__.split('.')[-1]
                    layer_class_name_journal = layer_journal.__class__.__name__.split('.')[-1]
                    if 'Conv2d' in layer_class_name_torch:
                        self.assertEqual('Linear', layer_class_name_journal)
                    if layer_class_name_torch == 'Bottleneck':
                        self.assertTrue(layer_journal.use_skip_connections)
                        layer_torch = [(name_torch.replace('relu', 'activation').replace('conv', 'dense'), layer_torch) for
                                       (name_torch, layer_torch) in
                                       layer_torch.named_children()]
                        layer_torch = sorted(layer_torch, key=lambda x: x[0])
                        layer_journal = [(name_journal.replace('relu', 'activation').replace('conv', 'dense'), layer_journal) for
                                         (name_journal, layer_journal) in
                                         layer_journal.named_children()]
                        layer_journal = sorted(layer_journal, key=lambda x: x[0])
                        for (name_torch, layer_torch), (name_journal, layer_journal) in zip(layer_torch, layer_journal):
                            self.assertEqual(name_torch.replace('relu', 'activation').replace('conv', 'dense'), name_journal)
                            if 'Conv2d' in layer_class_name_torch:
                                self.assertEqual('Linear', layer_class_name_journal)

    def test_resnet50_no_bn(self):
        resnet_journal = resnet50_dense(norm_layer=None)
        for (name_journal, layer_journal) in resnet_journal.named_children():
            self.assertNotEqual(layer_journal.__class__, nn.BatchNorm1d)
            if isinstance(layer_journal, nn.Sequential):
                for (name_journal, layer_journal) in layer_journal.named_children():
                    self.assertNotEqual(layer_journal.__class__, nn.BatchNorm1d)
                    layer_class_name = layer_journal.__class__.__name__.split('.')[-1]
                    if layer_class_name == 'Bottleneck':
                        for (name_journal, layer_journal) in layer_journal.named_children():
                            self.assertNotEqual(layer_journal.__class__, nn.BatchNorm1d)

    def test_resnet50_gelu(self):
        resnet_journal = resnet50_dense(activation=nn.GELU)
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

    def test_skip_connection(self):
        with torch.no_grad():
            resnet_journal = resnet50_dense(use_skip_connections=False, pretrained=False)
            resnet_journal.eval()
            for (name_journal, layer_journal) in resnet_journal.named_children():
                if isinstance(layer_journal, nn.Sequential):
                    for (name_journal, layer_journal) in layer_journal.named_children():
                        layer_class_name = layer_journal.__class__.__name__.split('.')[-1]
                        if layer_class_name == 'Bottleneck':
                            layer_journal.activation = None
                            x = torch.ones((1, layer_journal.dense1.in_features))
                            no_skip = layer_journal(x)
                            layer_journal.use_skip_connections = True
                            skip = layer_journal(x)
                            if layer_journal.downsample:
                                identity = layer_journal.downsample(x)
                            else:
                                identity = x

                            self.assertTrue(np.allclose(identity.numpy(), skip.numpy() - no_skip.numpy(), rtol=0.001))
