import tracemalloc
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything
from torch import nn

from jumpstart.jumpstart import JumpstartRegularization
from jumpstart.metrics import JRMetricManager, pad_and_concat_table
from model.rectangular import Rectangular
from model.resnet_conv import resnet50_conv



class TestMetrics(TestCase):
    def test_all_dead(self):
        metric = JRMetricManager()
        n_points = 2
        unit_status = pd.DataFrame([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [np.nan, np.nan, -1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status = pd.DataFrame([
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['point 1', 'point 2']).T

        unit_status, point_status = torch.from_numpy(unit_status.values), torch.from_numpy(point_status.values)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)

        metric._negative_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._positive_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 10]
        metric._negative_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0]
        metric._positive_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 10]

        expected_point = torch.concat((point_status, point_status, point_status))
        self.assertTrue(torch.equal(metric.unit_status.nan_to_num(-10), unit_status.nan_to_num(-10)))
        self.assertTrue(torch.equal(metric.point_status, expected_point))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 1,
                    'point_dead_ratio': 1,
                    'deathness_ratio': 1,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 5,
                    'point_loss': 5,
                    'positive_unit_loss': 10,
                    'negative_unit_loss': 0,
                    'positive_point_loss': 10,
                    'negative_point_loss': 0,
                    'point_efficiency': 0,
                    }
        expected = {k: torch.tensor(v) for k, v in expected.items()}
        self.assertEqual(expected, result)

    def test_point_mixed(self):
        metric = JRMetricManager()
        n_points = 2
        unit_status = pd.DataFrame([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [np.nan, np.nan, -1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status = pd.DataFrame([
            [1, 1, 1, 1],  # False
            [-1, -1, -1, -1],  # True
            [1, -1, 1, 1],  # True
            [1, -1, -1, -1],  # True
            [0, 1, 0, 1],  # False
            [0, 0, 0, 0],  # False

        ], columns=['layer1', 'layer2', 'layer3', 'output'], index=['point 1', 'point 2', 'point 3',
                                                                    'point 4', 'point 5', 'point 6'])

        unit_status, point_status = torch.from_numpy(unit_status.values), torch.from_numpy(point_status.values)
        metric.update_status(unit_status, point_status)

        metric._negative_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._positive_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 10]
        metric._negative_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0]
        metric._positive_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 10]

        expected_point = point_status
        self.assertTrue(torch.equal(metric.point_status, expected_point))
        result = metric.compute()
        expected = {'unit_dead_ratio': 1.0,
                    'unit_nonlinear_ratio': 0.0,
                    'unit_linear_ratio': 0.0,
                    'point_nonlinear_ratio': 1 / 6,
                    'point_linear_ratio': 1 / 6,
                    'point_dead_ratio': 3 / 6,
                    'nonlinearity_ratio': 10 / 24,
                    'linearity_ratio': 6 / 24,
                    'deathness_ratio': 8 / 24,
                    'unit_loss': 5.0,
                    'point_loss': 5.0,
                    'positive_unit_loss': 10.0,
                    'negative_unit_loss': 0.0,
                    'positive_point_loss': 10.0,
                    'negative_point_loss': 0.0}
        for key, value in expected.items():
            self.assertAlmostEqual(value, result[key].numpy(), places=5, msg=key)

    def test_all_nonlinear(self):
        metric = JRMetricManager()
        n_points = 2
        unit_status = pd.DataFrame([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [np.nan, np.nan, 1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status = pd.DataFrame([
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['point 1', 'point 2']).T

        unit_status, point_status = torch.from_numpy(unit_status.values), torch.from_numpy(point_status.values)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)
        expected_point = torch.concat((point_status, point_status, point_status))
        metric._negative_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._positive_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._negative_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0]
        metric._positive_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0]
        self.assertTrue(torch.equal(metric.unit_status.nan_to_num(-10), unit_status.nan_to_num(-10)))
        self.assertTrue(torch.equal(metric.point_status, expected_point))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 1,
                    'point_nonlinear_ratio': 1,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 0,
                    'point_dead_ratio': 0,
                    'deathness_ratio': 0,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 1,
                    'unit_loss': 0,
                    'point_loss': 0,
                    'positive_unit_loss': 0,
                    'negative_unit_loss': 0,
                    'positive_point_loss': 0,
                    'negative_point_loss': 0,
                    'point_efficiency': 1,
                    }
        self.assertEqual(result, expected)

    def test_all_linear(self):
        metric = JRMetricManager()
        n_points = 2
        unit_status = pd.DataFrame([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [np.nan, np.nan, 0],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status = pd.DataFrame([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['point 1', 'point 2']).T
        unit_status, point_status = torch.from_numpy(unit_status.values), torch.from_numpy(point_status.values)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)
        metric.update_status(unit_status, point_status)
        expected_point = torch.concat((point_status, point_status, point_status))
        metric._negative_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 10]
        metric._positive_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._negative_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 10]
        metric._positive_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0]
        self.assertTrue(torch.equal(metric.unit_status.nan_to_num(-10), unit_status.nan_to_num(-10)))
        self.assertTrue(torch.equal(metric.point_status, expected_point))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 1,
                    'point_linear_ratio': 1,
                    'unit_dead_ratio': 0,
                    'point_dead_ratio': 0,
                    'deathness_ratio': 0,
                    'linearity_ratio': 1,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 5,
                    'point_loss': 5,
                    'positive_unit_loss': 0,
                    'negative_unit_loss': 10,
                    'positive_point_loss': 0,
                    'negative_point_loss': 10,
                    'point_efficiency': 1,
                    }
        self.assertEqual(expected, result)

    def test_all_nonlinear_batch(self):
        self.maxDiff = None
        metric = JRMetricManager()
        n_points = 2
        unit_status = pd.DataFrame([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [np.nan, np.nan, 0],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status = pd.DataFrame([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['point 1', 'point 2']).T
        unit_status, point_status = torch.from_numpy(unit_status.values), torch.from_numpy(point_status.values)

        metric.update_status(unit_status, point_status)
        unit_status = pd.DataFrame([
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
            [np.nan, np.nan, -1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T

        point_status_2 = pd.DataFrame([
            [-1, -1],
            [1, 1],
            [-1, -1],
            [-1, -1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['point 1', 'point 2']).T
        unit_status, point_status_2 = torch.from_numpy(unit_status.values), torch.from_numpy(point_status_2.values)
        metric.update_status(unit_status, point_status_2)
        metric.update_status(unit_status, point_status)
        unit_status = pd.DataFrame([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [np.nan, np.nan, 1],

        ], index=['layer1', 'layer2', 'layer3', 'output'], columns=['Unit 1', 'Unit 2', 'Unit 3']).T
        unit_status = torch.from_numpy(unit_status.values)

        expected_point = torch.concat((point_status, point_status_2, point_status))

        metric._negative_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 0,
                                        torch.ones_like(unit_status, dtype=torch.float32) * 10]
        metric._positive_unit_losses = [torch.ones_like(unit_status, dtype=torch.float32) * 10,
                                        torch.ones_like(unit_status, dtype=torch.float32) * 0]
        metric._negative_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 0,
                                         torch.ones_like(point_status, dtype=torch.float32) * 10]
        metric._positive_point_losses = [torch.ones_like(point_status, dtype=torch.float32) * 10,
                                         torch.ones_like(point_status, dtype=torch.float32) * 0]
        self.assertTrue(torch.equal(metric.unit_status.nan_to_num(-10), unit_status.nan_to_num(-10)))
        self.assertTrue(torch.equal(metric.point_status, expected_point))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 1,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 2 / 3,
                    'unit_dead_ratio': 0,
                    'point_dead_ratio': 1 / 3,
                    'deathness_ratio': 1 / 4,
                    'linearity_ratio': 2 / 3,
                    'nonlinearity_ratio': 2 / 24,
                    'unit_loss': 5,
                    'point_loss': 5,
                    'positive_unit_loss': 5,
                    'negative_unit_loss': 5,
                    'positive_point_loss': 5,
                    'negative_point_loss': 5,
                    'point_efficiency': 2/3,
                    }
        self.assertEqual(result, expected)

    def test_resnet_simple(self):
        resnet_journal = resnet50_conv(use_skip_connections=True, pretrained=True)
        resnet_journal.eval()
        jumpstart = JumpstartRegularization(skip_downsample=True)
        jumpstart.model = resnet_journal
        x = torch.ones((1, 3, 224, 224))
        resnet_journal(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        resnet_journal(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (2048, 49))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0.7591109275817871, 'point_nonlinear_ratio': 1.0,
                    'unit_linear_ratio': 0.05088028311729431,
                    'point_linear_ratio': 0.0, 'unit_dead_ratio': 0.19000880420207977, 'point_dead_ratio': 0.0,
                    'nonlinearity_ratio': 1.0,
                    'linearity_ratio': 0.0, 'deathness_ratio': 0.0, 'unit_loss': 0.19882231950759888,
                    'point_loss': 0.3257100582122803, 'positive_unit_loss': 0.20333628356456757,
                    'negative_unit_loss': 0.1943082958459854, 'positive_point_loss': 0.29698383808135986,
                    'negative_point_loss': 0.3544362485408783, 'point_efficiency': 1.0,}
        for key, value in result.items():
            self.assertAlmostEqual(float(value), expected[key])

    def test_compute_status_tables_dead(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.zeros_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        metric = JRMetricManager()
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (5, 10))
        self.assertTrue((metric.unit_status[:, :-1] == -1).all().all())
        self.assertTrue(((metric.point_status == -1).all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 1,
                    'point_dead_ratio': 1,
                    'deathness_ratio': 1,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 1,
                    'point_loss': 1,
                    'positive_unit_loss': 1,
                    'negative_unit_loss': 1,
                    'positive_point_loss': 1,
                    'negative_point_loss': 1,
                    'point_efficiency': 0,
                    }
        self.assertEqual(result, expected)

    def test_compute_status_tables_linear(self):
        self.maxDiff = None
        seed_everything(42)
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.ones_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.rand((1, 2))
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (5, 10))
        self.assertTrue((metric.unit_status[:, :-1] == 0).all().all())
        self.assertTrue(((metric.point_status == 0).all().all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0.0, 'point_nonlinear_ratio': 0.0, 'unit_linear_ratio': 1.0,
                    'point_linear_ratio': 1.0,
                    'unit_dead_ratio': 0.0, 'point_dead_ratio': 0.0, 'nonlinearity_ratio': 0.0, 'linearity_ratio': 1.0,
                    'deathness_ratio': 0.0, 'unit_loss': 216288.546875, 'point_loss': 216288.546875,
                    'positive_unit_loss': 0.0, 'negative_unit_loss': 432577.09375, 'positive_point_loss': 0.0,
                    'negative_point_loss': 432577.09375, 'point_efficiency': 1.0}
        self.assertEqual(result, expected)

    def test_compute_status_tables_dead_2(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.tensor([[1, -1]], dtype=torch.float)
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.tensor([[1, -1]], dtype=torch.float)
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (5, 10))
        self.assertTrue((metric.unit_status[:, :-1] == -1).all().all())
        self.assertTrue(((metric.point_status == -1).all().all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0.0, 'point_nonlinear_ratio': 0.0, 'unit_linear_ratio': 0.0,
                    'point_linear_ratio': 0.0,
                    'unit_dead_ratio': 1.0, 'point_dead_ratio': 1.0, 'nonlinearity_ratio': 0.0, 'linearity_ratio': 0.0,
                    'deathness_ratio': 1.0, 'unit_loss': 1.0, 'point_loss': 1.0, 'positive_unit_loss': 1.0,
                    'negative_unit_loss': 1.0, 'positive_point_loss': 1.0, 'negative_point_loss': 1.0,
                    'point_efficiency': 0.0,}
        self.assertEqual(result, expected)

    def test_compute_status_tables_nonlinear_unit(self):
        seed_everything(42)
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.tensor([[1, 1]], dtype=torch.float)
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.tensor([[-1, -1]], dtype=torch.float)
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (5, 10))
        self.assertTrue((metric.unit_status[:, :-1] == 1).all().all())
        self.assertTrue((metric.point_status[0] == 0).all())
        self.assertTrue((metric.point_status[1] == -1).all())
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 1.0, 'point_nonlinear_ratio': 0.0, 'unit_linear_ratio': 0.0,
                    'point_linear_ratio': 0.5,
                    'unit_dead_ratio': 0.0, 'point_dead_ratio': 0.5, 'nonlinearity_ratio': 0.0, 'linearity_ratio': 0.5,
                    'deathness_ratio': 0.5, 'unit_loss': 122071.0703125, 'point_loss': 122071.078125,
                    'positive_unit_loss': 0.6000000238418579, 'negative_unit_loss': 244141.5625,
                    'positive_point_loss': 0.6000000238418579, 'negative_point_loss': 244141.546875,
                    'point_efficiency': 0.5}
        self.assertEqual(result, expected)

    def test_pad_and_concat(self):
        data = {'lel': torch.tensor([1, 2, 3]), 'lol': torch.tensor([1, 2]),
                'lil': torch.tensor([1, 2, 3, 4, 5, 6])}
        data = pad_and_concat_table(data)
        expected = torch.tensor(
            [[1, 2, 3, torch.nan, torch.nan, torch.nan],
             [1, 2, torch.nan, torch.nan, torch.nan, torch.nan],
             [1, 2, 3, 4, 5, 6]
             ]).T
        self.assertTrue(torch.eq(data.nan_to_num(-100), expected.nan_to_num(-100)).all())

    def test_compute_loss_tables_dead(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.zeros_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        metric = JRMetricManager()
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_losses.shape[0], 2)
        self.assertEqual(metric.unit_losses.shape, (5, 10))
        self.assertTrue((metric.unit_losses == 1).all())
        self.assertTrue((metric.positive_unit_losses == 1).all())
        self.assertTrue((metric.negative_unit_losses == 1).all())
        self.assertTrue(((metric.point_losses == 1).all()))
        self.assertTrue(((metric.positive_point_losses == 1).all()))
        self.assertTrue(((metric.negative_point_losses == 1).all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 1,
                    'point_dead_ratio': 1,
                    'deathness_ratio': 1,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 1,
                    'point_loss': 1,
                    'positive_unit_loss': 1,
                    'negative_unit_loss': 1,
                    'positive_point_loss': 1,
                    'negative_point_loss': 1,
                    'point_efficiency': 0,
                    }
        # expected = {k: torch.tensor(v) for k, v in expected.items()}
        self.assertEqual(result, expected)
        self.maxDiff = None

    def test_compute_loss_tables_linear(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.ones_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.rand((1, 2))
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_losses.shape[0], 2)
        self.assertEqual(metric.unit_losses.shape, (5, 10))
        self.assertTrue((metric.unit_losses > 1).all())
        self.assertTrue((metric.positive_unit_losses == 0).all())
        self.assertTrue((metric.negative_unit_losses > 0).all())
        self.assertTrue(((metric.point_losses > 1).all()))
        self.assertTrue(((metric.positive_point_losses == 0).all()))
        self.assertTrue(((metric.negative_point_losses > 0).all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 1,
                    'point_linear_ratio': 1,
                    'unit_dead_ratio': 0,
                    'point_dead_ratio': 0,
                    'deathness_ratio': 0,
                    'linearity_ratio': 1,
                    'nonlinearity_ratio': 0,
                    'positive_unit_loss': 0,
                    'positive_point_loss': 0,
                    }

        expected = {k: torch.tensor(v) for k, v in expected.items()}

        self.assertTrue(expected.items() <= result.items())

        self.assertGreater(result['unit_loss'], 1)
        self.assertGreater(result['point_loss'], 1)
        self.assertGreater(result['negative_unit_loss'], 1)
        self.assertGreater(result['negative_point_loss'], 1)

    def test_compute_loss_tables_dead_2(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.tensor([[1, -1]], dtype=torch.float)
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.tensor([[1, -1]], dtype=torch.float)
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_losses.shape[0], 2)
        self.assertEqual(metric.unit_losses.shape, (5, 10))
        self.assertTrue((metric.unit_losses == 1).all())
        self.assertTrue((metric.positive_unit_losses == 1).all())
        self.assertTrue((metric.negative_unit_losses == 1).all())
        self.assertTrue(((metric.point_losses == 1).all()))
        self.assertTrue(((metric.positive_point_losses == 1).all()))
        self.assertTrue(((metric.negative_point_losses == 1).all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 1,
                    'point_dead_ratio': 1,
                    'deathness_ratio': 1,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 1,
                    'point_loss': 1,
                    'positive_unit_loss': 1,
                    'negative_unit_loss': 1,
                    'positive_point_loss': 1,
                    'negative_point_loss': 1,
                    'point_efficiency': 0,
                    }
        # expected = {k: torch.tensor(v) for k, v in expected.items()}
        self.assertEqual(result, expected)
        self.maxDiff = None

    def test_compute_loss_tables_nonlinear_unit(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.ones_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        x = torch.tensor([[1, 1]], dtype=torch.float)
        rectangular(x)
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.tensor([[-1, -1]], dtype=torch.float)
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_losses.shape[0], 2)
        self.assertEqual(metric.unit_losses.shape, (5, 10))
        self.assertTrue((metric.unit_losses > 0).all())
        self.assertTrue((metric.positive_unit_losses > 0).all())
        self.assertTrue((metric.negative_unit_losses > 0).all())
        self.assertTrue(((metric.point_losses > 0).all()))
        self.assertTrue(((metric.positive_point_losses[0] == 0).all()))
        self.assertTrue(((metric.negative_point_losses[1, 1:] == 1).all()))
        self.assertTrue(((metric.negative_point_losses[1, 0] == 0).all()))
        result = metric.compute()
        self.assertGreater(result['unit_loss'], 0)
        self.assertGreater(result['point_loss'], 0)
        self.assertGreater(result['positive_unit_loss'], 0)
        self.assertGreater(result['negative_unit_loss'], 0)
        self.assertGreater(result['positive_point_loss'], 0)
        self.assertGreater(result['negative_point_loss'], 0)

    def test_compute_status_tables_dead(self):
        rectangular = Rectangular(depth=10, width=5, input_shape=(2,), output_shape=1)

        def reset_model_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                torch.nn.init.zeros_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        reset_model_weights(child)

        rectangular.apply(reset_model_weights)
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = rectangular
        metric = JRMetricManager()
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        x = torch.rand((1, 2))
        rectangular(x)
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (5, 10))
        self.assertTrue((metric.unit_status[:, :-1] == -1).all().all())
        self.assertTrue(((metric.point_status == -1).all()))
        result = metric.compute()
        expected = {'unit_nonlinear_ratio': 0,
                    'point_nonlinear_ratio': 0,
                    'unit_linear_ratio': 0,
                    'point_linear_ratio': 0,
                    'unit_dead_ratio': 1,
                    'point_dead_ratio': 1,
                    'deathness_ratio': 1,
                    'linearity_ratio': 0,
                    'nonlinearity_ratio': 0,
                    'unit_loss': 1,
                    'point_loss': 1,
                    'positive_unit_loss': 1,
                    'negative_unit_loss': 1,
                    'positive_point_loss': 1,
                    'negative_point_loss': 1,
                    'point_efficiency': 0,
                    }
        self.assertEqual(result, expected)

    def test_status_tables(self):
        self.maxDiff = None
        seed_everything(42)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        model[0].weight.data = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float)
        model[0].bias.data = torch.tensor([0, 0], dtype=torch.float)
        model[2].weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        model[2].bias.data = torch.tensor([0, 0], dtype=torch.float)

        jumpstart = JumpstartRegularization(model=model, skip_last=False)

        jumpstart.hook(model[0], None, torch.tensor([[1, 1], [-1, -1]], dtype=torch.float))
        jumpstart.hook(model[1], None, torch.tensor([[1, 1], [-1, -1]], dtype=torch.float))
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (2, 2))
        self.assertTrue((metric.unit_status == 1).all().all())
        self.assertTrue(((metric.point_status[0] == 0).all().all()))
        self.assertTrue(((metric.point_status[1] == -1).all().all()))
        result = metric.compute()
        self.assertEqual(result['point_linear_ratio'], 0.5)
        self.assertEqual(result['point_dead_ratio'], 0.5)
        self.assertEqual(result['point_nonlinear_ratio'], 0)

    def test_status_tables_dead(self):
        self.maxDiff = None
        seed_everything(42)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        model[0].weight.data = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float)
        model[0].bias.data = torch.tensor([0, 0], dtype=torch.float)
        model[2].weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        model[2].bias.data = torch.tensor([0, 0], dtype=torch.float)

        jumpstart = JumpstartRegularization(model=model, skip_last=False)

        jumpstart.hook(model[0], None, torch.tensor([[1, 1],
                                                     [-1, -1]], dtype=torch.float))
        jumpstart.hook(model[1], None, torch.tensor([[-1, -1],
                                                     [-1, -1]], dtype=torch.float))
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 2)
        self.assertEqual(metric.unit_status.shape, (2, 2))
        self.assertTrue(((metric.point_status[0] == torch.tensor([0, -1])).all().all()))
        self.assertTrue(((metric.point_status[1] == -1).all().all()))
        result = metric.compute()
        self.assertEqual(result['point_linear_ratio'], 0)
        self.assertEqual(result['point_dead_ratio'], 1)
        self.assertEqual(result['point_nonlinear_ratio'], 0)

    def test_status_tables_mixed(self):
        self.maxDiff = None
        seed_everything(42)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        model[0].weight.data = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float)
        model[0].bias.data = torch.tensor([0, 0], dtype=torch.float)
        model[2].weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
        model[2].bias.data = torch.tensor([0, 0], dtype=torch.float)

        jumpstart = JumpstartRegularization(model=model, skip_last=False)

        jumpstart.hook(model[0], None, torch.tensor([[1, 1],
                                                     [-1, -1],
                                                     [1, 1]], dtype=torch.float))
        jumpstart.hook(model[1], None, torch.tensor([[0.1, -0.1],
                                                     [-1, -1],
                                                     [1, 1]], dtype=torch.float))
        metric = JRMetricManager()
        metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses, jumpstart.positive_point_losses,
                      jumpstart.negative_point_losses)
        self.assertEqual(metric.point_status.shape[0], 3)
        self.assertEqual(metric.unit_status.shape, (2, 2))
        self.assertTrue(((metric.point_status[0] == torch.tensor([0, 1])).all().all()))
        self.assertTrue(((metric.point_status[1] == -1).all().all()))
        result = metric.compute()
        self.assertEqual(result['point_linear_ratio'], 1 / 3)
        self.assertEqual(result['point_dead_ratio'], 1 / 3)
        self.assertEqual(result['point_nonlinear_ratio'], 0)

    def test_metric_cpu_device(self):
        if not torch.cuda.is_available():
            self.maxDiff = None
            seed_everything(42)
            model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
            model[0].weight.data = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float)
            model[0].bias.data = torch.tensor([0, 0], dtype=torch.float)
            model[2].weight.data = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
            model[2].bias.data = torch.tensor([0, 0], dtype=torch.float)

            jumpstart = JumpstartRegularization(model=model, skip_last=False)

            jumpstart.hook(model[0], None, torch.tensor([[1, 1],
                                                         [-1, -1],
                                                         [1, 1]], dtype=torch.float, device=torch.device('cuda')))
            jumpstart.hook(model[1], None, torch.tensor([[0.1, -0.1],
                                                         [-1, -1],
                                                         [1, 1]], dtype=torch.float, device=torch.device('cuda')))
            metric = JRMetricManager()
            metric.update(jumpstart.positive_unit_losses, jumpstart.negative_unit_losses,
                          jumpstart.positive_point_losses, jumpstart.negative_point_losses)

            self.assertEqual(metric.positive_unit_losses.device, torch.device('cpu'))
            self.assertEqual(metric.negative_unit_losses.device, torch.device('cpu'))
            self.assertEqual(metric.unit_losses.device, torch.device('cpu'))
            self.assertEqual(metric.positive_point_losses.device, torch.device('cpu'))
            self.assertEqual(metric.negative_point_losses.device, torch.device('cpu'))
            self.assertEqual(metric.point_losses.device, torch.device('cpu'))
            self.assertEqual(metric.unit_status.device, torch.device('cpu'))
            self.assertEqual(metric.point_status.device, torch.device('cpu'))
            for losses in metric._positive_point_losses:
                self.assertEqual(losses.device, torch.device('cpu'))
            for losses in metric._negative_point_losses:
                self.assertEqual(losses.device, torch.device('cpu'))
            for losses in metric._positive_unit_losses:
                self.assertEqual(losses.device, torch.device('cpu'))
            for losses in metric._negative_unit_losses:
                self.assertEqual(losses.device, torch.device('cpu'))

        else:
            print('No cuda device found, skipping test')

