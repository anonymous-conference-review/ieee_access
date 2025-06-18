import tracemalloc
from unittest import TestCase

import numpy as np
import torch

from jumpstart.jumpstart import JumpstartRegularization
from model.resnet_conv import resnet50_conv


class TestJumpstartRegularization(TestCase):

    def test_positive_point_storage(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertTrue(torch.equal(jumpstart.positive_point_losses['LEL'], torch.tensor([0.0, 0.0, 2.0])))

    def test_negative_point_storage(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertTrue(torch.equal(jumpstart.negative_point_losses['LEL'], torch.tensor([0.0, 2.0, 0.0])))

    def test_positive_unit_storage(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertTrue(torch.equal(jumpstart.positive_unit_losses['LEL'], torch.tensor([0.0, 0.0, 2.0])))

    def test_negative_unit_storage(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertTrue(torch.equal(jumpstart.negative_unit_losses['LEL'], torch.tensor([0.0, 2.0, 0.0])))

    def test_full_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_full_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_unit_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='unit')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_unit_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='unit')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='point')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='point')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_unit_positive_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='unit_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_unit_positive_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='unit_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_positive_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_positive_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_positive_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_positive_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_full_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1.0, 1.0, -1.0],
                               [3.0, 4.0, -1.0],
                               [-3.0, 4.0, -1.0]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', )
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), np.sqrt(2))

    def test_full_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', )
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), np.sqrt(2))

    def test_unit_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='unit')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), np.sqrt(2))

    def test_unit_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='unit')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='point')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='point')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), np.sqrt(2))

    def test_unit_positive_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='unit_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1)

    def test_unit_positive_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='unit_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_positive_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 0)

    def test_point_positive_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1)

    def test_positive_norm_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1)

    def test_positive_2_norm_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(aggr='norm', jr_mode='positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1)

    def test_all_but_unit_positive_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_unit_positive', tie_breaking='average')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_all_but_unit_positive_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_unit_positive', tie_breaking='average')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_all_but_unit_negative_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_unit_negative', tie_breaking='average')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_all_but_unit_negative_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_unit_negative', tie_breaking='average')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_all_but_point_positive_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_all_but_point_positive_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_point_positive')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_all_but_point_negative_balanced_loss(self):
        #                     non-linear | linear | dead
        preact = torch.tensor([[1, 1, -1],
                               [3, 4, -1],
                               [-3, 4, -1]
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_point_negative')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 2 / 3)

    def test_all_but_point_negative_2_balanced_loss(self):
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization(jr_mode='all_but_point_negative')
        jumpstart.hook('LEL', None, preact)
        self.assertAlmostEqual(jumpstart.loss.numpy(), 1 / 3)

    def test_skip_downsample(self):
        resnet_journal = resnet50_conv(use_skip_connections=True, pretrained=False)
        resnet_journal.eval()
        jumpstart = JumpstartRegularization(skip_downsample=True)
        jumpstart.model = resnet_journal

        for layer_name in jumpstart.hook_handlers.keys():
            self.assertNotIn('downsample', layer_name)

    def test_no_skip_downsample(self):
        resnet_journal = resnet50_conv(use_skip_connections=True, pretrained=False)
        resnet_journal.eval()
        jumpstart = JumpstartRegularization(skip_downsample=False)
        jumpstart.model = resnet_journal
        i = 0
        for layer_name in jumpstart.hook_handlers.keys():
            if 'downsample' in layer_name:
                i += 1

        self.assertEqual(4, i)

    def test_metric_leak(self):
        n_iter = 1000
        preact = torch.tensor([[1, 2, -1],  # non-linear
                               [3, 4, 1],  # linear
                               [-3, -4, -1]  # dead
                               ]).to(torch.float)
        jumpstart = JumpstartRegularization()

        tracemalloc.start()
        jumpstart.hook('LEL', None, preact)
        snapshot1 = tracemalloc.take_snapshot()
        for _ in range(n_iter):
            # wait until the memory is stable
            jumpstart.hook('LEL', None, preact)
            first_size, first_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        for _ in range(n_iter):
            jumpstart.hook('LEL', None, preact)
            second_size, second_peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()

            print(f"{second_size=}, {second_peak=}")

        snapshot2 = tracemalloc.take_snapshot()
        second_size, second_peak = tracemalloc.get_traced_memory()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        tracemalloc.stop()

        print(f"{first_size=}, {first_peak=}")
        print(f"{second_size=}, {second_peak=}")

        for stat in top_stats:
            print(stat)
