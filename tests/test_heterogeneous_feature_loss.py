"""CPU tests for the optional heterogeneous FDCL feature objective."""

import unittest
from types import SimpleNamespace

import torch

from loss.heterogeneous_feature_loss import HeterogeneousFeatureLoss
from main import _compute_feature_objective, _feature_diversification_loss


class HeterogeneousFeatureLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.loss = HeterogeneousFeatureLoss(
            semantic_in_channels=4,
            num_classes=2,
            projection_dim=4,
            margin=0.2,
            confidence_threshold=0.7,
        )

    def test_supervised_loss_is_finite_and_backpropagates(self):
        appearance = torch.randn(2, 4, 8, 8, requires_grad=True)
        structure = torch.randn(2, 4, 8, 8, requires_grad=True)
        labels = torch.randint(0, 2, (2, 8, 8))

        output = self.loss(appearance, structure, labels=labels)
        output.total.backward()

        self.assertTrue(torch.isfinite(output.total))
        self.assertTrue(torch.isfinite(output.semantic))
        self.assertTrue(torch.isfinite(output.complement))
        self.assertEqual(output.valid_ratio.item(), 1.0)
        self.assertIsNotNone(appearance.grad)
        self.assertIsNotNone(structure.grad)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in self.loss.parameters()
            )
        )

    def test_unlabeled_loss_uses_only_confident_consensus(self):
        appearance = torch.randn(1, 4, 4, 4, requires_grad=True)
        structure = torch.randn(1, 4, 4, 4, requires_grad=True)
        logits_a = torch.zeros(1, 2, 4, 4)
        logits_s = torch.zeros(1, 2, 4, 4)
        logits_a[:, 0] = 8.0
        logits_s[:, 0] = 8.0

        output = self.loss(
            appearance,
            structure,
            appearance_logits=logits_a,
            structure_logits=logits_s,
        )

        self.assertEqual(output.valid_ratio.item(), 1.0)
        self.assertTrue(torch.isfinite(output.total))

    def test_hierarchical_semantic_and_private_features(self):
        hierarchical_loss = HeterogeneousFeatureLoss(
            semantic_in_channels=8,
            private_in_channels=4,
            num_classes=2,
            projection_dim=4,
        )
        semantic_a = torch.randn(2, 8, 4, 4, requires_grad=True)
        semantic_s = torch.randn(2, 8, 4, 4, requires_grad=True)
        private_a = torch.randn(2, 4, 8, 8, requires_grad=True)
        private_s = torch.randn(2, 4, 8, 8, requires_grad=True)
        labels = torch.randint(0, 2, (2, 16, 16))

        output = hierarchical_loss(
            semantic_a,
            semantic_s,
            appearance_private_feature=private_a,
            structure_private_feature=private_s,
            labels=labels,
        )
        output.total.backward()

        self.assertTrue(torch.isfinite(output.total))
        self.assertEqual(output.valid_ratio.item(), 1.0)
        self.assertIsNotNone(semantic_a.grad)
        self.assertIsNotNone(semantic_s.grad)
        self.assertIsNotNone(private_a.grad)
        self.assertIsNotNone(private_s.grad)

    def test_no_consensus_returns_differentiable_zero(self):
        appearance = torch.randn(1, 4, 4, 4, requires_grad=True)
        structure = torch.randn(1, 4, 4, 4, requires_grad=True)
        logits_a = torch.zeros(1, 2, 4, 4)
        logits_s = torch.zeros(1, 2, 4, 4)
        logits_a[:, 0] = 8.0
        logits_s[:, 1] = 8.0

        output = self.loss(
            appearance,
            structure,
            appearance_logits=logits_a,
            structure_logits=logits_s,
        )
        output.total.backward()

        self.assertEqual(output.valid_ratio.item(), 0.0)
        self.assertEqual(output.total.item(), 0.0)
        self.assertIsNotNone(appearance.grad)
        self.assertIsNotNone(structure.grad)

    def test_original_mode_keeps_original_fdcl_formula(self):
        appearance = torch.randn(1, 4, 4, 4, requires_grad=True)
        structure = torch.randn(1, 4, 4, 4, requires_grad=True)
        args = SimpleNamespace(feature_loss_mode='original', w_dis=0.75)

        selected = _compute_feature_objective(
            args,
            None,
            appearance,
            structure,
        )
        expected = 0.75 * _feature_diversification_loss(
            appearance, structure
        )

        torch.testing.assert_close(selected['weighted'], expected)


if __name__ == '__main__':
    unittest.main()
