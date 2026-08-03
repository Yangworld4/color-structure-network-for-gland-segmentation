"""CPU tests for the optional complementarity feature objective."""

import unittest
from types import SimpleNamespace

import torch

from loss.heterogeneous_feature_loss import ComplementarityFeatureLoss
from main import _compute_feature_objective, _feature_diversification_loss


class ComplementarityFeatureLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.loss = ComplementarityFeatureLoss(
            private_in_channels=4,
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

    def test_feature_mask_is_resized_to_private_resolution(self):
        appearance = torch.randn(2, 4, 4, 4, requires_grad=True)
        structure = torch.randn(2, 4, 4, 4, requires_grad=True)
        labels = torch.randint(0, 2, (2, 16, 16))

        output = self.loss(appearance, structure, labels=labels)
        output.total.backward()

        self.assertTrue(torch.isfinite(output.total))
        self.assertEqual(output.valid_ratio.item(), 1.0)
        self.assertIsNotNone(appearance.grad)
        self.assertIsNotNone(structure.grad)

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

    def test_complementarity_mode_applies_only_complement_weight(self):
        args = SimpleNamespace(
            feature_loss_mode='complementarity',
            w_comp=0.05,
        )
        features_a = {
            'private': torch.randn(2, 4, 8, 8),
            'final': torch.randn(2, 4, 16, 16),
        }
        features_s = {
            'private': torch.randn(2, 4, 8, 8),
            'final': torch.randn(2, 4, 16, 16),
        }
        labels = torch.randint(0, 2, (2, 16, 16))

        selected = _compute_feature_objective(
            args,
            self.loss,
            features_a,
            features_s,
            labels=labels,
        )

        torch.testing.assert_close(
            selected['weighted'], 0.05 * selected['complement']
        )
        self.assertNotIn('semantic', selected)


if __name__ == '__main__':
    unittest.main()
