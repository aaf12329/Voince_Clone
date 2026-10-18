"""Synthetic LoRA/distribution tests; no real training data or network.

Call tree: DistillationTests -> shared LoRA/input/loss functions -> toy PyTorch models.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from voice_training.common.lora import LoRALinear, build_inputs, codec_distributions, distribution_loss


class EchoEmbedding(nn.Module):
    @property
    def weight(self):
        return torch.zeros(1)

    def forward(self, ids):
        return ids.float().unsqueeze(-1).expand(*ids.shape, 4)


class EchoModel(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def toy_model():
    embedding = EchoEmbedding()
    predictor = SimpleNamespace(small_to_mtp_projection=nn.Identity(), model=EchoModel(),
                                get_input_embeddings=lambda: [embedding] * 15,
                                lm_head=[nn.Identity()] * 15)
    talker = SimpleNamespace(device=torch.device('cpu'), text_projection=nn.Identity(),
                             get_text_embeddings=lambda: embedding, get_input_embeddings=lambda: embedding,
                             code_predictor=predictor, model=EchoModel(), codec_head=nn.Identity())
    tc = SimpleNamespace(codec_think_id=10, codec_think_bos_id=11, codec_think_eos_id=12,
                         codec_language_id={'chinese': 13}, codec_pad_id=14, codec_bos_id=15,
                         codec_eos_token_id=16)
    cfg = SimpleNamespace(tts_pad_token_id=20, tts_bos_token_id=21, tts_eos_token_id=22, talker_config=tc)
    return SimpleNamespace(talker=talker, config=cfg, dtype=torch.float32)


class DistillationTests(unittest.TestCase):




    def test_lora_zero_start_and_merged_equivalence(self):
        torch.manual_seed(10)
        base = nn.Linear(5, 3).requires_grad_(False)
        value = torch.randn(2, 5)
        expected = base(value).detach()
        layer = LoRALinear(base, 2, 4)
        self.assertTrue(torch.equal(layer(value), expected))
        layer(value).square().mean().backward()
        self.assertGreater(float(layer.b.grad.norm()), 0)
        with torch.no_grad():
            layer.b.add_(layer.b.grad, alpha=-0.1)
        before = layer(value).detach()
        self.assertFalse(torch.equal(before, expected))
        self.assertTrue(torch.allclose(layer.merge()(value), before, atol=1e-6))

    def test_identical_distributions_have_zero_kd(self):
        logits = torch.tensor([[1., 3., -2.], [-1., 0., 2.]], requires_grad=True)
        teacher = logits.detach().clone().requires_grad_(True)
        loss, ce, kd = distribution_loss(logits, teacher, torch.tensor([1, 2]), 2., 0.35)
        self.assertAlmostEqual(float(kd.detach()), 0., places=5)
        self.assertAlmostEqual(float(loss.detach()), float(ce.detach()) * 0.65, places=5)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(logits.grad.norm()), 0)

    def test_kd_gradient_moves_student_toward_teacher(self):
        student = torch.tensor([[0., 0.]], requires_grad=True)
        teacher = torch.tensor([[4., -4.]])
        loss, _, _ = distribution_loss(student, teacher, torch.tensor([0]), 2., 1.)
        loss.backward()
        self.assertLess(float(student.grad[0, 0]), 0)
        self.assertGreater(float(student.grad[0, 1]), 0)

    def test_exact_one_frame_shift_and_eos(self):
        model = toy_model()
        codes = torch.stack([torch.arange(16), torch.arange(16) + 30])
        sample = {'role_ids': torch.tensor([1, 2, 3]), 'content_ids': torch.tensor([4, 5]), 'codes': codes}
        inputs, prefix, _, labels = build_inputs(model, sample['role_ids'], sample['content_ids'],
                                                 codes, torch.full((4,), 7.))
        first, sub, returned, sub_labels = codec_distributions(model, sample, torch.full((4,), 7.), 1)
        self.assertEqual(first.shape, (3, 4))
        self.assertEqual(sub.shape, (2, 15, 4))
        self.assertTrue(torch.equal(labels, torch.tensor([0, 30, 16])))
        self.assertTrue(torch.equal(returned, labels))
        self.assertTrue(torch.equal(first[0], inputs[0, prefix - 1]))
        self.assertTrue(torch.equal(first[1], inputs[0, prefix]))
        self.assertTrue(torch.equal(first[2], inputs[0, prefix + 1]))
        self.assertTrue(torch.equal(sub_labels, codes[:, 1:]))
        # Each codebook head sees the preceding code input, not its target code.
        self.assertEqual(float(sub[0, 0, 0]), float(codes[0, 0]))
        self.assertEqual(float(sub[0, 1, 0]), float(codes[0, 1]))
        self.assertEqual(float(sub[1, 14, 0]), float(codes[1, 14]))
        _, sub_large_chunk, _, _ = codec_distributions(model, sample, torch.full((4,), 7.), 32)
        self.assertTrue(torch.equal(sub_large_chunk, sub))


if __name__ == '__main__':
    unittest.main()
