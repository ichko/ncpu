import unittest
from argparse import Namespace

import torch

from ncpu.nca import NeuralCA
from ncpu.dataset import NCPUDataset, sample_4bit_adder
from ncpu.noise import (
    DAMAGE_REGISTRY,
    INPUT_DAMAGE_KINDS,
    STATE_DAMAGE_KINDS,
    Damage,
    apply_input_damage,
    apply_state_damage,
    parse_damage_specs,
    perlin_field_2d,
)
from ncpu.trainer import NCPUTrainer


def make_input_damage(kind, amplitude=1.0, prob=1.0, **params):
    return Damage(kind=kind, amplitude=amplitude, prob=prob, params=params)


class TestDamageSpec(unittest.TestCase):
    def test_parse_roundtrip(self):
        d = Damage.parse("perlin:40:0.5:octaves=3,base_scale=16")
        self.assertEqual(d.kind, "perlin")
        self.assertAlmostEqual(d.amplitude, 40.0)
        self.assertAlmostEqual(d.prob, 0.5)
        self.assertEqual(d.params, {"octaves": 3, "base_scale": 16.0})
        d2 = Damage.from_dict(d.to_dict())
        self.assertEqual(d, d2)
        self.assertIn("perlin", d2.label())

    def test_parse_defaults(self):
        d = Damage.parse("salt_pepper:0.1")
        self.assertEqual(d.prob, 1.0)
        self.assertEqual(d.params, {})

    def test_parse_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            Damage.parse("bogus:1.0")

    def test_parse_damage_specs(self):
        ds = parse_damage_specs(["gaussian:10:0.5", "jitter:2"])
        self.assertEqual([d.kind for d in ds], ["gaussian", "jitter"])
        self.assertEqual(parse_damage_specs(None), [])

    def test_registry_sides(self):
        for k in INPUT_DAMAGE_KINDS:
            self.assertEqual(DAMAGE_REGISTRY[k]["side"], "input")
        for k in STATE_DAMAGE_KINDS:
            self.assertEqual(DAMAGE_REGISTRY[k]["side"], "state")
        # all proposed kinds are present
        self.assertEqual(
            set(INPUT_DAMAGE_KINDS),
            {
                "gaussian",
                "perlin",
                "salt_pepper",
                "occlusion",
                "blur",
                "brightness_contrast",
                "quantize",
                "jitter",
                "speckle",
            },
        )
        self.assertEqual(
            set(STATE_DAMAGE_KINDS),
            {"state_gaussian", "state_perlin", "channel_dropout", "block_dropout"},
        )


class TestInputDamages(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.img = torch.randint(0, 256, (4, 1, 32, 32)).float()

    def test_shape_range_dtype(self):
        for kind in INPUT_DAMAGE_KINDS:
            d = make_input_damage(
                kind,
                0.5 if kind == "salt_pepper" else 2.0,
                **{"levels": 4} if kind == "quantize" else {},
            )
            out = apply_input_damage(self.img, d)
            self.assertEqual(out.shape, self.img.shape, kind)
            self.assertTrue(torch.is_floating_point(out), kind)
            self.assertGreaterEqual(out.min().item(), 0.0, kind)
            self.assertLessEqual(out.max().item(), 255.0, kind)

    def test_3d_input_keeps_shape(self):
        img3 = self.img.squeeze(1)
        out = apply_input_damage(img3, make_input_damage("perlin", 40.0))
        self.assertEqual(out.shape, img3.shape)

    def test_zero_amplitude_is_identity(self):
        for kind, params in [
            ("gaussian", {}),
            ("perlin", {}),
            ("salt_pepper", {}),
            ("occlusion", {}),
            ("blur", {}),
            ("brightness_contrast", {}),
            ("jitter", {}),
            ("speckle", {}),
            ("quantize", {"levels": 256}),
        ]:
            d = make_input_damage(kind, 0.0, **params)
            out = apply_input_damage(self.img, d)
            self.assertTrue(torch.equal(out, self.img), kind)

    def test_prob_zero_is_identity(self):
        for kind in INPUT_DAMAGE_KINDS:
            d = make_input_damage(kind, 5.0, prob=0.0)
            out = apply_input_damage(self.img, d)
            self.assertTrue(torch.equal(out, self.img), kind)

    def test_salt_pepper_flips_to_extremes(self):
        d = make_input_damage("salt_pepper", 0.5)
        out = apply_input_damage(self.img, d)
        changed = out != self.img
        self.assertTrue(changed.any())
        vals = out[changed].unique()
        self.assertTrue(set(vals.tolist()).issubset({0.0, 255.0}))

    def test_occlusion_writes_fill_value(self):
        d = make_input_damage("occlusion", 1, **{"block_size": 8, "fill": 0})
        out = apply_input_damage(self.img, d)
        self.assertTrue((out == 0).any())

    def test_jitter_shifts_content(self):
        d = make_input_damage("jitter", 3)
        out = apply_input_damage(self.img, d)
        # same multiset of values, different arrangement (with high probability)
        self.assertEqual(out.sum().item(), self.img.sum().item())

    def test_state_kind_rejected_on_input(self):
        with self.assertRaises(ValueError):
            apply_input_damage(self.img, make_input_damage("state_gaussian", 1.0))


class TestPerlinField(unittest.TestCase):
    def test_shape_and_range(self):
        f = perlin_field_2d(4, 32, 48, octaves=4, base_scale=8)
        self.assertEqual(f.shape, (4, 1, 32, 48))
        self.assertLessEqual(f.max().item(), 1.0 + 1e-5)
        self.assertGreaterEqual(f.min().item(), -1.0 - 1e-5)

    def test_smoothness(self):
        # spatially adjacent pixels should be much more correlated than white noise
        f = perlin_field_2d(1, 64, 64, octaves=4, base_scale=8)[0, 0]
        white = torch.randn(64, 64) * 2 - 1

        def lag1_corr(t):
            return torch.corrcoef(
                torch.stack([t[:, :-1].flatten(), t[:, 1:].flatten()])
            )[0, 1]

        self.assertGreater(lag1_corr(f).item(), 0.5)
        self.assertLess(lag1_corr(white).item(), 0.2)

    def test_seeded_reproducible(self):
        torch.manual_seed(42)
        a = perlin_field_2d(2, 16, 16)
        torch.manual_seed(42)
        b = perlin_field_2d(2, 16, 16)
        self.assertTrue(torch.equal(a, b))


class TestStateDamages(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 4, 16, 16).clamp(-10, 10)
        # channel 1 is read-only
        self.mask = torch.ones(1, 4, 1, 1)
        self.mask[:, 1] = 0.0

    def test_shape_and_clamp(self):
        for kind in STATE_DAMAGE_KINDS:
            d = make_input_damage(kind, 1.0)
            out = apply_state_damage(self.x, d, channel_mask=self.mask)
            self.assertEqual(out.shape, self.x.shape, kind)
            self.assertLessEqual(out.max().item(), 10.0, kind)
            self.assertGreaterEqual(out.min().item(), -10.0, kind)

    def test_read_only_channel_untouched(self):
        for kind in STATE_DAMAGE_KINDS:
            d = make_input_damage(kind, 1.0)
            out = apply_state_damage(self.x, d, channel_mask=self.mask)
            self.assertTrue(torch.equal(out[:, 1], self.x[:, 1]), kind)

    def test_channel_dropout_full_drops_writable(self):
        d = make_input_damage("channel_dropout", 1.0)
        out = apply_state_damage(self.x, d, channel_mask=self.mask)
        self.assertTrue(torch.all(out[:, 0] == 0))
        self.assertTrue(torch.all(out[:, 2:] == 0))

    def test_block_dropout_zeros_blocks(self):
        d = make_input_damage("block_dropout", 2, **{"block_size": 8})
        out = apply_state_damage(self.x, d, channel_mask=self.mask)
        self.assertTrue((out[:, 0] == 0).any())

    def test_input_kind_rejected_on_state(self):
        with self.assertRaises(ValueError):
            apply_state_damage(self.x, make_input_damage("perlin", 1.0))


class TestNeuralCADamages(unittest.TestCase):
    def make_nca(self, state_damages=None):
        return NeuralCA(
            channels=4,
            hidden_channels=[8],
            fire_rate=1.0,
            alive_threshold=0.0,
            zero_initialization=False,
            kernel_size=3,
            read_only_dims=[1],
            state_damages=state_damages,
        )

    def test_forward_with_state_damages(self):
        torch.manual_seed(0)
        nca = self.make_nca(
            state_damages=[
                Damage("state_gaussian", 0.5, 1.0),
                Damage("state_perlin", 0.5, 0.5),
            ]
        )
        x = torch.randn(2, 4, 12, 12)
        out = nca.forward(x, steps=5)
        self.assertEqual(out.shape, (2, 6, 4, 12, 12))
        self.assertLessEqual(out.max().item(), 10.0)

    def test_forward_backward_compatible(self):
        torch.manual_seed(0)
        nca = self.make_nca()
        x = torch.randn(2, 4, 12, 12)
        out_default = nca.forward(x, steps=4)
        out_explicit_empty = nca.forward(x, steps=4, state_damages=[])
        self.assertEqual(out_default.shape, out_explicit_empty.shape)

    def test_forward_override_per_call(self):
        torch.manual_seed(0)
        nca = self.make_nca()
        x = torch.randn(2, 4, 12, 12)
        out = nca.forward(
            x, steps=4, state_damages=[Damage("channel_dropout", 1.0, 1.0)]
        )
        self.assertEqual(out.shape, (2, 5, 4, 12, 12))
        # read-only channel 1: delta is masked to 0 and dropout must not touch
        # it either, so it stays at its initial value in every frame
        self.assertTrue(
            torch.equal(out[:, :, 1], x[:, 1].unsqueeze(1).expand_as(out[:, :, 1]))
        )


class TestTrainerIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        ds_config = Namespace(
            W=32, H=32, r=3, spacing=(2, 6), sampler=sample_4bit_adder, balanced=False
        )
        self.dataset = NCPUDataset(ds_config)
        nca = NeuralCA(
            channels=4,
            hidden_channels=[8],
            fire_rate=1.0,
            alive_threshold=0.0,
            zero_initialization=False,
            kernel_size=3,
            read_only_dims=[1],
            state_damages=[Damage("state_gaussian", 0.2, 0.5)],
        )
        self.trainer = NCPUTrainer(
            nca,
            self.dataset.get_dataloader(batch_size=2),
            lr=1e-3,
            gaussian_noise=-1,
            loss_fn=lambda rollout, out, **kw: ((rollout[:, -1, 0] - out) ** 2).mean(),
            input_implant_type="all",
            input_damages=[Damage("perlin", 40.0, 1.0)],
            damage_mode="sample",
        )

    def test_optim_step_with_damages(self):
        info = self.trainer.optim_step(steps=4)
        self.assertIn("loss", info)
        self.assertTrue(torch.isfinite(torch.tensor(info["loss"])))
        self.assertEqual(info["rollout"], None)

    def test_evaluate_damage_matrix(self):
        eval_batch = next(iter(self.dataset.get_dataloader(batch_size=2)))
        configs = [
            {"label": "clean"},
            {"label": "perlin", "input_damages": [Damage("perlin", 40.0)]},
            {"label": "state", "state_damages": [Damage("state_gaussian", 0.5, 1.0)]},
        ]
        result = self.trainer.evaluate(
            eval_batch, [], n_rollouts=2, steps=4, damage_configs=configs
        )
        self.assertEqual(
            [c["label"] for c in result["per_cfg"]], ["clean", "perlin", "state"]
        )
        for c in result["per_cfg"]:
            self.assertIn("bits", c)
            self.assertIn("loss", c)
            self.assertEqual(
                c["input_damages"][0]["kind"] if c["input_damages"] else None,
                "perlin" if c["label"] == "perlin" else None,
            )
        self.assertIn("eval_bits", result)

    def test_evaluate_legacy_noise_configs_still_work(self):
        eval_batch = next(iter(self.dataset.get_dataloader(batch_size=2)))
        result = self.trainer.evaluate(eval_batch, [(0.0, 0.0)], n_rollouts=1, steps=4)
        self.assertEqual(len(result["per_cfg"]), 1)
        self.assertAlmostEqual(result["per_cfg"][0]["noise_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
