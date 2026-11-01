"""Regression: observation guard must not ban required initial latent noise."""
import contextlib
import unittest
from frozen_smoke import sample_frozen


class LatentGuardTests(unittest.TestCase):
    def test_latent_is_seeded_outside_guard_and_passed_to_sampler(self):
        class FakeTorch:
            guarded = False
            events = []

            @classmethod
            def manual_seed(cls, seed):
                cls.events.append(('seed', seed))

            @classmethod
            def randn(cls, shape, **kwargs):
                if cls.guarded:
                    raise AssertionError('Observation regeneration is forbidden')
                cls.events.append(('latent', shape))
                return 'seeded-latent'

        FakeTorch.cuda = FakeTorch

        @contextlib.contextmanager
        def guard():
            FakeTorch.guarded = True
            try:
                yield
            finally:
                FakeTorch.guarded = False

        class Measurement:
            shape = (1, 3, 64, 64)

        class Sampler:
            vae_scale_factor, device, dtype = 8, 'cuda', 'bf16'

            def sample_inverse_problem(self, measurement, operator, task, **kwargs):
                self.options = kwargs
                assert FakeTorch.guarded
                assert kwargs['latent'] == 'seeded-latent'
                with self_test.assertRaises(AssertionError):
                    FakeTorch.randn((1, 3, 64, 64))
                return 'image'

        self_test, sampler = self, Sampler()
        cfg = dict(seed=0, task='sr4', num_steps=50, step_size=6, grad_mode='euc',
                   num_optim_iters=5, normalize_grad=False)
        self.assertEqual(sample_frozen(sampler, Measurement(), object(), cfg, guard, FakeTorch), 'image')
        self.assertEqual(FakeTorch.events, [('seed', 0), ('seed', 0), ('latent', (1, 16, 32, 32))])
        self.assertEqual(sampler.options['loss_func'], 'norm')
        self.assertFalse(FakeTorch.guarded)


if __name__ == '__main__':
    unittest.main()
