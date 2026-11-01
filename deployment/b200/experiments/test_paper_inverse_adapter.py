import unittest
from paper_inverse_adapter import cutoff_grid, e_gain


class ScheduleTests(unittest.TestCase):
    def test_gain_uses_data_time_not_source_noise_time(self):
        self.assertAlmostEqual(e_gain(.8, .6, 6), .72)
        self.assertNotAlmostEqual(e_gain(.8, .6, 6), 6*.8*(1-.6))
        self.assertEqual(e_gain(1, .9, 6), 0)
        self.assertEqual(e_gain(.1, 0, 6), 0)

    def test_cutoff_insertion_and_distinct_completion(self):
        full = [1, .8, .6, .4, .2, 0]
        guided = cutoff_grid(full, .67)
        self.assertEqual(guided[:-1], [1, .8, .6, .4])
        self.assertAlmostEqual(guided[-1], .33)
        self.assertAlmostEqual(1-guided[-1], .67)
        self.assertNotIn(0, guided)

    def test_exact_existing_cutoff_is_not_duplicated(self):
        self.assertEqual(cutoff_grid([1, .75, .5, .25, 0], .5), [1, .75, .5])

    def test_full_schedule_is_preserved(self):
        full = [1, .9, .1, 0]
        self.assertEqual(cutoff_grid(full, 1), full)

    def test_invalid_grids_and_cutoffs_fail(self):
        for grid, cut in [([1, .5, .5, 0], .67), ([1, float('nan'), 0], .5),
                          ([.9, .5, 0], .5), ([1, .5, 0], 0), ([1, .5, 0], 1.1)]:
            with self.assertRaises(ValueError):
                cutoff_grid(grid, cut)


class TorchFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        import torch
        sys.path.insert(0, '/opt/fmrg/source')
        from fluxfm_sampler_ip import FluxFlowMapSampler
        cls.torch, cls.Sampler = torch, FluxFlowMapSampler

    def sampler(self):
        s = object.__new__(self.Sampler)
        s.dtype = self.torch.float64
        s.decode = lambda z: z
        s.predict_vector = lambda z, t_cur, **kw: .2*z + .1 + .5*(kw.get('t_next') or 0)
        return s

    def assert_pixel_close(self, got, expected, operations=8):
        # Source deliberately casts decoded pixels toFP32, including in a
        # float64 toy. Bound that rounding explicitly, not as float64 exactness.
        scale=max(1.,float(expected.detach().abs().max()))
        error=float((got-expected).detach().abs().max())
        bound=operations*self.torch.finfo(self.torch.float32).eps*scale
        print('ANALYTIC_FP32_BOUND',self._testMethodName,'error',error,'bound',bound,flush=True)
        self.assertLessEqual(error,bound)

    def test_squared_pixel_loss_and_derivative_match_eq134(self):
        torch, s = self.torch, self.sampler()
        z = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64, requires_grad=True)
        loss = s._compute_residual_loss(z, 'sum')
        g, = torch.autograd.grad(loss, z)
        self.assertEqual(float(loss), 30.)
        self.assertTrue(torch.equal(g, 2*z))

    def test_E_inner_gain_is_applied_each_iteration(self):
        torch, s = self.torch, self.sampler()
        class Identity:
            def A(self, x): return x
        gain = e_gain(.8, .6, .5)
        start = torch.tensor([[2.]], dtype=torch.float64)
        y = torch.zeros_like(start)
        got, _ = s.fmrg_gradient_step_euclidean(start, Identity(), y, 'sr',
            stepsize=gain, num_iters=3, loss_mode='pixel', loss_func='sum')
        expected = start * (1-2*gain)**3
        self.assert_pixel_close(got,expected)

    def test_inverse_J_source_matches_normalized_eta_dt_equation(self):
        torch, s = self.torch, self.sampler()
        class Identity:
            def A(self, x): return x
        z = torch.tensor([[[[.2, .4]]]], dtype=torch.float64, requires_grad=True)
        sigma, sigma_next, eta = .6, .4, 3.
        endpoint_velocity = s.predict_vector(z, sigma, t_next=0.)
        step_velocity = s.predict_vector(z, sigma, t_next=sigma_next)
        vel_norm = torch.linalg.vector_norm(step_velocity.detach(), dim=(1, 2, 3), keepdim=True)
        y = torch.tensor([[[[.7, -.1]]]], dtype=torch.float64)
        got, _, original = s.fmrg_gradient_step_jacobian(z, sigma, Identity(), y, 'sr',
            velocity=endpoint_velocity, stepsize=eta, num_iters=1, normalize_grad=True,
            velocity_norm=vel_norm, loss_mode='pixel', loss_func='sum', t_next=0.)
        endpoint = z.detach() - sigma * endpoint_velocity.detach()
        analytic = 2*(1-.2*sigma)*(endpoint-y)
        correction = eta * analytic/(torch.linalg.vector_norm(analytic, dim=(1, 2, 3), keepdim=True)+1e-8)*vel_norm
        self.assert_pixel_close(got,correction)
        advanced = original - (sigma-sigma_next)*(step_velocity.detach()+got)
        expected = z.detach() - (sigma-sigma_next)*step_velocity.detach() - (sigma-sigma_next)*correction
        self.assert_pixel_close(advanced,expected)

    def test_actual_E_adapter_cutoff_and_final_jump_without_rng(self):
        from unittest.mock import patch
        from paper_inverse_adapter import PaperInverseAdapter
        torch, s = self.torch, self.sampler()
        class Identity:
            def A(self, x): return x
        calls = []
        def velocity(z, sigma, **kw):
            calls.append((sigma, kw['t_next']))
            return torch.zeros_like(z)
        s.predict_vector = velocity
        s.cached_prompt_embeds = torch.zeros(1, 1, 1)
        s.cached_pooled_prompt_embeds = torch.zeros(1, 1)
        full = torch.tensor([1, .8, .6, .4, .2, 0], dtype=torch.float64)
        start = torch.tensor([[[[2.]]]], dtype=torch.float64)
        adapter = PaperInverseAdapter(s)
        with patch('paper_inverse_adapter.scheduler_grid', return_value=full), patch('torch.randn', side_effect=AssertionError('RNG forbidden')):
            result = adapter.sample(torch.zeros_like(start), Identity(), 'sr', latent=start,
                method='E', num_steps=5, eta=.05, nopt=3, t_stop=.67)
        expected = start.clone()
        guided = cutoff_grid(full.tolist(), .67)
        for sigma, sigma_next in zip(guided, guided[1:]):
            expected *= (1-2*e_gain(sigma, sigma_next, .05))**3
        self.assert_pixel_close(result,expected,operations=32)
        self.assertEqual(adapter.last_trace['events'][-1]['kind'], 'unguided_completion')
        self.assertAlmostEqual(calls[-1][0], .33)
        self.assertEqual(calls[-1][1], 0.)
        self.assertEqual(adapter.last_trace['expected_autograd_calls'], 9)
        self.assertEqual(adapter.last_trace['expected_predict_calls'], len(calls))

    def test_unqualified_J_profiles_fail_before_sampling(self):
        from paper_inverse_adapter import PaperInverseAdapter
        torch = self.torch
        adapter = PaperInverseAdapter(self.sampler())
        with self.assertRaises(ValueError):
            adapter.sample(None, None, 'sr', latent=torch.zeros(1), method='J',
                           num_steps=2, eta=1, nopt=2, t_stop=1)


if __name__ == '__main__':
    unittest.main()
