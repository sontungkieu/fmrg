"""Explicit E.2/E.3 practical-formula reconstruction for inverse problems.

The released sampler is never edited. Generic Algorithms5/6 and the practical
appendix do not specify an identical inner-update realization: this version
chooses TableE.1's clean-latent E updates with the E.2 gain inside each update.
It does not claim recovered author code, grid nodes or full-paper reproduction.
"""
import math

VERSION = 'paper_e2_practical_inverse_v1'
PAPER = 'https://arxiv.org/html/2604.27147v3#A5.SS2'


def e_gain(sigma, sigma_next, eta):
    if not (0 <= sigma_next < sigma <= 1 and eta >= 0):
        raise ValueError('Expected descending noise time and nonnegative eta')
    return eta * (1 - sigma) * sigma_next


def cutoff_grid(full_sigma_grid, t_stop):
    """Keep scheduler nodes above cutoff; insert exact data-time cutoff.

    num_steps refers to the original full scheduler grid. Early stopping may
    reduce actual guided intervals. This is a declared reconstruction policy.
    """
    grid = [float(x) for x in full_sigma_grid]
    if not (0 < t_stop <= 1 and len(grid) >= 2):
        raise ValueError('Invalid data-time cutoff/grid')
    if abs(grid[0] - 1) > 1e-6 or grid[-1] != 0:
        raise ValueError('Grid must run from sigma1 to sigma0')
    if not all(math.isfinite(x) and a > x for a, x in zip(grid, grid[1:])):
        raise ValueError('Sigma grid must be finite and strictly decreasing')
    if t_stop == 1:
        return grid
    cut = 1 - t_stop
    above = [s for s in grid if s > cut + 1e-12]
    return above + [cut]


def scheduler_grid(sampler, latent, num_steps, height, width):
    import numpy as np
    import torch
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
    if num_steps < 1:
        raise ValueError('num_steps must be positive')
    h = 2 * (height // (sampler.vae_scale_factor * 2))
    w = 2 * (width // (sampler.vae_scale_factor * 2))
    cfg = sampler.pipeline.scheduler.config
    mu = calculate_shift((h // 2) * (w // 2), cfg.get('base_image_seq_len', 256),
                         cfg.get('max_image_seq_len', 4096), cfg.get('base_shift', .5),
                         cfg.get('max_shift', 1.15))
    times, _ = retrieve_timesteps(sampler.pipeline.scheduler, num_steps, latent.device,
                                 sigmas=np.linspace(1, 1 / num_steps, num_steps), mu=mu)
    return torch.cat([times / 1000, torch.zeros(1, device=latent.device)])


class PaperInverseAdapter:
    def __init__(self, sampler):
        self.base = sampler
        self.last_trace = None

    def sample(self, measurement, operator, task, *, latent, method, num_steps,
               eta, nopt, t_stop, guidance_scale=3.5, img_shape=(256, 256)):
        import torch
        if method not in ['E', 'J'] or latent is None or nopt < 1:
            raise ValueError('Require explicit latent,method E/J and positive nopt')
        if method == 'J' and (nopt != 1 or t_stop != 1):
            raise ValueError('This inverse J profile qualifies only nopt1,no early stop')
        full = scheduler_grid(self.base, latent, num_steps, *img_shape).detach().cpu().tolist()
        guided = cutoff_grid(full, t_stop)
        self.last_trace = dict(version=VERSION, full_sigma_grid=full, guided_sigma_grid=guided,
                               requested_data_time_stop=t_stop, effective_data_time_stop=1-guided[-1],
                               loss='sum((A(D(z))-y)**2)', method=method, events=[],
                               gain_interpretation='AppendixE.2 practical per-inner-update gain; no extra dt/nopt',
                               grid_policy='preserve full released shifted grid; insert exact cutoff; then separate unguided jump',
                               scope='Practical formula reconstruction; generic post-flow Algorithms5/6 ordering not certified')
        if method == 'J':
            # The inverse J source already implements the E.2 normalized
            # eta*dt update. The reward/Adam sampler is a separate contract.
            result = self.base.sample_inverse_problem(
                measurement, operator, task, latent=latent, num_steps=num_steps,
                step_size=eta, num_optim_iters=1, grad_mode='jac',
                guidance_scale=guidance_scale, img_shape=img_shape,
                sample_mode='flow_map2', loss_mode='pixel', loss_func='sum',
                normalize_grad=True, early_stop=0, enable_callback=False)
            self.last_trace.update(implementation='delegate_unchanged_inverse_J',
                                   expected_predict_calls=2*num_steps,
                                   expected_autograd_calls=num_steps)
            return result
        z = latent.detach()
        pe = self.base.cached_prompt_embeds
        pp = self.base.cached_pooled_prompt_embeds
        if pe.shape[0] != measurement.shape[0]:
            pe = pe.expand(measurement.shape[0], -1, -1)
            pp = pp.expand(measurement.shape[0], -1)
        events = self.last_trace['events']
        calls = grads = 0
        for index, (sigma, sigma_next) in enumerate(zip(guided, guided[1:])):
            delta = sigma - sigma_next
            gain = e_gain(sigma, sigma_next, eta)
            with torch.no_grad():
                velocity = self.base.predict_vector(z, sigma, prompt_embeds=pe,
                    pooled_prompt_embeds=pp, guidance_scale=guidance_scale, t_next=sigma_next)
                advanced = z - delta * velocity
                calls += 1
            loss = None
            if gain > 0:
                with torch.no_grad():
                    endpoint_velocity = self.base.predict_vector(z, sigma, prompt_embeds=pe,
                        pooled_prompt_embeds=pp, guidance_scale=guidance_scale, t_next=0.0)
                    endpoint = z - sigma * endpoint_velocity
                    calls += 1
                optimized, loss = self.base.fmrg_gradient_step_euclidean(
                    endpoint, operator, measurement, task=task, stepsize=gain,
                    num_iters=nopt, loss_mode='pixel', loss_func='sum', normalize_grad=False)
                advanced = advanced + (optimized - endpoint)
                grads += nopt
            z = advanced.detach()
            if not torch.isfinite(z).all():
                raise FloatingPointError('Nonfinite paper E latent; do not retune silently')
            events.append(dict(index=index, kind='guided_interval', sigma=sigma,
                               sigma_next=sigma_next, data_time=1-sigma,
                               data_time_next=1-sigma_next, eta=eta, gain_per_inner=gain,
                               nopt=nopt if gain > 0 else 0, final_inner_loss=loss))
        if guided[-1] > 0:
            sigma = guided[-1]
            with torch.no_grad():
                velocity = self.base.predict_vector(z, sigma, prompt_embeds=pe,
                    pooled_prompt_embeds=pp, guidance_scale=guidance_scale, t_next=0.0)
                z = z - sigma * velocity
            calls += 1
            events.append(dict(kind='unguided_completion', sigma=sigma, sigma_next=0.0,
                               data_time=1-sigma, data_time_next=1.0, gain_per_inner=0, nopt=0))
        self.last_trace.update(implementation='clean_endpoint_E_inner_gain',
                               expected_predict_calls=calls, expected_autograd_calls=grads)
        with torch.no_grad():
            return self.base.decode(z)
