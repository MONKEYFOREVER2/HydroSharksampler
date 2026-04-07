"""
HydroSharkSampler - nodes.py
ComfyUI custom node package optimized for Z-Image-Turbo (rectified-flow/CONST)
and compatible with RES4LYF ClownsharkSampler pipelines.

Nodes:
  - HydroSharkScheduler  : Custom sigma schedule for flow models
  - HydroSharkSampler    : Pure SAMPLER node (plug into SamplerCustom)
  - HydroSharkKSampler   : All-in-one convenience KSampler
"""

import math
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
from tqdm.auto import trange


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_sigmas(steps, mode, midpoint_bias, sharpness):
    t = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)

    if mode == "flow_linear":
        s = 1.0 - t

    elif mode == "flow_sigmoid":
        raw = sharpness * (t - midpoint_bias)
        sig = 1.0 / (1.0 + torch.exp(raw))
        sig_min, sig_max = sig[-1].item(), sig[0].item()
        denom = sig_max - sig_min
        s = (sig - sig_min) / denom if abs(denom) > 1e-8 else 1.0 - t

    elif mode == "flow_cosine":
        s = 0.5 * (1.0 + torch.cos(math.pi * t))
        if midpoint_bias > 0:
            bias_exp = math.log(0.5) / (math.log(midpoint_bias + 1e-8) + 1e-8)
            bias_exp = max(0.1, min(10.0, bias_exp))
            s = s ** bias_exp

    elif mode == "adaptive_blend":
        s_lin = 1.0 - t
        raw = sharpness * (t - midpoint_bias)
        sig = 1.0 / (1.0 + torch.exp(raw))
        sig_min, sig_max = sig[-1].item(), sig[0].item()
        denom = sig_max - sig_min
        s_sig = (sig - sig_min) / denom if abs(denom) > 1e-8 else s_lin
        w = midpoint_bias
        s = (1.0 - w) * s_lin + w * s_sig

    else:
        s = 1.0 - t

    return torch.clamp(s, 0.0, 1.0)


def _apply_denoise(sigmas_full, denoise, steps):
    start_idx = int((1.0 - denoise) * steps)
    start_idx = max(0, min(start_idx, steps - 1))
    sliced = sigmas_full[start_idx:]
    terminal = torch.zeros(1, dtype=sliced.dtype, device=sliced.device)
    return torch.cat([sliced, terminal], dim=0)


def _cfg_rescale(x0, cfg, rescale_alpha):
    if rescale_alpha <= 0.0 or cfg <= 1.0:
        return x0
    std_guided = x0.std() + 1e-8
    std_target = std_guided / cfg
    x0_rescaled = x0 * (std_target / std_guided)
    return rescale_alpha * x0_rescaled + (1.0 - rescale_alpha) * x0


def _get_model_output(model, x, sigma, extra_args):
    sigma_tensor = sigma * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    return model(x, sigma_tensor, **extra_args)


def _flow_step(x, x0, sigma, sigma_next):
    """
    Exact analytical flow-matching interpolation step.
    x_next = (sigma_next / sigma) * x + (1 - sigma_next / sigma) * x0
    Equivalent to: velocity=(x-x0)/sigma, x_next=x+velocity*(sigma_next-sigma)
    but avoids division blowup near sigma=0.
    """
    if sigma.item() < 1e-6:
        return x0
    ratio = sigma_next / sigma
    return ratio * x + (1.0 - ratio) * x0


# ---------------------------------------------------------------------------
# Inner sampling functions
# ---------------------------------------------------------------------------

def _sample_hydro_euler(model, x, sigmas, extra_args, callback, disable,
                        eta, s_noise, cfg_rescale_factor, **kwargs):
    extra_args = extra_args or {}
    cfg = extra_args.get("cfg", 1.0)
    n_steps = len(sigmas) - 1

    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        x0 = _get_model_output(model, x, sigma, extra_args)
        x0 = _cfg_rescale(x0, cfg, cfg_rescale_factor)
        x = _flow_step(x, x0, sigma, sigma_next)

        if eta > 0.0 and sigma_next.item() > 1e-6:
            x = x + eta * sigma_next * s_noise * torch.randn_like(x)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": x0})

    return x


def _sample_hydro_heun2(model, x, sigmas, extra_args, callback, disable,
                        eta, s_noise, cfg_rescale_factor, momentum, corrector_steps,
                        **kwargs):
    extra_args = extra_args or {}
    cfg = extra_args.get("cfg", 1.0)
    n_steps = len(sigmas) - 1
    x0_ema = None

    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Predictor
        x0 = _get_model_output(model, x, sigma, extra_args)
        x0 = _cfg_rescale(x0, cfg, cfg_rescale_factor)

        if momentum > 0.0:
            x0_ema = x0.clone() if x0_ema is None else momentum * x0_ema + (1.0 - momentum) * x0
            x0_eff = x0_ema
        else:
            x0_eff = x0

        x_pred = _flow_step(x, x0_eff, sigma, sigma_next)

        # Corrector (Heun) - use raw x0, NOT the EMA-smoothed one, to preserve 2nd-order accuracy
        if sigma_next.item() > 1e-6 and corrector_steps > 0:
            x0_corr = _get_model_output(model, x_pred, sigma_next, extra_args)
            x0_corr = _cfg_rescale(x0_corr, cfg, cfg_rescale_factor)
            x0_avg = 0.5 * (x0 + x0_corr)
            x = _flow_step(x, x0_avg, sigma, sigma_next)
        else:
            x = x_pred

        if eta > 0.0 and sigma_next.item() > 1e-6:
            x = x + eta * sigma_next * s_noise * torch.randn_like(x)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": x0})

    return x


def _sample_hydro_dpm(model, x, sigmas, extra_args, callback, disable,
                      eta, s_noise, cfg_rescale_factor, **kwargs):
    """
    DPM-Solver++(2M) for rectified-flow.
    Step 1: Euler. Step 2+: linear extrapolation of x0 estimates.
      h      = sigma - sigma_next   (positive, current step size)
      h_prev = sigma_prev - sigma   (positive, previous step size)
      r      = h_prev / h           (positive ratio)
      x0_prime = (1 + 1/(2r)) * x0_curr - (1/(2r)) * x0_prev
    """
    extra_args = extra_args or {}
    cfg = extra_args.get("cfg", 1.0)
    n_steps = len(sigmas) - 1
    x0_prev = None
    sigma_prev = None

    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        x0 = _get_model_output(model, x, sigma, extra_args)
        x0 = _cfg_rescale(x0, cfg, cfg_rescale_factor)

        if x0_prev is None:
            x = _flow_step(x, x0, sigma, sigma_next)
        else:
            h = sigma - sigma_next
            h_prev = sigma_prev - sigma
            r = h_prev / (h + 1e-8)
            x0_prime = (1.0 + 0.5 / r) * x0 - (0.5 / r) * x0_prev
            x = _flow_step(x, x0_prime, sigma, sigma_next)

        if eta > 0.0 and sigma_next.item() > 1e-6:
            x = x + eta * sigma_next * s_noise * torch.randn_like(x)

        x0_prev = x0
        sigma_prev = sigma

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": x0})

    return x


def _sample_hydro_momentum(model, x, sigmas, extra_args, callback, disable,
                           eta, s_noise, cfg_rescale_factor, momentum, **kwargs):
    extra_args = extra_args or {}
    cfg = extra_args.get("cfg", 1.0)
    n_steps = len(sigmas) - 1
    x0_ema = None

    for i in trange(n_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        x0 = _get_model_output(model, x, sigma, extra_args)
        x0 = _cfg_rescale(x0, cfg, cfg_rescale_factor)

        if momentum > 0.0:
            x0_ema = x0.clone() if x0_ema is None else momentum * x0_ema + (1.0 - momentum) * x0
            x0_eff = x0_ema
        else:
            x0_eff = x0

        x = _flow_step(x, x0_eff, sigma, sigma_next)

        if eta > 0.0 and sigma_next.item() > 1e-6:
            x = x + eta * sigma_next * s_noise * torch.randn_like(x)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": x0})

    return x


_SAMPLER_FUNCTIONS = {
    "hydro_euler":    _sample_hydro_euler,
    "hydro_heun2":    _sample_hydro_heun2,
    "hydro_dpm":      _sample_hydro_dpm,
    "hydro_momentum": _sample_hydro_momentum,
}


# ---------------------------------------------------------------------------
# NODE 1: HydroSharkScheduler
# ---------------------------------------------------------------------------

class HydroSharkScheduler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "steps":         ("INT",   {"default": 9,   "min": 1,   "max": 100,  "step": 1}),
                "denoise":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "midpoint_bias": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "sharpness":     ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "mode":          (["flow_linear", "flow_sigmoid", "flow_cosine", "adaptive_blend"],),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION     = "get_sigmas"
    CATEGORY     = "HydroShark/Scheduling"

    def get_sigmas(self, model, steps, denoise, midpoint_bias, sharpness, mode):
        sigmas_full = _build_sigmas(steps, mode, midpoint_bias, sharpness)
        sigmas = _apply_denoise(sigmas_full, denoise, steps)
        device = comfy.model_management.get_torch_device()
        return (sigmas.to(device),)


# ---------------------------------------------------------------------------
# NODE 2: HydroSharkSampler
# ---------------------------------------------------------------------------

class HydroSharkSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "method":          (["hydro_euler", "hydro_heun2", "hydro_dpm", "hydro_momentum"],),
                "eta":             ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "momentum":        ("FLOAT", {"default": 0.0,  "min": 0.0,  "max": 0.99, "step": 0.01}),
                "cfg_rescale":     ("FLOAT", {"default": 0.7,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "s_noise":         ("FLOAT", {"default": 1.0,  "min": 0.5,  "max": 2.0,  "step": 0.01}),
                "corrector_steps": ("INT",   {"default": 1,    "min": 0,    "max": 3,    "step": 1}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION     = "get_sampler"
    CATEGORY     = "HydroShark/Sampling"

    def get_sampler(self, method, eta, momentum, cfg_rescale, s_noise, corrector_steps):
        inner_fn     = _SAMPLER_FUNCTIONS.get(method, _sample_hydro_euler)
        _eta         = eta
        _momentum    = momentum
        _cfg_rescale = cfg_rescale
        _s_noise     = s_noise
        _corrector   = corrector_steps

        def sampler_fn(model, x, sigmas, extra_args, callback, disable, **kwargs):
            return inner_fn(
                model, x, sigmas, extra_args, callback, disable,
                eta=_eta, s_noise=_s_noise,
                cfg_rescale_factor=_cfg_rescale,
                momentum=_momentum, corrector_steps=_corrector,
                **kwargs,
            )

        return (comfy.samplers.KSAMPLER(sampler_fn),)


# ---------------------------------------------------------------------------
# NODE 3: HydroSharkKSampler
# ---------------------------------------------------------------------------

class HydroSharkKSampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":           ("MODEL",),
                "positive":        ("CONDITIONING",),
                "negative":        ("CONDITIONING",),
                "latent_image":    ("LATENT",),
                "seed":            ("INT",   {"default": 0,   "min": 0,   "max": 0xffffffffffffffff}),
                "steps":           ("INT",   {"default": 9,   "min": 1,   "max": 100,  "step": 1}),
                "cfg":             ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_method":  (["hydro_euler", "hydro_heun2", "hydro_dpm", "hydro_momentum"],),
                "scheduler_mode":  (["flow_linear", "flow_sigmoid", "flow_cosine", "adaptive_blend"],),
                "denoise":         ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "eta":             ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "momentum":        ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "cfg_rescale":     ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "midpoint_bias":   ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "sharpness":       ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "corrector_steps": ("INT",   {"default": 1,   "min": 0,   "max": 3,    "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION     = "sample"
    CATEGORY     = "HydroShark/Sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_method, scheduler_mode, denoise, eta, momentum,
               cfg_rescale, midpoint_bias, sharpness, corrector_steps):

        # Build sigma schedule
        sigmas_full = _build_sigmas(steps, scheduler_mode, midpoint_bias, sharpness)
        sigmas = _apply_denoise(sigmas_full, denoise, steps)
        device = comfy.model_management.get_torch_device()
        sigmas = sigmas.to(device)

        # Build sampler closure
        inner_fn     = _SAMPLER_FUNCTIONS.get(sampler_method, _sample_hydro_euler)
        _eta         = eta
        _momentum    = momentum
        _cfg_rescale = cfg_rescale
        _corrector   = corrector_steps

        def sampler_fn(model_inner, x, sigmas_inner, extra_args, callback, disable, **kwargs):
            return inner_fn(
                model_inner, x, sigmas_inner, extra_args, callback, disable,
                eta=_eta, s_noise=1.0,
                cfg_rescale_factor=_cfg_rescale,
                momentum=_momentum, corrector_steps=_corrector,
                **kwargs,
            )

        sampler = comfy.samplers.KSAMPLER(sampler_fn)

        # Prepare latent and noise.
        # Pass noise separately to sample_custom - it handles adding noise at
        # the correct sigma internally. Do NOT pre-add noise to latent_samples,
        # that would double-noise the input and produce pure static.
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        noise = comfy.sample.prepare_noise(latent_samples, seed)

        # disable_pbar=False so ComfyUI step previews fire correctly.
        out_samples = comfy.sample.sample_custom(
            model,
            noise,
            cfg,
            sampler,
            sigmas,
            positive,
            negative,
            latent_samples,
            noise_mask=latent.get("noise_mask"),
            disable_pbar=False,
            seed=seed,
        )

        out_latent = latent.copy()
        out_latent["samples"] = out_samples
        return (out_latent,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "HydroSharkScheduler": HydroSharkScheduler,
    "HydroSharkSampler":   HydroSharkSampler,
    "HydroSharkKSampler":  HydroSharkKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HydroSharkScheduler": "HydroShark Scheduler",
    "HydroSharkSampler":   "HydroShark Sampler",
    "HydroSharkKSampler":  "HydroShark KSampler",
}
