# HydroSharkSampler

A ComfyUI custom node package providing advanced sampling and scheduling primitives
optimized for **Z-Image-Turbo** (rectified-flow / CONST diffusion) and compatible
with **RES4LYF ClownsharkSampler** pipelines.

---

## Table of Contents

1. [Why HydroSharkSampler?](#why-hydrosharksampler)
2. [Installation](#installation)
3. [Nodes Overview](#nodes-overview)
4. [Node Reference](#node-reference)
   - [HydroShark Scheduler](#hydroshark-scheduler)
   - [HydroShark Sampler](#hydroshark-sampler)
   - [HydroShark KSampler](#hydroshark-ksampler)
5. [Recommended Settings](#recommended-settings)
   - [Z-Image-Turbo (standalone)](#z-image-turbo-standalone)
   - [With RES4LYF ClownsharkSampler](#with-res4lyf-clownsharksampler)
6. [Workflow Examples](#workflow-examples)
7. [Parameter Deep-Dive](#parameter-deep-dive)
8. [Tips and Tuning Notes](#tips-and-tuning-notes)
9. [Troubleshooting](#troubleshooting)

---

## Why HydroSharkSampler?

Z-Image-Turbo is a rectified-flow model trained with a CONST noise schedule
(linear sigmas from 1.0 to 0.0). Stock ComfyUI schedulers and samplers are
designed for DDPM/EDM models and apply sub-optimal step distributions and
denoising logic when used with flow models.

**HydroSharkSampler addresses four failure modes:**

| Failure mode | HydroShark solution |
|---|---|
| Over-saturated colors at low CFG (1.0) | Per-step CFG std-rescale (`cfg_rescale`) |
| Harsh step boundaries with large sigma gaps | Adaptive flow schedules (sigmoid, cosine, blend) with tunable `midpoint_bias` |
| Blurry/oversmoothed detail in few-step generation | Front-loaded schedule bias (`midpoint_bias` < 0.5) |
| First-order Euler accumulation error | 2nd-order Heun corrector (`hydro_heun2` + `corrector_steps=1`) |

The package ships three nodes covering both modular and all-in-one workflows.

---

## Installation

### Option A — Drop-in (recommended)

```
ComfyUI/
  custom_nodes/
    HydroSharkSampler/      <-- clone or copy here
      __init__.py
      nodes.py
      README.md
```

1. Clone or download this repository.
2. Copy the `HydroSharkSampler/` folder into your `ComfyUI/custom_nodes/` directory.
3. Restart ComfyUI (or use the Manager "Restart" button).
4. The three nodes appear under the **HydroShark** category in the node search.

### Option B — Git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-repo/HydroSharkSampler
```

### Requirements

- ComfyUI (any recent version with `comfy.samplers.KSAMPLER` and `comfy.sample.sample_custom`)
- Python >= 3.10
- PyTorch >= 2.0
- `tqdm` (included with ComfyUI)

No additional pip packages are required.

---

## Nodes Overview

| Node | Category | Output | Purpose |
|---|---|---|---|
| **HydroShark Scheduler** | HydroShark/Scheduling | SIGMAS | Generates flow-optimized sigma schedules |
| **HydroShark Sampler** | HydroShark/Sampling | SAMPLER | Pure sampler for use with `SamplerCustom` |
| **HydroShark KSampler** | HydroShark/Sampling | LATENT | All-in-one convenience node |

---

## Node Reference

### HydroShark Scheduler

Generates a sigma schedule tailored for rectified-flow / CONST models.
Connect its `SIGMAS` output to the `sigmas` input of ComfyUI's built-in
**SamplerCustom** node (alongside a HydroShark Sampler or any other SAMPLER).

#### Inputs

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | The loaded model (used for device resolution) |
| `steps` | INT | 9 | 1–100 | Number of denoising steps |
| `denoise` | FLOAT | 1.0 | 0.0–1.0 | Denoising strength. 1.0 = full generation from noise |
| `midpoint_bias` | FLOAT | 0.4 | 0.0–1.0 | Shifts the curve center. < 0.5 front-loads detail steps, > 0.5 back-loads structure steps |
| `sharpness` | FLOAT | 6.0 | 1.0–20.0 | Controls the steepness of the sigmoid transition |
| `mode` | COMBO | flow_linear | see below | Schedule curve type |

#### Schedule Modes

| Mode | Behavior | Best for |
|---|---|---|
| `flow_linear` | Standard linear 1.0 → 0.0. Baseline for turbo models. | Default / reference |
| `flow_sigmoid` | Sigmoid remapped to [0,1]. Slower at extremes, faster in the middle. | Reducing banding at large step sizes |
| `flow_cosine` | Cosine curve warped by `midpoint_bias`. Smooth acceleration/deceleration. | Detail-rich scenes with moderate step counts |
| `adaptive_blend` | Weighted blend of linear and sigmoid controlled by `midpoint_bias`. | Fine-grained control over pacing |

#### Output

- `sigmas` — SIGMAS tensor of length `(effective_steps + 1)`, ending with 0.0

---

### HydroShark Sampler

A pure **SAMPLER** node. Connect its output to the `sampler` input of
ComfyUI's **SamplerCustom** node. Pair with **HydroShark Scheduler** or
any other sigma source.

#### Inputs

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `method` | COMBO | hydro_euler | see below | Core ODE solver |
| `eta` | FLOAT | 0.0 | 0.0–1.0 | Stochastic noise injection. 0 = fully deterministic |
| `momentum` | FLOAT | 0.0 | 0.0–0.99 | Velocity EMA momentum across steps |
| `cfg_rescale` | FLOAT | 0.7 | 0.0–1.0 | Per-step CFG std-rescale strength |
| `s_noise` | FLOAT | 1.0 | 0.5–2.0 | Noise amplitude scaling for stochastic steps |
| `corrector_steps` | INT | 1 | 0–3 | Number of Heun corrector passes (ignored by euler/dpm/momentum methods) |

#### Solver Methods

| Method | Order | Description |
|---|---|---|
| `hydro_euler` | 1st | Fast first-order Euler with CFG rescale. Best for 8–9 step turbo workflows. |
| `hydro_heun2` | 2nd | Heun predictor-corrector. Uses `corrector_steps` Heun passes and optional momentum on the velocity delta. Best quality/speed tradeoff. |
| `hydro_dpm` | 2nd | DPM-Solver++(2M) adapted for flow matching via linear extrapolation of denoised estimates. Good for > 12 steps. |
| `hydro_momentum` | 1st+ | Euler with velocity EMA accumulation. Produces smooth, coherent trajectories. Good for style-heavy prompts. |

#### Output

- `sampler` — SAMPLER object (pass to SamplerCustom)

---

### HydroShark KSampler

An all-in-one convenience node equivalent to wiring **HydroShark Scheduler** +
**HydroShark Sampler** + **SamplerCustom** together. Best for simple workflows
where you want a single node to drop in.

#### Inputs

All inputs from HydroShark Scheduler and HydroShark Sampler, plus:

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `model` | MODEL | — | — | Loaded model |
| `positive` | CONDITIONING | — | — | Positive conditioning |
| `negative` | CONDITIONING | — | — | Negative conditioning |
| `latent_image` | LATENT | — | — | Input latent (empty for txt2img) |
| `seed` | INT | 0 | 0–2^64 | RNG seed for reproducibility |
| `steps` | INT | 9 | 1–100 | Total sampling steps |
| `cfg` | FLOAT | 1.0 | 0.0–30.0 | Classifier-Free Guidance scale |
| `sampler_method` | COMBO | hydro_euler | — | Solver (same as HydroShark Sampler) |
| `scheduler_mode` | COMBO | flow_linear | — | Schedule (same as HydroShark Scheduler) |
| `denoise` | FLOAT | 1.0 | 0.0–1.0 | Denoising strength |
| `eta` | FLOAT | 0.0 | 0.0–1.0 | Stochastic noise injection |
| `momentum` | FLOAT | 0.0 | 0.0–0.99 | Velocity EMA momentum |
| `cfg_rescale` | FLOAT | 0.7 | 0.0–1.0 | Per-step CFG std-rescale |
| `midpoint_bias` | FLOAT | 0.4 | 0.0–1.0 | Schedule midpoint shift |
| `sharpness` | FLOAT | 6.0 | 1.0–20.0 | Sigmoid sharpness |
| `corrector_steps` | INT | 1 | 0–3 | Heun corrector passes |

#### Output

- `latent` — LATENT with sampled result

---

## Recommended Settings

### Z-Image-Turbo (standalone)

Z-Image-Turbo is a CONST rectified-flow model trained for fast 8–9 step
generation at CFG=1.0 with linear sigmas.

```
steps           = 9
cfg             = 1.0
sampler_method  = hydro_heun2
scheduler_mode  = flow_linear   (or adaptive_blend for more flexibility)
denoise         = 1.0
eta             = 0.0           (deterministic for reproducibility)
momentum        = 0.0
cfg_rescale     = 0.7
midpoint_bias   = 0.4
sharpness       = 6.0
corrector_steps = 1
```

**Quick quality boost** — switch `sampler_method` to `hydro_heun2` with
`corrector_steps=1`. This doubles the model calls per step but produces
noticeably cleaner detail with zero extra configuration.

**Speed priority** — use `hydro_euler` with `corrector_steps=0`.
Same step count, one model call per step. Slightly lower quality but
fast enough for iteration.

**Stylised / painterly output** — try `hydro_momentum` with `momentum=0.3`
and `scheduler_mode=flow_cosine`. The velocity EMA softens hard edges
and produces more cohesive tonality.

### With RES4LYF ClownsharkSampler

HydroSharkSampler complements ClownsharkSampler pipelines. A typical
hybrid setup:

1. Use **HydroShark Scheduler** to produce the sigma schedule and feed
   it into ClownsharkSampler's sigma input.
2. Use ClownsharkSampler's built-in sampler for the main pass, then
   refine with **HydroShark Sampler** on a hi-res fix / img2img pass
   at `denoise=0.4–0.6`.

```
# Sigma schedule for ClownsharkSampler
HydroShark Scheduler:
  steps           = 20
  mode            = adaptive_blend
  midpoint_bias   = 0.45
  sharpness       = 5.0
  denoise         = 1.0

# Hi-res fix pass with HydroShark Sampler
HydroShark Sampler:
  method          = hydro_heun2
  eta             = 0.05
  cfg_rescale     = 0.6
  corrector_steps = 1
  momentum        = 0.0
```

**Note:** When pairing with ClownsharkSampler, set `eta=0.0` on
HydroShark Sampler if ClownsharkSampler is handling stochastic injection
to avoid double-adding noise.

---

## Workflow Examples

### Workflow A — Modular (recommended for advanced users)

```
[Load Checkpoint]
       |
       v
[CLIP Text Encode (positive)] --> [SamplerCustom]
[CLIP Text Encode (negative)] --> [SamplerCustom]
[Empty Latent Image]          --> [SamplerCustom]
                                         ^
                                         |
              [HydroShark Scheduler] ----+---- sigmas
              [HydroShark Sampler]   ----+---- sampler
                                         |
                                         v
                                   [VAE Decode]
                                         |
                                         v
                                   [Save Image]
```

This gives you full control: swap the sigma schedule independently from
the sampler, or mix HydroShark Scheduler with other sampler nodes.

### Workflow B — All-in-one (quick start)

```
[Load Checkpoint] -------> [HydroShark KSampler] --> [VAE Decode] --> [Save Image]
[CLIP Text Encode pos] -/
[CLIP Text Encode neg] -/
[Empty Latent Image]   -/
```

Drop in HydroShark KSampler exactly where you would normally place a
KSampler or KSamplerAdvanced node. All parameters are on one node.

### Workflow C — HydroShark Scheduler + ClownsharkSampler

```
[Load Checkpoint]
       |
       v
[HydroShark Scheduler]  -->  sigmas  -->  [ClownsharkSampler]
[CLIP Text Encode pos]  ----------------> [ClownsharkSampler]
[CLIP Text Encode neg]  ----------------> [ClownsharkSampler]
[Empty Latent Image]    ----------------> [ClownsharkSampler]
                                                   |
                                                   v
                                            [VAE Decode]
```

---

## Parameter Deep-Dive

### `cfg_rescale` — Why it matters at CFG=1.0

Rectified-flow turbo models are distilled to work at CFG=1.0. However,
the model output at each step can still have a very different standard
deviation from the noisy input, causing the latent to drift toward
high-saturation, over-contrasted outputs.

CFG std-rescale corrects this by matching the output's channel-wise
standard deviation to that of the current noisy latent:

```
rescaled = cfg_out * (std(x) / (std(cfg_out) + eps))
final    = alpha * rescaled + (1 - alpha) * cfg_out
```

At `cfg_rescale=0.7`, 70% of the correction is applied. This reliably
prevents blown-out highlights and colour oversaturation without softening
detail. Effective range for Z-Image-Turbo: **0.5–0.8**.

Set `cfg_rescale=0.0` to disable entirely (raw model output).

---

### `midpoint_bias` — The 0.35–0.45 sweet spot for turbo

For 8–9 step turbo generation the sigma jump between steps is large
(roughly 0.12 per step with linear spacing). This means each step must
cover a lot of "trajectory distance" and errors accumulate quickly.

Setting `midpoint_bias` below 0.5 shifts the sigmoid/cosine curve so
that:
- **Early steps** (high sigma, coarse structure) get **smaller** sigma
  increments — more careful traversal of the noisy regime.
- **Late steps** (low sigma, fine detail) get **larger** increments —
  detail refinement converges quickly.

The sweet spot **0.35–0.45** balances:
- Enough early-step precision to avoid structural artifacts.
- Enough late-step speed to complete within the budget.

Above 0.5, the schedule back-loads: early steps are coarse but fast,
late steps are fine-grained. This suits img2img refinement at
`denoise < 0.6` where coarse structure is already present.

---

### `hydro_heun2` + `corrector_steps=1` — Quality/speed sweet spot

Heun's method is a 2nd-order Runge-Kutta predictor-corrector:

1. **Predictor**: Euler step from sigma_i to sigma_{i+1} → x_pred
2. **Corrector**: Evaluate model at x_pred, average the two velocity
   estimates, take step from x → x_corr

This costs **2× the model calls** per step but reduces the local
truncation error from O(h^2) (Euler) to O(h^3) (Heun), where h is
the step size.

For Z-Image-Turbo at 9 steps:
- Euler (9 steps): 9 model evaluations
- Heun `corrector_steps=1` (9 steps): 18 model evaluations
- Heun `corrector_steps=2` (9 steps): 27 model evaluations

`corrector_steps=1` gives the bulk of the quality improvement (the
predictor is the main source of error). Going to 2 corrector steps
yields marginal gains at significant compute cost.

For fastest iteration use `hydro_euler` (`corrector_steps` is ignored).
For final renders use `hydro_heun2` with `corrector_steps=1`.

---

### `eta` and `s_noise` — When to use stochastic sampling

By default (`eta=0.0`) all HydroShark methods are **deterministic**:
given the same seed you get identical output. This is ideal for
reproducible experiments.

Setting `eta > 0` injects Gaussian noise at each step:

```
x = x + eta * sigma_next * randn_like(x) * s_noise
```

This can help escape local optima in complex prompts and adds subtle
textural variety. Effective range: **0.0–0.15** for turbo models.
Higher values cause quality degradation at low step counts.

`s_noise` scales the noise amplitude. Values > 1.0 amplify variance
(useful for high-diversity batches). Values < 1.0 reduce it.

---

### `momentum` — Smooth trajectories

The EMA momentum parameter accumulates velocity across steps:

```
v_ema = m * v_ema_prev + (1 - m) * v_current
```

This dampens high-frequency oscillations in the velocity field that can
cause fine-grained noise artifacts in large-step generation.

Recommended range: **0.0–0.4**. Higher values (> 0.5) can cause the
trajectory to lag behind the true score and produce blurring.

Momentum has the most visible effect with `hydro_momentum` (where it
is the primary mechanism) and secondary effect with `hydro_heun2`
(where it modifies the predictor velocity before the corrector pass).

---

## Tips and Tuning Notes

- **Start with defaults**: `hydro_heun2`, `flow_linear`, `cfg_rescale=0.7`,
  `midpoint_bias=0.4`, `corrector_steps=1`. This is the best general-purpose
  starting point for Z-Image-Turbo.

- **Blurry output?** Reduce `midpoint_bias` to 0.3 or switch from
  `flow_cosine` to `flow_linear`. Lower `momentum` if non-zero.

- **Oversaturated/neon colors?** Increase `cfg_rescale` toward 0.8–0.9.

- **Structural artifacts / anatomical errors?** Try `flow_sigmoid` with
  `sharpness=4.0` and `midpoint_bias=0.35`. This gives the most careful
  early-step traversal.

- **Slow generation?** Switch to `hydro_euler` (`corrector_steps` has no
  effect and is ignored). Consider reducing `steps` to 7–8; turbo
  models degrade gracefully.

- **img2img / inpainting?** Set `denoise=0.5–0.7` and
  `scheduler_mode=adaptive_blend` with `midpoint_bias=0.55` to
  back-load the schedule (structure is already present).

- **Pairing with ClownsharkSampler?** Use HydroShark Scheduler only
  (for sigma generation) and let ClownsharkSampler handle the ODE solve.
  Recommended: `mode=adaptive_blend`, `midpoint_bias=0.45`,
  `sharpness=5.0`.

---

## Troubleshooting

**Node not appearing in ComfyUI:**
- Confirm the folder is directly inside `custom_nodes/` (not nested deeper).
- Check the ComfyUI console for import errors on startup.
- Ensure `__init__.py` is present in the folder.

**`AttributeError: module 'comfy.sample' has no attribute 'sample_custom'`:**
- Update ComfyUI to a recent version. `sample_custom` was added in mid-2024.

**`TypeError` on KSAMPLER call:**
- Some older ComfyUI builds expect a different KSAMPLER constructor signature.
  Update ComfyUI or open an issue with the traceback.

**Black / all-noise output:**
- Ensure the model is a flow model (Z-Image-Turbo, Stable Diffusion 3,
  FLUX, etc.). DDPM models require different sigma ranges.
- Verify `cfg=1.0` for turbo models — higher CFG can destabilize them.

**Out of memory:**
- `hydro_heun2` with `corrector_steps=2` requires 3× the VRAM bandwidth
  of a single Euler step due to multiple model evaluations. Reduce
  `corrector_steps` to 1 or use `hydro_euler`.

---

## License

MIT License. Free for personal and commercial use. Attribution appreciated.
