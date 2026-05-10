# Goodfire Manifold Steering — Educational Demo

Two notebooks illustrating Goodfire's **activation space vs. behavior space** isometry claim, with steering demos and stress tests. Built as a hands-on companion to Max Ma's critique of the Manifold Steering paper.

## Starting Point: Max Ma's Critique

> [Single Token Geometry: A Critique of Manifold Steering](https://deepmanifold.substack.com/p/single-token-geometry-a-critique)

Max Ma argues that Goodfire's "manifold" language is mathematically loose: what they fit are spline curves through empirically observed concept centroids, not true manifolds derived from the model's architecture and boundary conditions. The approximate isometry they report is a local, smooth-region phenomenon — it breaks at concept boundaries ("cracks"), which the spline representation cannot express.

## Goodfire's Original Work

> [Manifold Steering (arXiv:2605.05115)](https://arxiv.org/abs/2605.05115)  
> [Goodfire Research Page](https://www.goodfire.ai/research/manifold-steering)

Goodfire's finding: pairwise distances between concept centroids in **activation space** (final-layer hidden states, PCA-reduced) and **behavior space** (output distributions, Hellinger-embedded, PCA-reduced) are approximately proportional — an empirical approximate isometry. This is used to motivate activation steering as a geometrically grounded intervention.

---

## Notebooks

### `demo_gpt2.ipynb` — GPT-2 (start here)

**Hardware:** Any machine with 8 GB RAM. Runs on CPU. No GPU required.
**Download:** ~500 MB (GPT-2, cached after first run)
**Runtime:** ~5–8 minutes total on CPU

| Section | What it demonstrates |
|---------|----------------------|
| **1 — Synthetic** | What "approximate isometry" means geometrically (no model needed) |
| **2 — Activation Space** | GPT-2 hidden states → PCA → day-of-week concept centroids |
| **3 — Behavior Space** | Output distributions → √p Hellinger embedding → PCA → isometry scatter |
| **4 — Steering Demo** | Hook-based activation steering; Mon→Tue direction; cross-cycle stress test |

**Key result:** Pearson r ≈ 0.42 — weak isometry, far below Goodfire's r > 0.9 on large models. This gap itself supports Max's critique: isometry is not universal.

---

### `demo_llama.ipynb` — Llama-3-8B-Instruct (full experiment)

**Hardware:** 16 GB+ unified memory recommended (M1/M2/M3/M4 Pro or better). Tested on M4 Pro 48 GB.
**Download:** ~16 GB (Llama-3-8B in float16, cached after first run)
**Runtime:** ~15–25 minutes on M-series Mac with MPS
**Prerequisites:** HuggingFace account + accepted [Llama-3 license](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Run `huggingface-cli login` before starting.

| Section | What it demonstrates |
|---------|----------------------|
| **1 — Synthetic** | Same isometry explainer as GPT-2 notebook |
| **2 — Activation Space** | Llama-3 hidden states (4096D) → PCA → day-of-week centroids |
| **3 — Behavior Space** | Hellinger embedding (128k vocab) → PCA → isometry scatter |
| **4 — Steering Demo** | Mon→Tue steering via `model.model.norm` hook |
| **5 — Crack Test** | **Refusal boundary test**: steers a helpful prompt toward refusal-adjacent concept space; measures Hellinger distance discontinuity vs. α — the real test of Max's "crack" hypothesis |

**Expected outcome:** Potentially higher r than GPT-2, plus an empirical test of whether the helpful/refusal boundary in a safety-trained model creates a geometric discontinuity in behavior space.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# GPT-2 version (no login needed)
jupyter notebook demo_gpt2.ipynb

# Llama-3 version (requires HuggingFace login)
huggingface-cli login
jupyter notebook demo_llama.ipynb
```
