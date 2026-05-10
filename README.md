# Goodfire Manifold Steering — Educational Demo

A Jupyter notebook illustrating Goodfire's **activation space vs. behavior space** isometry claim, with a steering demo and stress test. Built as a hands-on companion to Max Ma's critique of the Manifold Steering paper.

## Starting Point: Max Ma's Critique

> [Single Token Geometry: A Critique of Manifold Steering](https://deepmanifold.substack.com/p/single-token-geometry-a-critique)

Max Ma argues that Goodfire's "manifold" language is mathematically loose: what they fit are spline curves through empirically observed concept centroids, not true manifolds derived from the model's architecture and boundary conditions. The approximate isometry they report is a local, smooth-region phenomenon — it breaks at concept boundaries ("cracks"), which the spline representation cannot express.

## Goodfire's Original Work

> [Manifold Steering (arXiv:2605.05115)](https://arxiv.org/abs/2605.05115)  
> [Goodfire Research Page](https://www.goodfire.ai/research/manifold-steering)

Goodfire's finding: pairwise distances between concept centroids in **activation space** (final-layer hidden states, PCA-reduced) and **behavior space** (output distributions, Hellinger-embedded, PCA-reduced) are approximately proportional — an empirical approximate isometry. This is used to motivate activation steering as a geometrically grounded intervention.

## What This Notebook Does

| Section | What it demonstrates |
|---------|----------------------|
| **1 — Synthetic** | What "approximate isometry" means geometrically (before loading any model) |
| **2 — Activation Space** | GPT-2 hidden states → PCA → day-of-week concept centroids |
| **3 — Behavior Space** | Output distributions → √p Hellinger embedding → PCA → isometry scatter |
| **4 — Steering Demo** | Hook-based activation steering; Mon→Tue direction; cross-cycle stress test |

**Model:** GPT-2 (local, ~500 MB, no API key needed)  
**Concepts:** Days of the week — an ordered cyclic set, matching Goodfire's own teaching example

### Key Result

On GPT-2 with weekday concepts, the activation–behavior Pearson r ≈ 0.42 — far weaker than the r > 0.9 Goodfire reports on large models. This gap partially corroborates Max's argument: isometry is not universal, and is substantially weaker on small models and semantically diffuse concept sets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook demo.ipynb
```

Run cells top-to-bottom. Section 2 and 3 each take 1–3 minutes on CPU (first run downloads GPT-2, ~500 MB, then caches).
