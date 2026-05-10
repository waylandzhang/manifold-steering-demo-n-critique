# Manifold Steering — Isometry Reproduction and Boundary Analysis

Replication of the core geometric claim in Goodfire's *Manifold Steering* (arXiv:2605.05115), with additional experiments probing isometry at semantic and register boundaries.

## Background

> [Manifold Steering (arXiv:2605.05115)](https://arxiv.org/abs/2605.05115) · [Goodfire Research](https://www.goodfire.ai/research/manifold-steering)
>
> [Single Token Geometry: A Critique of Manifold Steering — Max Ma](https://deepmanifold.substack.com/p/single-token-geometry-a-critique)

Goodfire's central claim: pairwise distances between concept centroids in **activation space** and **behavior space** are approximately proportional — an empirical isometry. This is used to motivate activation steering as a geometrically grounded intervention.

## Method

Follows Goodfire §2.1–2.3 exactly:

| Choice | This work | Goodfire |
|--------|-----------|---------|
| Model | Llama-3.1-8B (base) | Llama-3.1-8B |
| Layer | 28 of 32 | 28 |
| Prompts | "What day is k days after z? Answer:" | Same |
| Activation space | Hidden states, 64D PCA | 64D PCA |
| Behavior space | √p Hellinger embedding, 64D PCA | Same |
| Distance metric | Cubic spline arc length | Same |

## Results

### Isometry Reproduction

| Concept set | Chord r | Spline arc r | Goodfire (spline) |
|-------------|---------|-------------|-------------------|
| Days of week | 0.854 | **0.994** | 0.99 |
| Months of year | — | **0.965** | 0.89 |

Spline arc length is essential: chord distance alone gives r = 0.854 for days; the spline recovers r = 0.994, matching Goodfire's reported result. Layer 28 confirmed as the peak by a full layer sweep (layers 1–32).

### Layer Sweep (Days, Chord r)

| Layer | 1 | 8 | 16 | 24 | **28** | 30 | 32 |
|-------|---|---|----|----|--------|----|----|
| r | 0.51 | 0.38 | 0.65 | 0.73 | **0.85** | 0.81 | 0.83 |

### Boundary Analysis (Sections 5–6)

#### Section 5 — Cyclic Ordinal Concepts

Three experiments on whether isometry degrades at semantic boundaries within cyclic ordinal concept sets (weekdays, months).

**5.1 — Weekday/weekend boundary (local residuals)**

| Pair type | Mean residual | n |
|-----------|--------------|---|
| Weekday–weekday | 0.0455 | 10 |
| Weekend–weekend | 0.0396 | 1 |
| Cross-boundary (e.g. Fri–Sat) | 0.0331 | 10 |

Cross-boundary residual ratio vs. within-weekday: **0.73** — not elevated. Isometry holds uniformly across the weekday/weekend boundary.

**5.2 — Year boundary (months)**

Dec–Jan boundary pairs vs. all other month pairs: residual ratio ≈ **1.04**. No significant elevation.

**5.3 — Cross-concept steering (days → months)**

Steering a day prompt along the days → months centroid direction (α 0–30, step 1):

| α | Top token | P |
|---|-----------|---|
| 0 | Monday | 0.81 |
| 10 | Monday | 0.32, February 0.06 |
| 20 | February | 0.14, December 0.13 |
| 30 | December | 0.16, February 0.13 |

Behavior shifts smoothly from day to month vocabulary, then decelerates (ratio 0.35). No discontinuous jump.

#### Section 6 — Language Register / Code-Switching

Three concept clusters: English prose, Python code, French prose (15 prompts each). This targets a harder distributional boundary — code and natural language rarely interleave in training data.

**6.1 — Cross-register isometry**

| Pair | d\_activation | d\_behavior | d\_beh / d\_act |
|------|--------------|------------|-----------------|
| English ↔ Python | 20.1 | 0.544 | 0.027 |
| English ↔ French | 25.2 | 0.718 | 0.029 |
| Python ↔ French | 29.3 | 0.669 | 0.023 |

Ratio coefficient of variation: **9.1%** — isometry holds across all three pairs, including the hard English/Python boundary. Notably, Python is closer to English than French in both spaces.

**6.2 — Fine-grained steering: English → Python vs. English → French (α 0–40, step 0.5)**

| Direction | Boundary | Peak / median | Total displacement |
|-----------|----------|--------------|-------------------|
| English → French | soft | **2.00×** | 1.116 |
| English → Python | hard | 1.15× | 0.806 |

Token trajectories reveal a qualitative difference:

*English → French:* French function words enter top-20 at α = 15.5 and dominate by α = 30–40 (`la` 22%, `les` 9%, `le` 6%). The language switch completes.

*English → Python:* Python keywords never enter top-20 at any α. Vocabulary drifts within English scientific register (`effect` → `rate` → `value` → `x`).

**The steering direction toward Python code is not admissible.** Python production requires the entire forward pass to be in a code context — attention patterns, KV states, and MLP activations from all preceding layers process the English input before the layer-28 modification. A centroid shift at one layer cannot retroactively establish that context. By contrast, English and French share the same deep syntactic structure, so a layer-28 direction shift is sufficient to redirect the output vocabulary.

This illustrates a case where activation space moves but behavior space does not follow — not as a Hellinger spike, but as a complete non-response of the behavior space to the steering direction.

## Setup

```bash
# Requires HuggingFace account + accepted Llama-3 license
huggingface-cli login

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook demo_llama.ipynb
```

**Hardware:** 16 GB+ unified memory (tested on M4 Pro 48 GB). Runtime: ~40–50 min on MPS.
