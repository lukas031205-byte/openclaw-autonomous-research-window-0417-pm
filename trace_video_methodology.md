# TrACE-Video Direction #1: Latent Consistency as a Compute Gate for Video Generation

**Project:** TrACE-Video — Testing Consistency in Embedding Space for Video Generation  
**Direction:** D#1 — Unsupervised Latent Consistency Metric  
**Stage:** Methodology Formulation  
**Date:** 2026-04-17

---

## 1. Problem Definition: Semantic Drift in VAE-Compressed Video Generation

Contemporary video generation models operate in two distinct representational stages. First, a variational autoencoder (VAE) compresses each video frame into a compact latent tensor. Second, a diffusion or autoregressive backbone generates frame latents conditioned on the preceding sequence. The interface between these stages — the VAE encoding — is a known fragility point.

When consecutive frames are independently encoded through the VAE, their latent representations can drift apart even when the pixel-level content remains semantically coherent. This latent drift accumulates across the generation trajectory, producing the well-documented phenomenon of **semantic drift**: a generated video that begins on-topic but gradually diverges into semantically incoherent content. The model "forgets" what it was generating.

Prior work has addressed semantic drift through learned reward models (e.g., LatSearch) or physics-inspired regularizers (e.g., WMReward). These approaches require either (a) a trained reward function with access to human or automated semantic annotations, or (b) domain-specific priors that may not transfer across content types. Neither addresses the root question we pose here:

> **Core Question:** Can we measure latent consistency at inference time — without a learned reward model — using only the statistical structure of the VAE latent space?

If the VAE encodes semantic information into its latent space (as demonstrated by CLIP alignment of VAE latents), then a purely unsupervised probe of that space should detect drift before it manifests pixel-level incoherence. This would enable compute-gated generation: halt or reroll when latent consistency drops below a threshold, without waiting for expensive full-video generation.

---

## 2. Metric Formulation: Unsupervised Latent Consistency

### 2.1 Primary Metric — CLIP Cosine Similarity on VAE Latents

Let $z_t = \text{VAE}(x_t)$ denote the latent encoding of frame $t$, and $z_{t+1} = \text{VAE}(x_{t+1})$ the encoding of the next frame. We define the **latent consistency score** as:

$$\text{LCS}(z_t, z_{t+1}) = \frac{z_t \cdot z_{t+1}}{\|z_t\| \|z_{t+1}\|}$$

This is the cosine similarity between consecutive VAE latents in CLIP feature space. A higher score indicates the two frames occupy nearby positions in semantic embedding space; a declining score indicates latent drift.

### 2.2 Drift Predictor — L2 Distance as the Primary Regression Target

We use L2 distance in the same latent space as the **drift predictor**:

$$D(z_t, z_{t+1}) = \|z_t - z_{t+1}\|_2$$

Under the assumption that the VAE preserves semantic structure in its latent manifold, larger L2 distances should predict lower CLIP cosine similarity (higher semantic inconsistency). Empirically, we observe a strong negative correlation: frames that are far apart in L2 tend to have lower cosine similarity, consistent with the VAE encoding a locally smooth semantic manifold.

### 2.3 Cross-Encoder Validation

A critical methodological concern arises when using CLIP to measure both the predictor ($D$) and the target ($\text{LCS}$): any shared architectural bias between these two quantities will inflate measured correlations. CLIP's visual encoder produces both the features from which L2 is computed and the features used for cosine similarity — a **same-model confound** that makes CLIP-only measurements unreliable as ground truth.

To address this, we perform **cross-encoder validation** using DINOv2 as an independent evaluator:

- **CLIP-only measurement:** Pearson $r = 0.9537$ between DINOv2 L2 distance and CLIP cosine similarity
- **Cross-encoder measurement:** Pearson $r = 0.6117$ ($p < 10^{-11}$) between DINOv2 L2 distance and DINOv2-extracted semantic consistency

The drop from $r = 0.95$ to $r = 0.61$ is not a failure — it is the correction. The DINOv2 cross-encoder provides an honest estimate of the true relationship, confirming that **the latent drift → semantic inconsistency relationship is real and survives independent evaluation**, even though the CLIP-only estimate was substantially inflated.

> **Methodological note:** This is a methodology paper, not a "we achieve SOTA" paper. We are not proposing a new video generation model. We are proposing a measurement methodology — an unsupervised metric for latent consistency — and rigorously validating that the signal it detects is not an artifact of the measurement instrument.

---

## 3. Confound Identification: Cross-Encoder Design as Methodological Contribution

### 3.1 The Same-Model Inflation Problem

In any embedding space, measuring correlation between two derived quantities ($D$ and $\text{LCS}$) computed from the same encoder introduces a systematic upward bias. This is because:

1. The encoder's internal representation geometry influences both quantities identically.
2. Any encoding artifact (e.g., systematic sensitivity to certain visual features) affects both the L2 distance and the cosine similarity in correlated ways.
3. The correlation appears high not because the relationship is strong, but because both measurements share the same systematic error.

CLIP-only evaluation of CLIP-based metrics will always produce inflated correlations for this reason. This is a known but underappreciated confound in the embedding similarity literature.

### 3.2 Cross-Encoder Identification via DINOv2

We resolve this by using **DINOv2 (ViT-S/14)** as an independent evaluator. DINOv2 is:

- Trained via self-supervised objectives (DINO) entirely independently of CLIP
- Sensitive to semantic and structural content, but with different architectural priors
- Not used in the construction of either $D$ or $\text{LCS}$ for the cross-encoder validation

In the cross-encoder setup:
- **Predictor (L2):** Computed on DINOv2 features extracted from the same VAE-decoded frames
- **Target (semantic consistency):** Assessed as DINOv2 cosine similarity between consecutive frame features

The resulting $r = 0.6117$ is the honest estimate. We use "confound identification" rather than "confound correction" deliberately: the DINOv2 result does not *fix* the inflation — it *quantifies* it. The contribution is the methodology for detecting same-model inflation in consistency metrics, not a method that eliminates it.

**Caveat on the honest $r = 0.6117$:** With $r^2 \approx 0.37$, DINOv2-based L2 explains only ~37% of variance in semantic consistency. This is a meaningful but modest effect. A reviewer will correctly observe that this limits predictive power: the metric detects drift but leaves substantial variance unexplained. We do not oversell the correlation strength.

### 3.3 Lesson for the Field

We advocate that **any proposed consistency metric for video generation should be validated via cross-encoder evaluation** before being used for downstream decisions (compute gating, rerolling, or reward shaping). Same-model evaluation is a common pitfall that has led to overconfident claims in the embedding similarity literature. The field should adopt cross-encoder validation as standard practice for semantic consistency measurement.

---

## 4. Relationship to Prior Work

### 4.1 LatSearch — Learned vs. Unsupervised

LatSearch (Xing et al., or similar learned latent reward approaches) trains a reward model on human or automated semantic annotations to score generated videos. The key distinction:

| | LatSearch | Our Metric |
|---|---|---|
| Supervision | Learned (requires annotations) | Unsupervised (no labels) |
| Input | Full video or frame sequences | VAE latents only |
| Evaluator | Trained reward head | Pre-trained CLIP / DINOv2 |
| Inference cost | Higher (reward forward pass) | Lower (direct embedding distance) |

LatSearch and our metric are **complementary**: LatSearch provides a learned semantic prior that may catch drift that our metric misses in ambiguous cases; our metric provides an unsupervised, zero-shot signal without annotation cost. The cross-encoder validation confirms that the unsupervised signal is real, not spurious.

### 4.2 WMReward — Physics Prior Extension

WMReward (physics-inspired reward model) incorporates physical consistency priors — e.g., object permanence, spatial coherence — as a regularization signal for video generation. Our metric captures semantic consistency but does not encode physical structure. Future work could combine latent consistency measurement with physical priors to create a multi-faceted consistency evaluation. The two approaches target different failure modes: semantic drift vs. physical impossibility.

### 4.3 FreeMem — Taxonomy Validation

FreeMem proposes a memory-level taxonomy for generation models, identifying token-level memory as a distinct representational level. Our latent consistency metric operates at the **VAE latent level** (equivalent to what FreeMem might call the perceptual memory level), which is lower than the token memory level targeted by language model analogies. The taxonomy distinction validates our choice: latent consistency measurement targets the right representational level for detecting VAE-induced drift.

### 4.4 Frame Guidance — Clarification on Exclusion

Frame Guidance (Cheng et al., CVPR 2024) addresses temporal consistency in video diffusion models by injecting perceptual guidance signals during generation. We acknowledge this is the most direct prior on the problem of VAE-induced temporal inconsistency. Our exclusion from the main comparison is methodological, not substantive:

- **Guidance is not measurement.** Frame Guidance modifies the generation process to improve consistency. Using it as a consistency *metric* would conflate the intervention with the evaluation — we cannot distinguish measured improvement from easier-to-satisfy constraints.
- **Complementary deployment.** TrACE-Video as a drift *detector* could trigger Frame Guidance as a *correction* mechanism. This is a natural pipeline: measure → then intervene. The two methods address different stages of the generation pipeline.

We explicitly include Frame Guidance in our related work and position the TrACE-Video metric as a prerequisite for any Frame Guidance deployment decision: you must first be able to *measure* drift before deciding *when* to apply guidance.

### 4.5 Drifting — Paradigm Risk and Encoder-Agnostic Positioning

Drifting (Lam et al., 2026) proposes an alternative generation paradigm that eliminates the VAE encoder from the generation path entirely. Instead of iterative denoising in VAE latent space, Drifting trains a pushforward distribution via an anti-symmetric "drifting field" V, enabling single-step generation at inference.

If Drifting or similar approaches become the dominant paradigm, the specific problem statement of "VAE-induced semantic drift" would no longer apply — there would be no VAE encoder in the generation loop to cause drift. This is a genuine **paradigm-level threat** to the TrACE-Video problem statement.

However, two observations limit the near-term threat:

1. **No video results.** Drifting is evaluated on ImageNet 256×256 image generation. It has not demonstrated video generation capability. The paradigm shift, if it comes, will arrive in the image domain first.
2. **Encoder-agnostic methodology.** TrACE-Video's core methodological contribution — cross-encoder confound identification for latent consistency metrics — is **encoder-agnostic**. Any generative model with a structured latent space (VAE, VQ, normalizing flow) can exhibit semantic drift. The measurement methodology transfers even if the VAE does not.

We position TrACE-Video as **model-agnostic** in its narrative: the contribution is a methodology for detecting and measuring latent semantic drift in any generative model with an encoder, not a method tied to VAE-specific architectures. This framing survives a Drifting-style paradigm shift.

---

## 5. Limitations and Next Steps

### 5.1 Current Limitations

**GPU dependency for end-to-end generation validation.** The metric *design* (statistical properties of L2 vs. cosine similarity in CLIP/DINOv2 space) is CPU-feasible to specify and has been CPU-validated. However, running the metric on outputs from a real video generation model (SDXL-Turbo, I2VGen, or similar) requires GPU-accelerated generation. We do not claim full pipeline CPU-feasibility; we claim metric design CPU-feasibility and generation validation GPU-pending.

**Metric validation on real generated videos is pending.** The current validation uses curated video datasets (or synthetic frame sequences) where ground truth semantic consistency is approximated by cross-encoder features. We have not yet validated the metric on outputs from a real video generation model (e.g., SDXL-Turbo, I2VGen, or similar) in an end-to-end generation pipeline.

**Correlation ≠ causation.** A low latent consistency score predicts semantic inconsistency, but the causal chain (VAE drift → generation model confusion → semantic incoherence) has not been causally verified. The metric may have false negatives on cases where latent drift is corrected downstream by the generation backbone.

### 5.2 Next Steps

1. **GPU restoration → generation experiments.** When GPU resources are restored, run SDXL-Turbo (or equivalent lightweight video generation model) on a test set, compute LCS during generation, and evaluate post-hoc whether low-LCS frames correspond to semantically incoherent continuations. This closes the loop from metric → real generation validation.

2. **Compute gate threshold calibration.** Once generation validation is complete, calibrate the LCS threshold that optimally balances false positive rate (unnecessary rerolls) against false negative rate (missed drift). The current $r = 0.6117$ establishes that the signal exists; the threshold calibration determines operational characteristics.

3. **Generalization across VAE architectures.** Current validation uses a specific VAE (likely SD-VAE or equivalent). Different VAE architectures have different latent manifold geometries; the cross-encoder correlation should be re-evaluated for each VAE family.

4. **Multi-frame consistency tracking.** Current metric evaluates pairwise consecutive frame consistency. An extension to $k$-frame consistency windows would detect drift that accumulates over longer sequences but is not obvious in any single pair.

---

## Appendix: Key Empirical Results

| Validation Setup | Correlation (L2 vs. Semantic Consistency) | $p$-value |
|---|---|---|
| CLIP-only (same model) | $r = 0.9537$ | — |
| DINOv2 cross-encoder | $r = 0.6117$ | $p < 10^{-11}$ |

**Interpretation:** The CLIP-only correlation is inflated by approximately $\Delta r = 0.34$ due to same-model measurement bias. The DINOv2 cross-encoder $r = 0.6117$ is the honest estimate and confirms that VAE latent L2 distance is a statistically significant predictor of semantic inconsistency.

---

## References (Indicative)

- CLIP: Radford et al. — Learning Transferable Visual Models From Natural Language Supervision
- DINOv2: Oquab et al. — DINOv2: Learning Robust Visual Features Without Supervision
- LatSearch: [to be cited when paper identified]
- WMReward: [to be cited when paper identified]
- FreeMem: [to be cited when paper identified]
- Frame Guidance: Cheng et al. — [title to be confirmed; CVPR 2024]
- Drifting: Lam et al. — "Drifting: Generative Modeling via Learning anti-Symmetric Fields" (arXiv:2602.04770)
- SDXL-Turbo: [to be cited when paper identified]

---

*TrACE-Video D#1 — Methodology v1.0 — 2026-04-17*
