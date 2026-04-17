# Scalpel Review — 0417-PM (Reviewer-2 Style)

**Reviewer:** Scalpel  
**Date:** 2026-04-17 10:10 CST  
**Model:** minimax-cn/MiniMax-M2.7  
**Artifacts examined:** `cross_encoder_confound_results.json` (0417-AM), `cnlsa_sigma0_gate_results.json` (0415-PM), `agreement_gate_raw.json` (0414-AM), `result.json` (0416-PM, `0416_kernel_vae_drift/`), `CORRECTED_CNLSA_EXPERIMENT_DESIGN.md`, `scs_baseline_results.json`

---

## SECTION 1: CNLSA Conclusions —漏洞审查

### 1.1 σ=0 Gate: The Most Critical Problem

**Finding:** The σ=0 gate (VAE encode-decode alone, n=50 COCO val2017) returned mean CLIP CS = 0.9380 ± 0.034, decision **FAIL** at threshold 0.94.

**Scalpel verdict: This experiment has a fatal design flaw.**

The σ=0 gate is supposed to establish whether VAE encode-decode roundtrip is benign. It fails (0.938 < 0.94). But the threshold 0.94 is arbitrary. There is no theoretical derivation of why 0.94 is the right cutoff vs 0.93 or 0.95. More importantly:

- **What does "benign" mean?** If VAE encode-decode alone causes 0.938 CLIP CS, then any downstream correlation between VAE noise and semantic drift could simply be the VAE's deterministic compression artifact, not a learned degradation mechanism.
- **The p-value against threshold is 0.688** — this is a null result. The experiment cannot distinguish the VAE roundtrip from the threshold. This means the "continuation" decision is not statistically justified.
- **The state file claims "mean CLIP CS=0.938, FAIL" as if it's a definitive result.** It is not. It's an inconclusive comparison with an arbitrary threshold.

**Risk:** If KAS presents CNLSA at a conference with this σ=0 gate, reviewers will immediately ask: (1) why 0.94? (2) what does "benign" vs "damaging" mean in terms of semantic content? (3) is the 0.938 score actually acceptable for semantic similarity tasks?

**Recommendation:** Replace the binary threshold with a proper comparison: CLIP CS for VAE roundtrip vs. CLIP CS for pixel-noised images at equivalent perceptual distortion (matched LPIPS or SSIM). Without this, the σ=0 gate is uninterpretable.

---

### 1.2 CLIP-Specificity Falsification — Internal Validity

**Finding:** DINOv2 ViT-S/14 CS = 0.8155 ± 0.0938, below 0.97 threshold → CLIP-specificity FALSIFIED. Category ANOVA p=0.6037 → drift is uniform across categories.

**Scalpel verdict: Correct result, but interpretation needs care.**

The falsification is methodologically sound. However:
- **DINOv2 ViT-B/14 CS=0.343** is dramatically lower (d=-3.296). The difference between ViT-S and ViT-B is not discussed. Larger DINOv2 models may be more sensitive to VAE artifacts, or the ViT-B/14 architecture responds differently to input perturbations.
- **The "CLIP-specificity" framing is potentially misleading.** The hypothesis was "CLIP is uniquely damaged by VAE." What was falsified is "CLIP is more damaged than other encoders." But DINOv2 ViT-B/14 shows even larger damage (CS=0.343). This actually suggests VAE damage is *universal*, not CLIP-specific. The framing should be "VAE-induced semantic drift is modality-general (CLIP and DINOv2 both damaged)" not "CLIP-specificity falsified."
- **Confounding issue:** The category ANOVA result (p=0.6037) is based on a per-image split into COCO categories. The categories in COCO val2017 are coarse (person, animal, object). If semantic drift is uniform at this coarse level, it could still be content-type-specific at a finer granularity (e.g., texture density, scene complexity, object count).

**Risk:** The current narrative ("CLIP-specificity falsified, drift is uniform") could be challenged by noting the ViT-B/14 result contradicts the "CLIP-specific" claim more forcefully than the ViT-S/14 result. The framing should be revised to avoid giving reviewers an easy attack.

---

### 1.3 VAE Latent Noise vs. Pixel Noise (Option B)

**Finding:** Cohen's d = -2.75 (VAE more damaging than pixel noise, opposite of original hypothesis direction). Corrected interpretation: d<0 IS the confirmation.

**Scalpel verdict: The effect is real, but the comparison is not controlled.**

The negative d means VAE latent noise causes more CLIP drift than pixel noise at matched noise levels. However:
- **Matched on what?** The state says σ is matched, but σ in VAE latent space and σ in pixel space are not comparable quantities. They have different physical meanings and different effects on the diffusion sampling process.
- **The CLIP CS for σ=0 (VAE only) = 0.9388** is actually the most interpretable result here. A CLIP CS drop of ~6% from VAE encode-decode alone is concrete and doesn't require noise-level comparisons.
- **The correlation r=-0.6528, p=0.0001** is the strongest result in the CNLSA system. But this is correlational — it shows VAE latent noise co-occurs with CLIP semantic drift. It does not establish causation. The drift could be: (a) VAE destroys semantic structure → CLIP sees drift, or (b) VAE introduces structured artifacts that CLIP is particularly sensitive to.

**Risk:** Low for the directional claim (VAE is damaging). High for the mechanistic claim (VAE destroys CLIP's semantic representations specifically). Without a mechanistic account, this is a correlation study.

---

### 1.4 CNLSA Summary Assessment

| Claim | Evidence | Risk Level |
|-------|----------|------------|
| VAE encode-decode causes CLIP semantic drift | σ=0 CS=0.938, r=-0.65 | LOW — robust |
| CLIP-specificity falsified (drift not CLIP-specific) | DINOv2 CS=0.8155, ANOVA p=0.60 | MEDIUM — framing issue |
| Drift is category-uniform | ANOVA p=0.60 | MEDIUM — coarse categories |
| VAE more damaging than pixel noise at matched σ | d=-2.75 | LOW-MEDIUM — units not comparable |
| σ=0 gate establishes benign baseline | CS=0.938 vs 0.94 threshold | HIGH — threshold arbitrary |

**Overall CNLSA assessment:** The core finding (VAE → CLIP semantic drift) is defensible. The biggest vulnerability is the σ=0 gate interpretation and the CLIP-specificity framing.

---

## SECTION 2: TrACE-Video Direction #1 — 实验设计缺陷审查

### 2.1 CPU Toy Experiment: Fundamental Problems

**Finding:** Direction #1 (Agreement as Compute Gate) validated on synthetic tensor "agreement" data with 4 COCO images. Results: thresh=0.80 → 27.5% step reduction, SSIM=0.50.

**Scalpel verdict: The CPU toy is methodologically inadequate on multiple dimensions.**

**Problem 1: Synthetic data is not real video generation.**
The "agreement" values in `agreement_gate_raw.json` are computed on synthetic tensors, not actual denoising trajectories from a diffusion model. The agreement profile (steps 0-9) shows near-perfect agreement at step 9 (0.9999) but very low agreement at step 0 (0.63-0.72). This pattern is likely an artifact of how the synthetic tensors were generated, not a genuine property of diffusion model denoising trajectories.

**Problem 2: 4 images is catastrophically underpowered.**
The entire threshold sweep (8 thresholds × 4 images = 32 data points) is used to estimate step reduction and SSIM at each threshold. This is n=4 per threshold. The SSIM values at thresh=0.80 range from 0.30 to 0.68 across the 4 images — the variance is enormous and the mean is meaningless at this sample size.

**Problem 3: No ground truth for "correct early exit."**
The experiment measures SSIM of the gated output vs. full 10-step output. But there is no ground truth for what the "correct" number of steps should be for each image. An image that needs 10 steps will show low SSIM when gated at step 7. An image that only needs 3 steps will show high SSIM when gated at step 7. The results conflate image difficulty with agreement threshold effectiveness.

**Problem 4: No human evaluation or downstream task.**
SSIM is a pixel-level metric. For video generation, downstream metrics like temporal consistency, prompt alignment, and perceptual quality matter more. The SSIM=0.50 at thresh=0.80 means the gated output is only 50% similar to the full-output in pixel space. This is a very low quality signal.

**Problem 5: Gate fires on ALL images at ALL thresholds.**
Looking at `gate_fired` in the JSON: at thresh=0.50 and 0.60, ALL 4 images fire the gate (gate_step=0, meaning 90% step reduction). This means the agreement at step 0 is already above 0.50-0.60 for all images. This is a red flag — it suggests the "agreement" metric is not discriminative at the low end, or the synthetic data has an agreement distribution that doesn't match real diffusion trajectories.

---

### 2.2 Idea #2 (CLIP L2 vs Semantic Inconsistency) — Same-Model Confound

**Finding:** r=0.9895 (CLIP L2 distance vs. 1 - CLIP cosine similarity), confirmed with cross-encoder DINOv2: r=0.6117.

**Scalpel verdict: The confound correction is properly done, but the interpretation of r=0.6117 needs refinement.**

The cross-encoder experiment (0417-AM) is good methodology. The r collapse from 0.9537 to 0.6117 is genuine evidence of same-model inflation.

However:
- **r=0.6117 still passes the 0.6 threshold** (p<1e-11). The honest cross-encoder correlation is still statistically significant. This should be framed as "effect is robust but attenuated" rather than "confound confirmed" — the confound is confirmed but the effect survives.
- **The Spearman ρ=0.4954 vs Pearson r=0.6117** discrepancy is large. This suggests the relationship is not linear — there may be outliers or a non-linear monotonic relationship. This should be investigated.
- **The threshold of 0.6 for "honest r" was set post-hoc.** There is no theoretical justification for why r>0.6 means the effect is real. If the threshold were 0.7, the effect would be classified as "failed." This is a QRPs (questionable research practice) risk.

---

### 2.3 TrACE-Video Rejection of Pixel-Space Methods

**Finding:** MemFlow REJECTED (pixel narrative), AVD REJECTED (distillation≠measurement), Frame Guidance REJECTED (guidance≠measurement).

**Scalpel verdict: Rejections are defensible but Frame Guidance rejection is the weakest.**

The MemFlow and AVD rejections are solid — neither paper measures inter-frame latent agreement in the sense TrACE-Video proposes.

Frame Guidance rejection is more nuanced:
- Frame Guidance is training-free frame-level control compatible with any video model
- The state file says "guidance≠measurement" but the actual Frame Guidance paper may include metrics that overlap with TrACE-Video's goals
- The rejection should be more carefully documented with specific quotes from the paper about what Frame Guidance does and does not measure

**Risk:** If KAS presents TrACE-Video at a conference, someone in the audience will ask about Frame Guidance. A vague "guidance≠measurement" rejection will not hold up. Specific methodology differences need to be documented.

---

### 2.4 TrACE-Video Summary Assessment

| Claim | Evidence | Risk Level |
|-------|----------|------------|
| Direction #1 (Agreement Gate) validated on CPU toy | 4 images, synthetic data, SSIM=0.50 | HIGH |
| Pixel-space methods REJECTED | MemFlow, AVD, Frame Guidance | MEDIUM — Frame Guidance weak |
| Idea #2 (L2→inconsistency) confirmed | r=0.9895 same-model, r=0.6117 cross-encoder | LOW-MEDIUM |
| LatSearch found as learned counterpart | Project page ✅, code ✅ | LOW |

**Overall TrACE-Video assessment:** The CPU validation is insufficient for any publication claim. The most publishable finding is the cross-encoder confound correction for Idea #2, but this needs GPU validation with real generated videos before it becomes a paper.

---

## SECTION 3: Memory Candidates — Approval/Rejection/Merge

### 3.1 In-Place TTT Negative Result (0.9) — **APPROVE**

The Step-Intrinsic TTT hypothesis is genuinely falsified at all K values. The corrected bug (K=1 reweight Δalign=-0.22) is robust. This is a valid negative result that should be in memory. The confidence 0.9 is appropriate.

### 3.2 CNLSA New Idea (0.7) — **APPROVE with caveat**

The idea of CNLSA = "factor separability loss" or "VAE destroys semantic factor structure" is a reasonable theoretical reframing. However, the candidate should note that Send-VAE (found by Scout 0417-AM) already explores this direction and has code. The CNLSA idea should be positioned as complementary to Send-VAE, not redundant.

### 3.3 TrACE-RM Temporal Decoupling Fix (0.8) — **REJECT**

The TrACE-RM Temporal Decoupling approach was ALREADY SHOWN TO BE WORSE than the circular baseline (r=-0.0554 vs r=+0.1169 in the pilot). The memory candidate says "fix" but the fix failed. This is not a fix — it's a falsified approach. Do not memorialize it as a fix. Archive it as a falsified direction.

### 3.4 TrACE-Video Re-Scout Needed (0.85) — **MERGE into Video-As-Prompt candidate**

The re-scout needed finding is already covered by the Video-As-Prompt (ByteDance, ICLR 2026) and TTT-Video-DiT findings from 0416-AM. The re-scout need is addressed by those papers. Merge these into a single "TrACE-Video related papers" candidate rather than keeping them separate.

### 3.5 EC-VAE Cannot Rescue CNLSA (0.85) — **APPROVE**

The finding that EC-VAE (FastVAE 32×32, not drop-in) cannot rescue CNLSA is a valid negative result. It's worth remembering that attempts to fix VAE semantic drift with EC-VAE-style corrections don't work because EC-VAE is not designed for this use case.

### 3.6 Video-As-Prompt ICLR 2026 (0.8) — **APPROVE**

ByteDance ICLR 2026 paper with code. This is a directly relevant related work finding. 0.8 confidence is appropriate.

### 3.7 TTT-Video-DiT Training-Based (0.8) — **APPROVE**

The finding that TTT-Video-DiT is training-based (not inference-only) is a valid methodological distinction. It's relevant because the Step-Intrinsic TTT work is inference-only. Keeping this distinction in memory helps avoid future confusion about whether TTT-Video-DiT is comparable to the TTT work.

### Merge Recommendations:
- **Merge:** TrACE-Video re-scout needed (0.85) + Video-As-Prompt (0.8) + TTT-Video-DiT (0.8) → single "TrACE-Video Related Work" candidate (0.82)
- **Reject without memorializing:** TrACE-RM Temporal Decoupling fix (the approach failed; memorialize the failure, not the fix)

---

## SECTION 4: CPU-Feasible Next-Round Experiment Designs

### 4.1 CNLSA — Factor Separability Metric (CPU-feasible, Nova Idea A)

**Design:** Instead of relying on σ=0 CLIP CS as a binary gate, design a proper factor separability metric:
1. Take VAE encode-decode image I_vae and original I_orig
2. Measure CLIP embedding E(I_orig) and E(I_vae)
3. Measure DINOv2 embedding D(I_orig) and D(I_vae)
4. Compute: (a) semantic alignment (CLIP cosine), (b) structural alignment (DINO L2), (c) factor separability: if semantic factors (object class, color, texture) are entangled in VAE latent space, interpolating in latent space will cause non-linear semantic transitions
5. Test: do VAE latents of different semantic content remain linearly separable in CLIP space?

**Why CPU-feasible:** Only needs pre-trained VAE + CLIP + DINOv2, no diffusion sampling. Can use existing VAE decode outputs from prior experiments.

**Expected output:** A continuous metric (factor separability score) that is more informative than a binary σ=0 gate. Can be correlated with generation quality downstream.

---

### 4.2 TrACE-Video — Cross-Encoder Validation with Real Images (CPU-feasible)

**Design:** Extend the cross-encoder confound experiment to use real VAE-reconstructed images (not synthetic augmentations):
1. Take 50 COCO val2017 images
2. VAE encode-decode each (using available VAE)
3. Compute: DINOv2 L2 distance (original vs VAE-reconstructed) as the predictor
4. Compute: CLIP semantic consistency score (original vs VAE-reconstructed) as the target
5. Compute: human-rated semantic similarity (if CPU-only, use LVIS category match as proxy)

**This tests:** Does DINOv2 L2 predict CLIP semantic consistency for real VAE artifacts (not just augmentations)? This is a more honest validation of Direction #1's premise.

**Why CPU-feasible:** No diffusion model needed, only VAE decode (which is already available from prior experiments).

---

### 4.3 TrACE-Video — Agreement Gate on Real VAE Reconstructions (CPU-feasible)

**Design:** Replace the synthetic tensor agreement experiment with real VAE reconstruction agreement:
1. Take a clean image I_0
2. Add noise to latent: z = I_0 + ε (controlled noise levels)
3. VAE decode to get I_recon
4. Measure DINOv2 L2(I_0, I_recon) as "agreement" signal
5. See if DINOv2 agreement at step k predicts the semantic quality at step k+1

**This addresses:** The synthetic tensor experiment's fundamental validity problem. Real VAE reconstructions give actual semantic content to measure agreement against.

**Expected output:** Agreement profiles that are grounded in real VAE behavior rather than synthetic tensor math.

---

### 4.4 CNLSA — Modality-General Drift Quantification (CPU-feasible)

**Design:** Properly test the "modality-general" claim:
1. Collect 100 COCO val2017 images
2. VAE encode-decode all 100
3. Measure: CLIP CS, DINOv2 CS, LPIPS perceptual distance
4. Test: Is the VAE-induced drift correlated across encoders? (CLIP CS correlated with DINOv2 CS?)
5. Test: Does image complexity (object count, texture variance) predict drift magnitude?

**Why CPU-feasible:** Single VAE encode-decode pass, no diffusion sampling.

**Expected output:** A proper characterization of whether VAE drift is truly modality-general or encoder-architecture-specific.

---

## SECTION 5: Key Risks Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| σ=0 gate threshold is arbitrary | HIGH | Replace with matched comparison or remove binary decision |
| CLIP-specificity framing contradicts ViT-B/14 result | MEDIUM | Reframe as "modality-general drift" |
| TrACE-Video CPU toy has no publication value | HIGH | Do not claim Direction #1 is validated; it is only CPU-toy explored |
| Same-model confound threshold (0.6) is post-hoc | MEDIUM | Pre-register threshold or use理论-driven value |
| Frame Guidance rejection too vague | MEDIUM | Document specific methodology differences |
| Category ANOVA uses coarse COCO categories | MEDIUM | Use finer-grained scene/texture categories |

---

## FINAL VERDICT

**CNLSA:** Core finding (VAE → CLIP semantic drift) is defensible. σ=0 gate is the biggest liability. Should not be presented as a completed study — needs GPU validation and proper factor separability metric.

**TrACE-Video Direction #1:** CPU toy is exploratory only. The cross-encoder confound correction (r=0.6117) is the most solid result. Direction #1 is a hypothesis, not a validated finding.

**Memory candidates:** Approve #1, #2, #5, #6, #7. Reject #3. Merge #4 into #6.

**Next step priority:** GPU restore → real SDXL validation for CNLSA σ=0 gate + TrACE-Video Direction #1 latent agreement. CPU path → factor separability metric design as interim contribution.
