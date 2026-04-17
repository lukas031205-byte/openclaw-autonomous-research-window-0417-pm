# Scout Results — 0417-PM Window (2026-03-17 ~ 2026-04-17, 60-day window)

**Scout:** Scout subagent | **Model:** minimax-cn/MiniMax-M2.7 | **Date:** 2026-04-17

---

## Direction 1: TrACE-Video / Video Latent Consistency / Test-Time Compute

### 1. Consistency-Preserving Diverse Video Generation
- **Authors:** Xinshuang Liu et al.
- **Year:** 2026 (Feb)
- **arXiv:** https://arxiv.org/abs/2602.15287
- **GitHub:** Code not yet publicly released (paper states "Code will be released")
- **Relevance:** Flow-matching video generator with joint-sampling that preserves temporal consistency while maximizing batch diversity. Uses latent-space embedding/interpolation models for lightweight consistency computation without decoder backpropagation. Directly relevant to inter-frame latent agreement.

---

### 2. StableWorld: Towards Stable and Consistent Long Interactive Video Generation
- **Authors:** Ying Yang et al.
- **Year:** 2026 (Jan)
- **arXiv:** https://arxiv.org/abs/2601.15281
- **GitHub:** Not found
- **Relevance:** Addresses error accumulation and scene collapse in long-horizon interactive video generation. Proposes Dynamic Frame Eviction Mechanism that filters degraded frames using geometric consistency as anchor. Model-agnostic; applicable to Hunyuan-GameCraft, Matrix-Game, Open-Oasis. Core insight: cumulative drift originates from same-scene frame degradation propagating errors.

---

### 3. Video-T1: Test-Time Scaling for Video Generation
- **Authors:** Fangfu Liu et al. (Tsinghua University + Tencent)
- **Year:** 2025 (ICCV 2025, highly relevant as recent work)
- **Project Page:** https://liuff19.github.io/Video-T1/
- **GitHub:** https://github.com/THU-SI/Video-T1
- **Relevance:** Directly explores test-time compute scaling for video generation — search in Gaussian noise space guided by video verifiers (VisionReward). Tree-of-Frames (ToF) enables adaptive branch pruning in autoregressive video generation. Strong fit with "test-time compute gating for video diffusion."

---

### 4. Pathwise Test-Time Correction for Autoregressive Long Video Generation (TTC)
- **Authors:** Xunzhi Xiang et al.
- **Year:** 2026 (Feb, v2 Mar 2026)
- **arXiv:** https://arxiv.org/abs/2602.05871
- **GitHub:** Not found
- **Relevance:** Training-free test-time correction method using first frame as stable reference anchor to calibrate intermediate stochastic states. Works with distilled autoregressive diffusion models; extends generation to 30-second benchmarks with negligible overhead. Addresses error accumulation in long video — complementary to TTC's approach vs. StableWorld's frame eviction.

---

### 5. Test-Time Scaling of Diffusions with Flow Maps
- **Authors:** (OpenReview — ICLR 2026)
- **Venue:** ICLR 2026
- **OpenReview:** https://openreview.net/forum?id=lR8GufFQMb
- **GitHub:** Not found
- **Relevance:** Principled framework for test-time adaptation of diffusion models using flow map structure — enables efficient and scalable test-time compute allocation for diffusion models, relevant to video generation.

---

### 6. TTOM: Test-Time Optimization and Memorization for Compositional Video Generation
- **Authors:** (arXiv 2510.07940)
- **Year:** 2026 (from late 2025)
- **arXiv:** https://arxiv.org/html/2510.07940v2
- **GitHub:** Not found
- **Relevance:** Test-time optimization of attention-to-layout alignment for compositional T2V-CompBench; compares against commercial (Pika, Gen-3, Kling) and open-source (Open-Sora, CogVideoX, Wan2.1) baselines. Demonstrates TTO as viable strategy for video diffusion at inference.

---

### 7. ATHENA: Adaptive Test-Time Adaptation (for video diffusion)
- **Authors:** (mentioned in Scouts by Yutori — related to test-time texture consistency)
- **Year:** 2026
- **Relevance:** Physical simulator in-the-loop video diffusion with Test-Time Texture Consistency; adaptive test-time adaptation for video models.

---

## Direction 2: CNLSA / VAE Semantic Drift / Factor Separability

### 8. Send-VAE: Semantic-Disentangled VAE (Boosting Latent Diffusion Models via Semantic-Disentangled VAE)
- **Authors:** (OpenReview ICLR 2026 — submission #3322)
- **Venue:** ICLR 2026
- **OpenReview:** https://openreview.net/forum?id=bsmKEJfaar
- **GitHub:** Not found in current search
- **Relevance:** Directly addresses VAE semantic disentanglement for latent diffusion. Proposes that generation-friendly VAE should have semantic disentanglement ability — uses pre-trained Vision Foundation Model (VFM) features to guide VAE latent alignment via non-linear mapper. Linear probing on attribute prediction tasks validates semantic disentanglement correlation with downstream generation quality. New state-of-the-art FID 1.21 on ImageNet 256×256 with SiT. **Very high relevance to CNLSA's Send-VAE experiment finding.**

---

### 9. DecVAE: Variational Decomposition Autoencoding
- **Authors:** Ioannis et al.
- **Year:** 2026 (Jan)
- **arXiv:** https://arxiv.org/abs/2601.06844
- **GitHub:** https://github.com/GiannisZgs/DecVAE
- **Relevance:** VAE extended with signal decomposition structural bias; learns multiple latent subspaces aligned with time-frequency characteristics via contrastive SSL + orthogonalization. Improves disentanglement quality over state-of-the-art VAE methods. Orthogonality regularizer suppresses cross-factor interference — conceptually relevant to semantic drift correction via factor separation.

---

### 10. Disentangled Representation Learning via Flow Matching
- **Authors:** Jinjin Chi et al.
- **Year:** 2026 (Feb)
- **arXiv:** https://arxiv.org/abs/2602.05214
- **GitHub:** Not found
- **Relevance:** Flow-matching framework for disentangled representation learning — casts disentanglement as learning factor-conditioned flows in compact latent space. Non-overlap (orthogonality) regularizer enforces explicit semantic alignment and reduces cross-factor interference. Factor-level control within flow-matching framework — relevant to semantic drift and factor separability in diffusion latent spaces.

---

### 11. DA-VAE: Detail-Aligned VAE (Plug-in Latent Compression for Diffusion)
- **Authors:** Xin Cai et al.
- **Venue:** CVPR 2026
- **arXiv:** https://arxiv.org/abs/2603.22125
- **GitHub:** Not found
- **Relevance:** Increases compression ratio of pretrained VAE while preserving structured latent space via detail-alignment mechanism. Enables 1024×1024 generation with SD3.5 at 4× fewer tokens; unlocks 2048×2048. Directly relevant to understanding how VAE compression affects latent semantic structure — warm-start fine-tuning preserves original latent structure. **Relevant to CNLSA CPU experiments on Send-VAE and VAE-induced drift.**

---

### 12. ODC: Orthogonal Drift Correction (Improving Semantic Alignment via Training-Free Embedding Refinement)
- **Authors:** (Under review — ICLR 2026, double-blind)
- **Venue:** ICLR 2026
- **OpenReview PDF:** https://openreview.net/pdf/7787ec25a90eb720a34372a85e0e70fb1a06c98e.pdf
- **GitHub:** Not found
- **Relevance:** Training-free embedding refinement for text-to-image models. Orthogonal Drift Correction removes components that degrade semantic alignment without fine-tuning. Directly addresses concept drift in diffusion models — method name "drift correction" is highly relevant to CNLSA VAE semantic drift work.

---

## Direction 3: PhD-Relevant — ICLR 2026 / CVPR 2026 Alignment, Safety, Reasoning

### 13. Superficial Safety Alignment Hypothesis (SSAH)
- **Authors:** Jianwei Li, Jung-Eun Kim (NC State University)
- **Venue:** ICLR 2026
- **arXiv:** https://arxiv.org/abs/2410.10862
- **Project Page:** https://ssa-h.github.io/
- **GitHub:** Not explicitly in search; project page mentions code
- **Relevance:** Proposes safety alignment as implicit binary classification (fulfill vs. refuse) with identified Safety Critical Units (SCU), Utility Critical Units (UCU), Complex Units (CU), Redundant Units (RU). Freezing SCU/CU during fine-tuning preserves safety while adapting to new tasks. Conceptual framework relevant to any alignment/safety PhD方向. **Strong fit for Tsinghua AI PhD applicant interested in LLM safety.**

---

### 14. ReAlign: Safety-Aligning Reasoning Models with Verifier-Guided Reinforcement Learning
- **Authors:** Xiaomeng Hu, Fei Huang, Junyang Lin, Tsung-Yi Ho
- **Venue:** ICLR 2026
- **OpenReview:** https://openreview.net/forum?id=XxYNlbTFYS
- **GitHub:** Not found
- **Relevance:** Re-aligns Large Reasoning Models (LRMs) for safety using RL with safety verifier (guard model), general reward model, and refusal penalty. Maintains reasoning capability on Arena-Hard-V2, AIME-25, LiveCodeBench, GPQA. Strong alignment/safety+reasoning intersection — directly relevant to PhD方向.

---

### 15. Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check
- **Authors:** Chentao Cao, Xiaojun Xu, Bo Han, Hang Li
- **Venue:** ICLR 2026 (Poster #10010790)
- **OpenReview:** https://openreview.net/forum?id=DK6AToxJNo
- **arXiv:** https://arxiv.org/pdf/2509.11629
- **GitHub:** Not found
- **Relevance:** "Answer-Then-Check" reasoning enables model to evaluate safety before producing final answer. Constructs 80K ReSA dataset. SFT + RL post-training. Strong jailbreak defense while preserving general reasoning. Directly relevant to LLM safety PhD interests.

---

### 16. Unlocking Innate Safety Alignment of LLMs to Any-Depth
- **Authors:** (ICLR 2026 Poster #10011912)
- **Venue:** ICLR 2026
- **Relevance:** Observed that LLMs exhibit strong but shallow alignment — models refuse harmful queries only at surface layer but fail at deeper reasoning depths. Addresses depth-aware safety alignment.

---

### 17. UniDFlow: Best of Both Worlds — Multimodal Reasoning and Generation via Unified Discrete Flow Matching
- **Authors:** Onkar Susladkar, Tushar Prakash, Gayatri Deshmukh et al. (UIUC / CMU / others)
- **Year:** 2026 (Feb)
- **arXiv:** https://arxiv.org/abs/2602.12221
- **GitHub:** Not found
- **Relevance:** Unified discrete flow-matching for multimodal understanding + generation + editing. Task-specific low-rank adapters decouple understanding from generation, avoiding representation entanglement. Reference-based multimodal preference alignment improves faithfulness without retraining. SOTA on 8 benchmarks.

---

### 18. Reasoning-Preserved Safety Alignment for Large Reasoning Models (RPSA)
- **Authors:** (OpenReview ICLR 2026 — Withdrawn but notable)
- **OpenReview:** https://openreview.net/forum?id=3qJNTjvDrm
- **GitHub:** https://anonymous.4open.science/r/RPSA
- **Relevance:** Addresses the phenomenon that safety alignment impairs reasoning capabilities in LRMs. Proposes method to maintain both safety and reasoning. Withdrawn but approach and problem framing are notable for PhD方向 alignment+safety+reasoning.

---

## Summary Table

| # | Paper | Venue/Year | arXiv/Project | GitHub | Relevance |
|---|-------|-----------|---------------|--------|-----------|
| 1 | Consistency-Preserving Diverse Video Generation | arXiv 2026-02 | arxiv.org/abs/2602.15287 | TBD | ⭐⭐⭐ |
| 2 | StableWorld: Long Interactive Video Stability | arXiv 2026-01 | arxiv.org/abs/2601.15281 | — | ⭐⭐⭐ |
| 3 | Video-T1: Test-Time Scaling for Video Generation | ICCV 2025 | liuff19.github.io/Video-T1 | THU-SI/Video-T1 | ⭐⭐⭐ |
| 4 | TTC: Pathwise Test-Time Correction (Long Video) | arXiv 2026-02 | arxiv.org/abs/2602.05871 | — | ⭐⭐⭐ |
| 5 | Test-Time Scaling of Diffusions with Flow Maps | ICLR 2026 | openreview.net/forum?id=lR8GufFQMb | — | ⭐⭐⭐ |
| 6 | TTOM: Test-Time Optimization for Compositional Video | arXiv 2025 | arxiv.org/html/2510.07940v2 | — | ⭐⭐ |
| 7 | Send-VAE: Semantic-Disentangled VAE | ICLR 2026 | openreview.net/forum?id=bsmKEJfaar | — | ⭐⭐⭐⭐ |
| 8 | DecVAE: Variational Decomposition Autoencoding | arXiv 2026-01 | arxiv.org/abs/2601.06844 | GiannisZgs/DecVAE | ⭐⭐⭐ |
| 9 | Disentangled Representation Learning via Flow Matching | arXiv 2026-02 | arxiv.org/abs/2602.05214 | — | ⭐⭐⭐ |
| 10 | DA-VAE: Detail-Aligned VAE (CVPR 2026) | CVPR 2026 | arxiv.org/abs/2603.22125 | — | ⭐⭐⭐ |
| 11 | ODC: Orthogonal Drift Correction | ICLR 2026 | openreview.net/pdf/7787ec25a90eb720... | — | ⭐⭐⭐⭐ |
| 12 | SSAH: Superficial Safety Alignment Hypothesis | ICLR 2026 | arxiv.org/abs/2410.10862 | ssa-h.github.io | ⭐⭐⭐⭐ |
| 13 | ReAlign: Safety-Aligning LRMs with Verifier-Guided RL | ICLR 2026 | openreview.net/forum?id=XxYNlbTFYS | — | ⭐⭐⭐⭐ |
| 14 | Answer-Then-Check: Jailbreak Defense | ICLR 2026 | openreview.net/forum?id=DK6AToxJNo | — | ⭐⭐⭐ |
| 15 | UniDFlow: Unified Discrete Flow Matching | arXiv 2026-02 | arxiv.org/abs/2602.12221 | — | ⭐⭐ |
| 16 | RPSA: Reasoning-Preserved Safety Alignment | ICLR 2026 (Withdrawn) | openreview.net/forum?id=3qJNTjvDrm | anonymous.4open.science/r/RPSA | ⭐⭐⭐ |

---

## Scout Notes

- **Direction 1 (Video/TrACE):** The field is moving toward test-time compute allocation for video (Video-T1, TTC, TTOM) and consistency-preserving joint sampling. TrACE-Video concept of "inter-frame latent agreement" maps well to StableWorld's frame eviction mechanism and Consistency-Preserving Diverse Video Generation's DPP-based gradient regulation.
- **Direction 2 (VAE drift):** Send-VAE (ICLR 2026) is the most directly relevant work — aligns VAE latent space with Vision Foundation Model features to improve semantic disentanglement. ODC (Orthogonal Drift Correction) is conceptually very close to what CNLSA is investigating with "drift correction" — training-free embedding refinement.
- **Direction 3 (Safety/Alignment PhD):** SSAH, ReAlign, and Answer-Then-Check are the most substantive new ICLR 2026 papers. SSAH provides a conceptual framework; ReAlign shows RL-based approach; Answer-Then-Check is practical.
- **Gaps:** No exact "TrACE-Video" paper found — the concept may be specific to KAS's research group. The closest analogues are Consistency-Preserving Diverse Video Generation (flow-matching video latent consistency) and VideoLCM (earlier work).
