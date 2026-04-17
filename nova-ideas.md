# Nova Ideas — 0417-PM (CPU-only)

## 研究现状回顾

| 发现 | 结论 |
|------|------|
| VAE→CLIP drift | 证实，category-uniform |
| cross-encoder confound | r=0.6117（CLIP CS ∝ DINOv2 L2） |
| TrACE-Video pixel-space | 线全灭 |
| Send-VAE (ICLR 2026) | 解决 VAE semantic discriminability |
| LatSearch | learned inter-frame latent agreement |

---

## Idea D: TrACE-Video 跨模型 confound 分离实验 ⭐最高优先级

### 动机
cross-encoder confound r=0.6117 意味着 CLIP CS 和 DINOv2 L2 共享训练数据伪影，不等于 CLIP 真的在度量 semantic inconsistency。需要 third encoder 来切断 confound。

### 假设
DINOv2 L2 distance 可以预测 CLIP semantic inconsistency（beyond encoder confound）。

### 最小实验设计
```
数据: 合成帧 n=50
  - 锚帧: COCO val2017 随机 50 张
  - 扰动: 每个锚帧加 5 档 Gaussian noise (σ=5,10,20,40,80 pixel L2)
  - 合成帧总数: 50×5=250

测量:
  - DINOv2 ViT-S/14 L2 distance (锚帧 vs 扰动帧)
  - CLIP ViT-B/32 cosine similarity (锚帧 vs 扰动帧)

分析:
  - Pearson r(DINOv2 L2, CLIP CS) 
  - partial corr: r(CLIP CS, DINOv2 L2 | encoder_confound) ← 需要 DINOv2 和 CLIP 共有 variance 估计
  - 回归残差分析: DINOv2 L2 预测 CLIP CS，检视是否有未解释方差

失败条件: r < 0.5（全局相关）

理论风险: 仍然可能 shared training confound 污染。需要 partial correlation 设计。
```

### CPU 可行性
- DINOv2 ViT-S/14: ~0.3s/图，250图 ≈ 75s
- CLIP ViT-B/32: ~0.2s/图，250图 ≈ 50s
- 总计 ~2-3 分钟 CPU
- ✅ 完全可行

---

## Idea E: CLIP 协方差矩阵特征值衰减率替代指标

### 动机
VAE decode 太慢（几十秒/图），无法批量验证。CLIP embedding 的协方差矩阵特征值衰减率可以非监督地反映 embedding 空间的"各向同性"程度，作为 drift 的代理指标。

### 假设
VAE roundtrip 后 CLIP embedding 协方差矩阵变得更各向同性（特征值衰减更慢/更平），因为语义特异性信息在 VAE 压缩中丢失。

### 最小实验设计
```
数据: COCO val2017 随机 n=30
配对: 每张图生成 VAE roundtrip 版本（encode+decode，decode只做特征提取，不做像素重建质量评估）

测量:
  - 对 {原始30图} 提取 CLIP ViT-B/32 embedding → C_orig (30×512)
  - 对 {VAE roundtrip 30图} 提取 CLIP embedding → C_vae (30×512)
  - SVD: C_orig @ C_orig^T, C_vae @ C_vae^T → 特征值 λ_1≥λ_2≥...≥λ_512

分析:
  - 衰减曲线对比: log(λ_i) vs i
  - 归一化衰减率: λ_1 / Σλ_i （第一主成分解释方差比）
  - paired t-test: 原始 vs VAE 衰减率差异
  - 可视化: 2D t-SNE 对比两簇分离程度

失败条件: 差异不显著 p > 0.05
```

### CPU 可行性
- CLIP 特征提取: 30图 × 0.2s ≈ 6s
- VAE encode: 30图 × 0.5s ≈ 15s
- SVD 30×512: <1s
- 总计 ~30s-1min CPU
- ✅ 完全可行，且比 Idea D 更快

---

## Artifact 主线优先级评估

### Send-VAE (KlingAIResearch, ICLR 2026) ⭐⭐
- **主题**: semantic discriminator-enhanced VAE
- **直接解决**: CNLSA 的 VAE→CLIP drift 核心问题
- **价值**: 提供了架构层面的解释 — 为什么 VAE 会丢失 semantic discriminability
- **优先级**: 高。如果能复现或引用其关键发现（semantic discriminator loss），CNLSA 的 mechanism story 显著增强
- **风险**: ICLR 2026 论文刚录用，代码可能未公开；需要追踪
- **Scout 状态**: 搜索未命中，需人工追踪论文/GitHub

### LatSearch (2024) ⭐
- **主题**: learned inter-frame latent agreement for video
- **与 CNLSA 相关度**: 中。提供了 latent space temporal consistency 的 reference
- **优先级**: 低。视频方向 vs CNLSA 的单帧 VAE drift，不是同一个问题空间
- **可作为**: 背景 knowledge，证明 latent agreement 是值得研究的方向

### 综合优先级排序

| 优先级 | Idea | 理由 | CPU 可行 |
|--------|------|------|----------|
| 1 | Idea D | confound 分离实验，直接回答 CLIP CS 有效性 | ✅ |
| 2 | Idea E | 快速替代指标，30分钟可验证 | ✅ |
| 3 | Send-VAE | 理论增强，追踪代码/论文细节 | N/A |
| 4 | LatSearch | 背景知识，非核心 | N/A |

---

## 立即可执行的 next actions

1. **Idea D**: Kernel 写合成帧生成 + DINOv2/CLIP 双encoder 提取脚本
2. **Idea E**: 复用 Idea D 的部分 pipeline，添加 SVD 分析
3. **Send-VAE**: Scout 追踪 KlingAIResearch 是否有公开代码

---

*Nova — 0417-PM window | GPU: unavailable | model: MiniMax-M2.7*
