"""
Experiment D: TrACE-Video Cross-Encoder Confound Separation
Measures Pearson/Spearman correlation between DINOv2 L2 distance and CLIP cosine similarity
across noisy versions of COCO images. Partial correlation attempts to separate encoder confound.
Uses timm for DINOv2-vits14 (local cache) and open_clip for CLIP ViT-B/32.
"""

import os
import json
import random
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import timm
import open_clip
from open_clip import tokenize

# Paths
COCO_VAL2017 = "/home/kas/.cache/huggingface/hub/datasets--merve--coco/snapshots/9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017"
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0417-pm"
os.makedirs(OUT_DIR, exist_ok=True)

# Config
N_ANCHORS = 50
NOISE_LEVELS = [5, 10, 20, 40, 80]
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"

# Load DINOv2 via timm (uses local cache)
print("Loading DINOv2 ViT-S/14 via timm...")
dinov2_model = timm.create_model('vit_small_patch14_dinov2', img_size=224, pretrained=True)
dinov2_model.eval()
dinov2_model.to(device)
# Get transform from timm
dinov2_transform = timm.data.create_transform(
    input_size=224, is_training=False, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)

# Load CLIP ViT-B/32 via open_clip
print("Loading CLIP ViT-B/32 via open_clip...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval()
clip_model.to(device)

def get_dinov2_embedding(img_pil):
    """Extract DINOv2 [CLS] embedding via timm."""
    x = dinov2_transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = dinov2_model(x)
        if hasattr(out, 'z'):
            emb = out.z.squeeze().cpu().numpy()
        elif hasattr(out, 'query_output'):
            emb = out.query_output[:, 0, :].squeeze().cpu().numpy()
        else:
            emb = out.squeeze().cpu().numpy()
    return emb

def get_clip_embedding(img_pil):
    """Extract CLIP [CLS] embedding via open_clip."""
    x = clip_preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(x).squeeze().cpu().numpy()
    return emb

def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def add_gaussian_noise(img_array, sigma):
    noise = np.random.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

# Get anchor file list
all_files = sorted([f for f in os.listdir(COCO_VAL2017) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
anchor_files = random.sample(all_files, N_ANCHORS)
print(f"Selected {N_ANCHORS} anchor images from {COCO_VAL2017}")

# Pre-compute anchor embeddings
print("Computing anchor embeddings...")
anchor_dinov2 = []
anchor_clip = []
anchor_imgs = []

for af in tqdm(anchor_files, desc="Anchors"):
    img = Image.open(os.path.join(COCO_VAL2017, af)).convert("RGB")
    anchor_imgs.append(img)
    anchor_dinov2.append(get_dinov2_embedding(img))
    anchor_clip.append(get_clip_embedding(img))

anchor_dinov2 = np.array(anchor_dinov2)
anchor_clip = np.array(anchor_clip)
print(f"Anchor embeddings shape: DINOv2 {anchor_dinov2.shape}, CLIP {anchor_clip.shape}")

# Generate noisy versions and measure
dino_l2_list = []
clip_cs_list = []

print("Generating noisy frames and measuring...")
for i, (img, dino_anch, clip_anch) in enumerate(tqdm(zip(anchor_imgs, anchor_dinov2, anchor_clip), total=N_ANCHORS, desc="Noisy frames")):
    img_array = np.array(img)
    for sigma in NOISE_LEVELS:
        noisy_img = add_gaussian_noise(img_array, sigma)
        dino_noisy = get_dinov2_embedding(noisy_img)
        clip_noisy = get_clip_embedding(noisy_img)

        dino_l2 = l2_distance(dino_anch, dino_noisy)
        clip_cs = cosine_similarity(clip_anch, clip_noisy)

        dino_l2_list.append(dino_l2)
        clip_cs_list.append(clip_cs)

dino_l2_arr = np.array(dino_l2_list)
clip_cs_arr = np.array(clip_cs_list)
n_frames = len(dino_l2_arr)

print(f"\nTotal frames: {n_frames}")
print(f"DINOv2 L2 - mean: {dino_l2_arr.mean():.4f}, std: {dino_l2_arr.std():.4f}")
print(f"CLIP CS    - mean: {clip_cs_arr.mean():.4f}, std: {clip_cs_arr.std():.4f}")

# Pearson correlation
pearson_r, p_value = pearsonr(dino_l2_arr, clip_cs_arr)
print(f"Pearson r: {pearson_r:.4f}, p-value: {p_value:.2e}")

# Spearman correlation
spearman_rho, spearman_p = spearmanr(dino_l2_arr, clip_cs_arr)
print(f"Spearman rho: {spearman_rho:.4f}, p-value: {spearman_p:.2e}")

# Partial correlation: r(CLIP CS, DINOv2 L2 | confound)
# Confound = embedding magnitude (norm) as proxy for encoder internal-state confound
dino_norms_anch = np.array([np.linalg.norm(x) for x in anchor_dinov2])
dino_norms_noisy = []
for i, (img, dino_anch) in enumerate(zip(anchor_imgs, anchor_dinov2)):
    img_array = np.array(img)
    for sigma in NOISE_LEVELS:
        noisy_img = add_gaussian_noise(img_array, sigma)
        dino_noisy = get_dinov2_embedding(noisy_img)
        dino_norms_noisy.append((np.linalg.norm(dino_anch) + np.linalg.norm(dino_noisy)) / 2)

dino_norms_noisy = np.array(dino_norms_noisy)
# Partial correlation: r(x,y|z) = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
r_xy = pearsonr(dino_l2_arr, clip_cs_arr)[0]
r_xz = pearsonr(dino_l2_arr, dino_norms_noisy)[0]
r_yz = pearsonr(clip_cs_arr, dino_norms_noisy)[0]

if (1 - r_xz**2) > 1e-10 and (1 - r_yz**2) > 1e-10:
    partial_r = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    print(f"Partial correlation (CLIP CS ~ DINOv2 L2 | confound): {partial_r:.4f}")
else:
    partial_r = None
    print("Partial correlation undefined (zero variance in confound)")

threshold_passed = abs(pearson_r) >= 0.5

results = {
    "pearson_r": round(float(pearson_r), 6),
    "spearman_rho": round(float(spearman_rho), 6),
    "p_value": float(p_value),
    "n_frames": n_frames,
    "partial_r": round(float(partial_r), 6) if partial_r is not None else None,
    "threshold_passed": bool(threshold_passed)
}

out_path = os.path.join(OUT_DIR, "idea_d_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {out_path}")
print(json.dumps(results, indent=2))
