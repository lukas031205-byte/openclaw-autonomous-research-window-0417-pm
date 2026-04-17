"""
Experiment E: CLIP Covariance Matrix Eigenvalue Decay Rate
Measures how VAE encode-decode roundtrip affects the principal eigenvalue decay
of CLIP ViT-B/32 embedding covariance matrices.
"""

import os
import json
import random
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from scipy.stats import ttest_rel
from tqdm import tqdm

# Paths
COCO_VAL2017 = "/home/kas/.cache/huggingface/hub/datasets--merve--coco/snapshots/9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017"
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0417-pm"
os.makedirs(OUT_DIR, exist_ok=True)

# Config
N_PAIRS = 30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cpu"

# Load CLIP
print("Loading CLIP ViT-B/32...")
clip_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)

# Load VAE (for encode-decode roundtrip)
print("Loading VAE...")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32)
vae = vae.to(device).eval()

def get_clip_embedding(img):
    with torch.no_grad():
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return emb

def vae_roundtrip(img):
    """Encode img to latents, decode back to image."""
    # Convert PIL to tensor [0,1]
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    # VAE expects 0-1, normalize to [-1,1] if needed, SDXL VAE expects 0-1
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        decoded = vae.decode(latents).sample
    # Convert back to PIL
    decoded = decoded.squeeze().permute(1, 2, 0).cpu().numpy()
    decoded = np.clip(decoded * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(decoded)

# Get file list
all_files = sorted([f for f in os.listdir(COCO_VAL2017) if f.endswith('.jpg') or f.endswith('.png')])
selected_files = random.sample(all_files, N_PAIRS)
print(f"Selected {N_PAIRS} images")

# Extract original embeddings
print("Extracting original CLIP embeddings...")
orig_embeddings = []
for f in tqdm(selected_files, desc="Original"):
    img = Image.open(os.path.join(COCO_VAL2017, f)).convert("RGB")
    emb = get_clip_embedding(img)
    orig_embeddings.append(emb)

orig_embeddings = np.array(orig_embeddings, dtype=np.float32)
print(f"Original embeddings shape: {orig_embeddings.shape}")

# Extract VAE roundtrip embeddings
print("Extracting VAE roundtrip CLIP embeddings...")
vae_embeddings = []
for f in tqdm(selected_files, desc="VAE roundtrip"):
    img = Image.open(os.path.join(COCO_VAL2017, f)).convert("RGB")
    vae_img = vae_roundtrip(img)
    emb = get_clip_embedding(vae_img)
    vae_embeddings.append(emb)

vae_embeddings = np.array(vae_embeddings, dtype=np.float32)
print(f"VAE embeddings shape: {vae_embeddings.shape}")

# Covariance matrices
C_orig = orig_embeddings @ orig_embeddings.T  # 30x30
C_vae = vae_embeddings @ vae_embeddings.T      # 30x30

# SVD
U_orig, S_orig, Vt_orig = np.linalg.svd(C_orig, full_matrices=False)
U_vae, S_vae, Vt_vae = np.linalg.svd(C_vae, full_matrices=False)

# First principal component variance ratio
var_ratio_orig = S_orig[0] / S_orig.sum()
var_ratio_vae = S_vae[0] / S_vae.sum()

print(f"\nOriginal:   λ_1/Σλ = {var_ratio_orig:.6f}")
print(f"VAE:        λ_1/Σλ = {var_ratio_vae:.6f}")
print(f"Difference: {var_ratio_orig - var_ratio_vae:.6f}")

# Paired t-test on the per-sample principal component loadings
# We compute per-image projection onto first PC
proj_orig = orig_embeddings @ U_orig[:, 0]
proj_vae = vae_embeddings @ U_vae[:, 0]

# Align signs by correlation
if np.corrcoef(proj_orig, proj_vae)[0, 1] < 0:
    proj_vae = -proj_vae

t_stat, p_value = ttest_rel(proj_orig, proj_vae)
print(f"\nPaired t-test on first PC projections:")
print(f"  t = {t_stat:.4f}, p = {p_value:.6f}")

# Failure condition: p > 0.05 means VAE didn't significantly change eigenvalue structure
p_threshold = 0.05
passed = bool(p_value <= p_threshold)

# Also compute eigenvalue decay rates
decay_orig = np.diff(S_orig)  # λ_i - λ_{i+1}
decay_vae = np.diff(S_vae)

# Top-5 eigenvalue concentration
top5_ratio_orig = S_orig[:5].sum() / S_orig.sum()
top5_ratio_vae = S_vae[:5].sum() / S_vae.sum()

results = {
    "var_ratio_orig": round(float(var_ratio_orig), 6),
    "var_ratio_vae": round(float(var_ratio_vae), 6),
    "var_ratio_diff": round(float(var_ratio_orig - var_ratio_vae), 6),
    "top5_ratio_orig": round(float(top5_ratio_orig), 6),
    "top5_ratio_vae": round(float(top5_ratio_vae), 6),
    "t_statistic": round(float(t_stat), 6),
    "p_value": round(float(p_value), 6),
    "n_pairs": N_PAIRS,
    "n_dims": orig_embeddings.shape[1],
    "threshold_passed": passed
}

out_path = os.path.join(OUT_DIR, "idea_e_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {out_path}")
print(json.dumps(results, indent=2))
