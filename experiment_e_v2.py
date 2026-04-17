"""
Experiment E: CLIP Covariance Matrix Eigenvalue Decay Rate
Computes VAE embeddings incrementally (saves after each), can be resumed if killed.
"""

import os, json, random, numpy as np, torch
from PIL import Image
from scipy.stats import ttest_rel
from tqdm import tqdm
import open_clip
from diffusers import AutoencoderKL

COCO_VAL2017 = "/home/kas/.cache/huggingface/hub/datasets--merve--coco/snapshots/9e50abcdc1361852f34841af4939cbcd2d37c92f/val2017"
OUT_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0417-pm"
N_PAIRS = 30
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval().to(device)

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32)
vae = vae.to(device).eval()

def get_clip_embedding(img_pil):
    x = clip_preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(x).squeeze().cpu().numpy()

def vae_roundtrip(img_pil):
    img = img_pil.resize((512, 512), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()
        decoded = vae.decode(latents).sample
    decoded = decoded.squeeze().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.clip(decoded * 255, 0, 255).astype(np.uint8))

all_files = sorted([f for f in os.listdir(COCO_VAL2017) if f.lower().endswith(('.jpg','.png'))])
selected_files = random.sample(all_files, N_PAIRS)

# Load or compute orig embeddings
orig_path = os.path.join(OUT_DIR, "orig_embeddings.npy")
if os.path.exists(orig_path):
    orig_embeddings = np.load(orig_path)
    print(f"Loaded orig_embeddings: {orig_embeddings.shape}")
else:
    print("Computing original CLIP embeddings...")
    orig_embeddings = np.array([get_clip_embedding(Image.open(os.path.join(COCO_VAL2017, f)).convert("RGB")) for f in tqdm(selected_files)], dtype=np.float32)
    np.save(orig_path, orig_embeddings)
    print(f"Saved orig_embeddings: {orig_embeddings.shape}")

# Load or compute VAE embeddings incrementally
vae_path = os.path.join(OUT_DIR, "vae_embeddings.npy")
if os.path.exists(vae_path):
    vae_embeddings = np.load(vae_path)
    print(f"Loaded vae_embeddings: {vae_embeddings.shape}")
    start_idx = len(vae_embeddings)
else:
    vae_embeddings = []
    start_idx = 0

if start_idx < N_PAIRS:
    print(f"Computing VAE embeddings from index {start_idx}...")
    for i in tqdm(range(start_idx, N_PAIRS), desc="VAE roundtrip"):
        img = Image.open(os.path.join(COCO_VAL2017, selected_files[i])).convert("RGB")
        vae_img = vae_roundtrip(img)
        vae_emb = get_clip_embedding(vae_img)
        vae_embeddings.append(vae_emb)
        # Incremental save every 5
        if (i + 1) % 5 == 0:
            np.save(vae_path, np.array(vae_embeddings, dtype=np.float32))
            print(f"\n  checkpoint saved at {i+1}/{N_PAIRS}")
    np.save(vae_path, np.array(vae_embeddings, dtype=np.float32))
    print(f"Saved vae_embeddings: {len(vae_embeddings)}")

vae_embeddings = np.array(vae_embeddings, dtype=np.float32)
print(f"\nFinal shapes: orig={orig_embeddings.shape}, vae={vae_embeddings.shape}")

# Correct PCA: SVD of data X, PC1 direction = right singular vector V[:, 0]
_, _, V_orig = np.linalg.svd(orig_embeddings, full_matrices=False)
_, _, V_vae = np.linalg.svd(vae_embeddings, full_matrices=False)

proj_orig = orig_embeddings @ V_orig[:, 0]
proj_vae = vae_embeddings @ V_vae[:, 0]

sign_corr = np.corrcoef(proj_orig, proj_vae)[0, 1]
if sign_corr < 0:
    proj_vae = -proj_vae

t_stat, p_value = ttest_rel(proj_orig, proj_vae)
print(f"\nPaired t-test on PC1 projections:")
print(f"  t = {t_stat:.4f}, p = {p_value:.6f}")

C_orig = orig_embeddings @ orig_embeddings.T
C_vae = vae_embeddings @ vae_embeddings.T
_, S_orig, _ = np.linalg.svd(C_orig, full_matrices=False)
_, S_vae, _ = np.linalg.svd(C_vae, full_matrices=False)

var_ratio_orig = S_orig[0] / S_orig.sum()
var_ratio_vae = S_vae[0] / S_vae.sum()
top5_orig = S_orig[:5].sum() / S_orig.sum()
top5_vae = S_vae[:5].sum() / S_vae.sum()
passed = bool(p_value <= 0.05)

results = {
    "var_ratio_orig": round(float(var_ratio_orig), 6),
    "var_ratio_vae": round(float(var_ratio_vae), 6),
    "var_ratio_diff": round(float(var_ratio_orig - var_ratio_vae), 6),
    "top5_ratio_orig": round(float(top5_orig), 6),
    "top5_ratio_vae": round(float(top5_vae), 6),
    "t_statistic": round(float(t_stat), 6),
    "p_value": round(float(p_value), 6),
    "n_pairs": N_PAIRS,
    "n_dims": orig_embeddings.shape[1],
    "threshold_passed": passed
}

with open(os.path.join(OUT_DIR, "idea_e_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nDone.")
