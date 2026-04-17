"""
Idea D: Cross-model confound separation using Gaussian noise perturbation
==========================================================================

Hypothesis: CLIP's r=0.9537 vs DINOv2 cross-encoder's r=0.6117 gap may be
due to CLIP's same-model inflation artifact.

Method: Add increasing Gaussian noise (σ=0.01, 0.05, 0.1, 0.2) to image pairs.
If DINOv2 L2 is sensitive to pixel noise but CLIP CS is not (or less so),
then DINOv2 L2 captures pixel-level changes, not semantic changes.

Interpretation:
- noise → DINOv2 L2 monotonic increase → DINOv2 is pixel-sensitive
- noise → CLIP CS monotonic decrease → CLIP is also pixel-sensitive
- If both are sensitive → both measure pixel-level, not semantic-level
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from scipy.stats import pearsonr, spearmanr
import glob
import warnings
warnings.filterwarnings("ignore")

# Paths
RESEARCH_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0417-pm"
IMAGE_DIR = "/home/kas/.openclaw/workspace-domain/research/autonomous-research-window-0417-pm/img_samples"

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
N_PAIRS = 25  # number of image pairs to test (25 pairs from 60 images = C(60,2) subset)
N_SAMPLES_PER_PAIR = 3  # repetitions per pair per noise level for stability

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ─── Load Models ──────────────────────────────────────────────────────────────
print("Loading DINOv2 ViT-S/14 (autoencoder)...")
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-small")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-with-registers-small").to(device)
dinov2_model.eval()

print("Loading CLIP ViT-B/32...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_model.eval()
print("Models loaded.")

# ─── Load Images ──────────────────────────────────────────────────────────────
def load_image_paths(n=100):
    """Load image paths from img_samples directory."""
    paths = []
    
    # Try img_samples directory
    if os.path.exists(IMAGE_DIR):
        img_exts = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]
        for ext in img_exts:
            paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    # Fallback: try to find any images in research directory
    if len(paths) < 20:
        research_img = "/home/kas/.openclaw/workspace-domain/research"
        for root, dirs, files in os.walk(research_img):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(root, f))
    
    paths = list(set(paths))[:n * 2]  # dedupe and cap
    print(f"Found {len(paths)} images from available sources")
    return paths

def load_and_preprocess_images(paths):
    """Load PIL images from paths."""
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"  Skip {p}: {e}")
    return images

# ─── Feature Extraction ───────────────────────────────────────────────────────
def get_dinov2_embedding(image, processor, model):
    """Get DINOv2 embedding (CLS token)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token (first token)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return emb

def get_clip_embeddings(image, processor, model):
    """Get normalized CLIP image embeddings."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling
        if hasattr(outputs, 'pooler_output'):
            emb = outputs.pooler_output
        else:
            emb = outputs
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.squeeze().cpu().numpy()
    return emb

# ─── Gaussian Noise Perturbation ──────────────────────────────────────────────
def add_gaussian_noise(image_tensor, sigma):
    """Add Gaussian noise to a normalized image tensor. sigma is in [0,1] range."""
    if sigma == 0:
        return image_tensor
    noise = torch.randn_like(image_tensor) * sigma
    noisy = image_tensor + noise
    return torch.clamp(noisy, 0, 1)

def pil_to_tensor(image):
    """Convert PIL to normalized tensor."""
    arr = np.array(image).astype(np.float32) / 255.0
    # Convert HWC -> CHW
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)

def tensor_to_pil(tensor):
    """Convert normalized tensor back to PIL."""
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)

# ─── Distance/Similarity Metrics ─────────────────────────────────────────────
def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)

# ─── Main Experiment ──────────────────────────────────────────────────────────
def run_experiment():
    print("\n=== Idea D: Gaussian Noise Perturbation ===")
    
    # Load images
    img_paths = load_image_paths(n=100)
    if len(img_paths) < 10:
        print("ERROR: Not enough images found!")
        return None
    
    # Create image pairs (take first 2*N for pairs)
    n_pairs = min(N_PAIRS, len(img_paths) // 2)
    pair_paths = [(img_paths[i], img_paths[i+1]) for i in range(n_pairs)]
    print(f"Using {n_pairs} image pairs")
    
    # Storage: noise_level -> list of (dinov2_l2, clip_cs)
    results_by_noise = {str(sigma): [] for sigma in NOISE_LEVELS}
    # Also track per-pair delta from baseline
    pair_baselines = []  # (pair_idx, baseline_dinov2_l2, baseline_clip_cs)
    
    for pair_idx, (path_a, path_b) in enumerate(pair_paths):
        if pair_idx % 10 == 0:
            print(f"  Pair {pair_idx}/{n_pairs}")
        
        try:
            img_a = Image.open(path_a).convert("RGB")
            img_b = Image.open(path_b).convert("RGB")
        except Exception as e:
            print(f"  Skip pair {pair_idx}: {e}")
            continue
        
        # Baseline embeddings (no noise)
        emb_a_dinov2_base = get_dinov2_embedding(img_a, dinov2_processor, dinov2_model)
        emb_b_dinov2_base = get_dinov2_embedding(img_b, dinov2_processor, dinov2_model)
        emb_a_clip_base = get_clip_embeddings(img_a, clip_processor, clip_model)
        emb_b_clip_base = get_clip_embeddings(img_b, clip_processor, clip_model)
        
        base_dinov2_l2 = l2_distance(emb_a_dinov2_base, emb_b_dinov2_base)
        base_clip_cs = cosine_similarity(emb_a_clip_base, emb_b_clip_base)
        
        pair_baselines.append({
            "pair_idx": pair_idx,
            "path_a": path_a,
            "path_b": path_b,
            "baseline_dinov2_l2": float(base_dinov2_l2),
            "baseline_clip_cs": float(base_clip_cs)
        })
        
        # For each noise level, average over N_SAMPLES_PER_PAIR repetitions
        for sigma in NOISE_LEVELS:
            sigma_dinov2_l2s = []
            sigma_clip_css = []
            
            for rep in range(N_SAMPLES_PER_PAIR):
                # Convert to tensors, add noise
                tens_a = pil_to_tensor(img_a)
                tens_b = pil_to_tensor(img_b)
                
                noisy_a = add_gaussian_noise(tens_a, sigma)
                noisy_b = add_gaussian_noise(tens_b, sigma)
                
                pil_a_noisy = tensor_to_pil(noisy_a)
                pil_b_noisy = tensor_to_pil(noisy_b)
                
                # Get embeddings
                emb_a_dinov2 = get_dinov2_embedding(pil_a_noisy, dinov2_processor, dinov2_model)
                emb_b_dinov2 = get_dinov2_embedding(pil_b_noisy, dinov2_processor, dinov2_model)
                emb_a_clip = get_clip_embeddings(pil_a_noisy, clip_processor, clip_model)
                emb_b_clip = get_clip_embeddings(pil_b_noisy, clip_processor, clip_model)
                
                l2 = l2_distance(emb_a_dinov2, emb_b_dinov2)
                cs = cosine_similarity(emb_a_clip, emb_b_clip)
                
                sigma_dinov2_l2s.append(l2)
                sigma_clip_css.append(cs)
            
            avg_l2 = np.mean(sigma_dinov2_l2s)
            avg_cs = np.mean(sigma_clip_css)
            
            results_by_noise[str(sigma)].append({
                "pair_idx": pair_idx,
                "dinov2_l2": float(avg_l2),
                "clip_cs": float(avg_cs)
            })
    
    return results_by_noise, pair_baselines

# ─── Analysis ─────────────────────────────────────────────────────────────────
def analyze_results(results_by_noise, pair_baselines):
    """Analyze noise sensitivity of each metric."""
    print("\n=== Analysis ===")
    
    # Aggregate statistics per noise level
    noise_stats = {}
    for sigma_str, records in results_by_noise.items():
        l2s = [r["dinov2_l2"] for r in records]
        css = [r["clip_cs"] for r in records]
        noise_stats[float(sigma_str)] = {
            "dinov2_l2_mean": float(np.mean(l2s)),
            "dinov2_l2_std": float(np.std(l2s)),
            "clip_cs_mean": float(np.mean(css)),
            "clip_cs_std": float(np.std(css)),
            "n_pairs": len(records)
        }
    
    # Print noise level summary
    print("\nNoise Level → Metric Means:")
    print(f"{'σ':>6} | {'DINOv2 L2 (mean±std)':>25} | {'CLIP CS (mean±std)':>25}")
    print("-" * 65)
    for sigma in sorted(noise_stats.keys()):
        s = noise_stats[sigma]
        print(f"{sigma:>6.2f} | {s['dinov2_l2_mean']:>10.4f} ± {s['dinov2_l2_std']:>8.4f} | "
              f"{s['clip_cs_mean']:>10.4f} ± {s['clip_cs_std']:>8.4f}")
    
    # Check monotonicity: correlation between noise level and metric change
    noise_levels_arr = np.array(sorted(noise_stats.keys()))
    
    # For each pair, compute delta from baseline
    # Then average delta across pairs
    deltas_from_baseline = {sigma: {"l2": [], "cs": []} for sigma in NOISE_LEVELS}
    
    baseline_map = {b["pair_idx"]: b for b in pair_baselines}
    
    for sigma_str, records in results_by_noise.items():
        sigma = float(sigma_str)
        if sigma == 0.0:
            continue
        for r in records:
            pid = r["pair_idx"]
            if pid not in baseline_map:
                continue
            base = baseline_map[pid]
            deltas_from_baseline[sigma]["l2"].append(r["dinov2_l2"] - base["baseline_dinov2_l2"])
            deltas_from_baseline[sigma]["cs"].append(r["clip_cs"] - base["baseline_clip_cs"])
    
    print("\n--- Delta from Baseline (noise σ vs baseline at σ=0) ---")
    print(f"{'σ':>6} | {'Δ DINOv2 L2 (mean)':>20} | {'Δ CLIP CS (mean)':>20}")
    print("-" * 55)
    for sigma in sorted(deltas_from_baseline.keys()):
        d = deltas_from_baseline[sigma]
        mean_l2_delta = np.mean(d["l2"]) if d["l2"] else 0
        mean_cs_delta = np.mean(d["cs"]) if d["cs"] else 0
        print(f"{sigma:>6.2f} | {mean_l2_delta:>20.4f} | {mean_cs_delta:>20.4f}")
    
    # Monotonicity test: correlation between sigma and delta
    sigmas_for_corr = sorted([s for s in deltas_from_baseline.keys() if s > 0])
    mean_l2_deltas = [np.mean(deltas_from_baseline[s]["l2"]) for s in sigmas_for_corr]
    mean_cs_deltas = [np.mean(deltas_from_baseline[s]["cs"]) for s in sigmas_for_corr]
    
    if len(sigmas_for_corr) >= 3:
        l2_monotonicity_r, l2_p = pearsonr(sigmas_for_corr, mean_l2_deltas)
        cs_monotonicity_r, cs_p = pearsonr(sigmas_for_corr, mean_cs_deltas)
        print(f"\nMonotonicity (Pearson r, sigma vs delta):")
        print(f"  DINOv2 L2: r={l2_monotonicity_r:.4f}, p={l2_p:.4f}")
        print(f"  CLIP CS:  r={cs_monotonicity_r:.4f}, p={cs_p:.4f}")
        
        # Interpretation
        print("\n--- Interpretation ---")
        if l2_monotonicity_r > 0.5 and l2_p < 0.1:
            print("✓ DINOv2 L2 shows significant monotonic increase with noise → pixel-sensitive")
        if cs_monotonicity_r < -0.5 and cs_p < 0.1:
            print("✓ CLIP CS shows significant monotonic decrease with noise → pixel-sensitive")
        if l2_monotonicity_r < 0.3:
            print("✗ DINOv2 L2 is NOT monotonic with noise → not purely pixel-level")
        if cs_monotonicity_r > -0.3:
            print("✗ CLIP CS is NOT decreasing with noise → less pixel-sensitive")
    
    # Correlation between the two metrics across all noisy conditions
    all_l2 = []
    all_cs = []
    for sigma_str, records in results_by_noise.items():
        for r in records:
            all_l2.append(r["dinov2_l2"])
            all_cs.append(r["clip_cs"])
    
    overall_r, overall_p = pearsonr(all_l2, all_cs)
    print(f"\nOverall correlation (DINOv2 L2 vs CLIP CS): r={overall_r:.4f}, p={overall_p:.4f}")
    
    # Correlation at baseline only
    baseline_l2 = [b["baseline_dinov2_l2"] for b in pair_baselines]
    baseline_cs = [b["baseline_clip_cs"] for b in pair_baselines]
    baseline_r, baseline_p = pearsonr(baseline_l2, baseline_cs)
    print(f"Baseline-only correlation: r={baseline_r:.4f}, p={baseline_p:.4f}")
    
    # Prepare output dict
    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_native(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    output = to_native({
        "experiment": "Idea D: Gaussian Noise Perturbation",
        "noise_levels": NOISE_LEVELS,
        "n_pairs": len(pair_baselines),
        "n_repetitions_per_pair": N_SAMPLES_PER_PAIR,
        "noise_stats_per_level": {str(k): v for k, v in noise_stats.items()},
        "baseline_pairs": pair_baselines,
        "deltas_from_baseline": {
            str(k): {"l2": v["l2"], "cs": v["cs"]}
            for k, v in deltas_from_baseline.items()
        },
        "monotonicity": {
            "dinov2_l2_r": float(l2_monotonicity_r) if len(sigmas_for_corr) >= 3 else None,
            "dinov2_l2_p": float(l2_p) if len(sigmas_for_corr) >= 3 else None,
            "clip_cs_r": float(cs_monotonicity_r) if len(sigmas_for_corr) >= 3 else None,
            "clip_cs_p": float(cs_p) if len(sigmas_for_corr) >= 3 else None,
        },
        "overall_correlation": {"r": float(overall_r), "p": float(overall_p)},
        "baseline_correlation": {"r": float(baseline_r), "p": float(baseline_p)},
        "interpretation": {
            "dinov2_pixel_sensitive": l2_monotonicity_r > 0.5 if len(sigmas_for_corr) >= 3 else "unknown",
            "clip_pixel_sensitive": cs_monotonicity_r < -0.5 if len(sigmas_for_corr) >= 3 else "unknown",
        }
    })
    
    return output

# ─── Save ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results, baselines = run_experiment()
    if results is None:
        print("Experiment failed - no results")
        exit(1)
    
    output = analyze_results(results, baselines)
    
    out_path = os.path.join(RESEARCH_DIR, "idea_d_cpu_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")
    print("\n=== FINAL SUMMARY ===")
    print(f"noise_levels: {NOISE_LEVELS}")
    print(f"n_pairs: {len(baselines)}")
    mono = output.get("monotonicity", {})
    print(f"DINOv2 L2 monotonicity r={mono.get('dinov2_l2_r','N/A')}, p={mono.get('dinov2_l2_p','N/A')}")
    print(f"CLIP CS monotonicity r={mono.get('clip_cs_r','N/A')}, p={mono.get('clip_cs_p','N/A')}")
    print(f"Overall DINOv2 L2 vs CLIP CS r={output['overall_correlation']['r']:.4f}")
    print(f"Baseline-only r={output['baseline_correlation']['r']:.4f}")
    interp = output.get("interpretation", {})
    print(f"Interpretation: DINOv2 pixel-sensitive={interp.get('dinov2_pixel_sensitive','?')}, "
          f"CLIP pixel-sensitive={interp.get('clip_pixel_sensitive','?')}")
