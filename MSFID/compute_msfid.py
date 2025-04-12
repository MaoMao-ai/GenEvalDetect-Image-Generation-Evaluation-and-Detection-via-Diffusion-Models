import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from scipy.linalg import sqrtm
from msfid_model import InceptionV3MultiFeature


def compute_fid_and_msfid(real_folder, fake_folder, model_path=None, max_images=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    def load_images(folder, transform, max_images=500):
        from tqdm import tqdm
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_paths = []

        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))

        image_paths = sorted(image_paths)[:max_images]
        if not image_paths:
            raise RuntimeError(f"❌ No valid images found in {folder}")

        images = []
        for path in tqdm(image_paths, desc=f"Loading images from {folder}"):
            img = Image.open(path).convert("RGB")
            images.append(transform(img))

        return torch.stack(images)

    def compute_statistics(feats):
        if feats.dim() == 4:
            feats = F.adaptive_avg_pool2d(feats, output_size=(1, 1)).squeeze(-1).squeeze(-1)
        feats = feats.cpu().numpy()
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    def frechet_distance(mu1, sigma1, mu2, sigma2):
        try:
            covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        except Exception as e:
            print(f"❌ Error computing sqrtm: {e}")
            return float('inf')

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    real_images = load_images(real_folder, transform, max_images).to(device)
    fake_images = load_images(fake_folder, transform, max_images).to(device)

    model = InceptionV3MultiFeature()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        real_feats = model(real_images)
        fake_feats = model(fake_images)

    msfid = 0.0
    fid = None
    layer_names = model.layers if hasattr(model, "layers") else [f"Layer_{i}" for i in range(len(real_feats))]

    for i, (rf, ff) in enumerate(zip(real_feats, fake_feats)):
        mu_r, sigma_r = compute_statistics(rf)
        mu_f, sigma_f = compute_statistics(ff)
        d = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)

        print(f"✓ MSFID[{layer_names[i]}]: {d:.4f}")
        msfid += d

        if layer_names[i] == "Mixed_7c":
            fid = d

    msfid /= len(real_feats)

    if fid is None:
        print("⚠️ Mixed_7c not found. FID will use MSFID as fallback.")
        fid = msfid

    print(f"\n✅ Final FID (Mixed_7c): {fid:.4f}")
    print(f"✅ Final MSFID (mean over layers): {msfid:.4f}")

    return fid, msfid


def log_to_csv(experiment_name, fid, msfid, csv_path="results.csv"):
    import csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment_name", "fid", "msfid"])
        writer.writerow([experiment_name, f"{fid:.4f}", f"{msfid:.4f}"])
