import numpy as np
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

import torch
from torch.utils.data import Subset

from .defaults import resolve_device
from .floorplan_dataset import FloorplanDataset
from .model import load_model


def evaluate(
    checkpoint_path,
    text_tokens_path,
    image_tensors_path,
    batch_size=8,
    max_samples=5000,
    device=None,
    output_json=None,
    text_model_name="t5-small",
):
    device = resolve_device(device)
    print(f"[SETUP] Using device: {device}")

    print("[INFO] Loading trained model...")
    model = load_model(checkpoint_path, device, text_model_name=text_model_name)
    print("[INFO] Model loaded successfully.")

    print("[INFO] Loading dataset...")
    dataset = FloorplanDataset(text_tokens_path, image_tensors_path)
    if max_samples:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if len(dataloader) == 0:
        raise ValueError("No evaluation batches available. Check the dataset inputs.")

    print(f"[INFO] Using {len(dataset)} samples for evaluation.")

    mse_scores = []
    psnr_scores = []
    ssim_scores = []
    cosine_scores = []

    print("\n[INFO] Evaluating model performance...")
    with torch.no_grad():
        for batch_idx, (text_input, real_images) in enumerate(dataloader, start=1):
            text_input = text_input.to(device)
            real_images = real_images.to(device)
            generated_images = model(text_input, real_images)

            for i in range(len(real_images)):
                mse, psnr, ssim_score = compute_image_metrics(real_images[i], generated_images[i])
                mse_scores.append(mse)
                psnr_scores.append(psnr)
                if ssim_score is not None:
                    ssim_scores.append(ssim_score)

            cosine_scores.append(cosine_similarity(real_images, generated_images))
            print(f"[INFO] Batch {batch_idx}/{len(dataloader)} complete.")

    results = {
        "mse": float(np.mean(mse_scores)),
        "psnr": float(np.mean(psnr_scores)),
        "cosine_similarity": float(np.mean(cosine_scores)),
    }
    if ssim_scores:
        results["ssim"] = float(np.mean(ssim_scores))

    print("\n=== Evaluation Results ===")
    print(f"Average MSE: {results['mse']:.6f} (lower is better)")
    print(f"Average PSNR: {results['psnr']:.2f} dB (higher is better)")
    print(f"Average Cosine Similarity: {results['cosine_similarity']:.4f} (higher is better)")
    if "ssim" in results:
        print(f"Average SSIM: {results['ssim']:.4f} (higher is better)")
    else:
        print("Average SSIM: skipped because scikit-image is not installed")

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Metrics saved to {output_json}")

    return results


def compute_image_metrics(real, generated):
    real_np = real.detach().cpu().squeeze().numpy()
    generated_np = generated.detach().cpu().squeeze().numpy()
    mse = float(np.mean((real_np - generated_np) ** 2))
    psnr = float(20 * np.log10(2.0 / np.sqrt(mse))) if mse > 0 else 100.0
    return mse, psnr, compute_ssim(real_np, generated_np)


def compute_ssim(real_np, generated_np):
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return None

    return float(structural_similarity(real_np, generated_np, data_range=2.0))


def cosine_similarity(tensor1, tensor2):
    tensor1 = tensor1.reshape(1, -1)
    tensor2 = tensor2.reshape(1, -1)
    return float(F.cosine_similarity(tensor1, tensor2).item())


def main():
    from .defaults import DEFAULT_CHECKPOINT, DEFAULT_PROCESSED_IMAGES, DEFAULT_TOKENIZED_TEXTS

    evaluate(DEFAULT_CHECKPOINT, DEFAULT_TOKENIZED_TEXTS, DEFAULT_PROCESSED_IMAGES)


if __name__ == "__main__":
    main()

