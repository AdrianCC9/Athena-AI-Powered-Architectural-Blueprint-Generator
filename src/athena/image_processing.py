from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_and_preprocess_images(image_dir, image_size=(256, 256)):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    image_tensors = {}

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(image_path) as image:
            image_tensors[image_path.name] = transform(image.convert("L"))

    print(f"[INFO] Processed {len(image_tensors)} images.")
    return image_tensors


def preprocess_images(image_dir, output_pt, image_size=(256, 256)):
    print("=== Starting Image Preprocessing ===")
    output_pt = Path(output_pt)
    image_tensors = load_and_preprocess_images(image_dir, image_size=image_size)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(image_tensors, output_pt)
    print(f"[INFO] Processed images saved to {output_pt}")
    print("=== Image Preprocessing Complete ===")
    return output_pt


def main():
    from .defaults import DEFAULT_IMAGES_DIR, DEFAULT_PROCESSED_IMAGES

    preprocess_images(DEFAULT_IMAGES_DIR, DEFAULT_PROCESSED_IMAGES)

if __name__ == "__main__":
    main()
