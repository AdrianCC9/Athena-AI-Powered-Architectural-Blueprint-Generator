import random
from pathlib import Path

from PIL import Image


def visualize_images(image_dir, num_images=6, output_path=None):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if output_path:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_files = [path for path in image_dir.iterdir() if path.suffix.lower() == ".png"]
    if not all_files:
        raise ValueError(f"No PNG images found in {image_dir}")
    random.shuffle(all_files)
    selected_files = all_files[:num_images]

    plt.figure(figsize=(12, 6))
    columns = max(1, min(3, len(selected_files)))
    rows = (len(selected_files) + columns - 1) // columns
    for i, img_path in enumerate(selected_files, 1):
        img = Image.open(img_path).convert("RGB")
        plt.subplot(rows, columns, i)
        plt.imshow(img)
        plt.title(img_path.name)
        plt.axis("off")

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"[INFO] Preview saved to {output_path}")
    else:
        plt.show()


def main():
    from .defaults import DEFAULT_IMAGES_DIR, OUTPUTS_DIR

    print("=== Previewing Blueprint Images ===")
    visualize_images(DEFAULT_IMAGES_DIR, 6, OUTPUTS_DIR / "image_preview.png")

if __name__ == "__main__":
    main()
