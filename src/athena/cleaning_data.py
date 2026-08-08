import csv
import pickle
from pathlib import Path

import pandas as pd

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_artificial_annotations(pkl_path):
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    records = []
    for entry in data_list:
        image_id = entry.get("image_id")
        short_description = entry.get("short_version", {}).get("string", "").strip()

        if image_id and short_description:
            records.append({
                "image_id": image_id,
                "annotation": short_description
            })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} artificial annotations from PKL.")
    return df


def validate_images(df, images_dir):
    images_dir = Path(images_dir)
    valid_rows = []
    missing_count = 0

    for _, row in df.iterrows():
        resolved_image_id = resolve_image_id(row["image_id"], images_dir)
        if resolved_image_id:
            row = row.copy()
            row["image_id"] = resolved_image_id
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[WARNING] {missing_count} records were dropped because the corresponding .png was not found.")

    return pd.DataFrame(valid_rows, columns=df.columns)


def resolve_image_id(image_id, images_dir):
    image_id = str(image_id)
    candidates = [Path(image_id)]
    if not Path(image_id).suffix:
        candidates.extend(Path(f"{image_id}{suffix}") for suffix in IMAGE_EXTENSIONS)

    for candidate in candidates:
        path = images_dir / candidate
        if path.exists():
            return path.name
    return None


def clean_annotations(pkl_path, images_dir, output_csv):
    print("=== Starting Data Cleaning ===")
    pkl_path = Path(pkl_path)
    images_dir = Path(images_dir)
    output_csv = Path(output_csv)

    if not pkl_path.exists():
        raise FileNotFoundError(f"Annotation PKL not found: {pkl_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    df = load_artificial_annotations(pkl_path)
    df = validate_images(df, images_dir)
    print(f"[INFO] After validation, {len(df)} valid image-text pairs remain.")

    before = len(df)
    df.drop_duplicates(subset=["image_id", "annotation"], inplace=True)
    df.dropna(subset=["annotation"], inplace=True)
    after = len(df)
    print(f"[INFO] Dropped {before - after} duplicate/empty rows. Total now: {len(df)}.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"[INFO] Cleaned data saved to {output_csv}")
    print("=== Data Cleaning Complete ===")
    return output_csv


def main():
    from .defaults import DEFAULT_ARTIFICIAL_PKL, DEFAULT_CLEANED_CSV, DEFAULT_IMAGES_DIR

    clean_annotations(DEFAULT_ARTIFICIAL_PKL, DEFAULT_IMAGES_DIR, DEFAULT_CLEANED_CSV)


if __name__ == "__main__":
    main()
