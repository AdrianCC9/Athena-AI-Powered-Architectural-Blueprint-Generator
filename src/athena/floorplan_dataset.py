from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FloorplanDataset(Dataset):
    def __init__(self, text_tokens_path, image_tensors_path, augment=False):
        text_tokens_path = Path(text_tokens_path)
        image_tensors_path = Path(image_tensors_path)
        if not text_tokens_path.exists():
            raise FileNotFoundError(f"Tokenized text file not found: {text_tokens_path}")
        if not image_tensors_path.exists():
            raise FileNotFoundError(f"Processed image tensor file not found: {image_tensors_path}")

        text_data = torch.load(text_tokens_path, map_location="cpu")
        self.image_ids = text_data["image_ids"]
        self.tokenized_texts = text_data["tokenized_texts"]
        self.image_tensors = torch.load(image_tensors_path, map_location="cpu")

        if len(self.image_ids) != len(self.tokenized_texts):
            raise ValueError(
                "Mismatch between image IDs "
                f"({len(self.image_ids)}) and tokenized texts ({len(self.tokenized_texts)})."
            )

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])
        text_tensor = self.tokenized_texts[idx]
        image_tensor = self._get_image_tensor(image_id)
        if image_tensor is None:
            sample_keys = list(self.image_tensors.keys())[:5]
            raise KeyError(
                f"Image tensor not found for {image_id}. Sample available keys: {sample_keys}"
            )

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return text_tensor, image_tensor

    def _get_image_tensor(self, image_id):
        candidates = [image_id, Path(image_id).name]
        if not Path(image_id).suffix:
            candidates.extend([f"{image_id}.png", f"{Path(image_id).name}.png"])
        for candidate in candidates:
            image_tensor = self.image_tensors.get(candidate)
            if image_tensor is not None:
                return image_tensor
        return None
