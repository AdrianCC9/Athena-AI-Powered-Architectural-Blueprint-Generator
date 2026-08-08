from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer

from .defaults import resolve_device
from .model import load_model


def generate_blueprint(
    prompt,
    checkpoint_path,
    output_path,
    reference_image=None,
    device=None,
    image_size=(256, 256),
    max_length=128,
    text_model_name="t5-small",
):
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    text_input = tokens["input_ids"].to(device)
    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    image_tensor = load_reference_image(reference_image, image_size).unsqueeze(0).to(device)

    print("[INFO] Loading model...")
    model = load_model(checkpoint_path, device, text_model_name=text_model_name)

    print("[INFO] Generating blueprint...")
    with torch.no_grad():
        generated = model(text_input, image_tensor, attention_mask=attention_mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tensor_image(generated[0], output_path)
    print(f"[INFO] Generated blueprint saved to {output_path}")
    return output_path


def load_reference_image(reference_image, image_size):
    if reference_image is None:
        return torch.ones(1, image_size[1], image_size[0])

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    with Image.open(reference_image) as image:
        return transform(image.convert("L"))


def save_tensor_image(tensor, output_path):
    tensor = tensor.detach().cpu().clamp(-1.0, 1.0)
    tensor = (tensor + 1.0) / 2.0
    image = transforms.ToPILImage()(tensor)
    image.save(output_path)
