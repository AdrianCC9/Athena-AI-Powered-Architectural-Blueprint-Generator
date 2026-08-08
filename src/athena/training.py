import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time
import json
from pathlib import Path

from .defaults import DEFAULT_LOSS_LOG, resolve_device
from .floorplan_dataset import FloorplanDataset
from .model import AthenaModel


def train(
    text_tokens_path,
    image_tensors_path,
    checkpoint_path,
    epochs=50,
    batch_size=8,
    learning_rate=0.0002,
    device=None,
    augment=False,
    limit=None,
    text_model_name="t5-small",
    pretrained_backbone=True,
    loss_log_path=DEFAULT_LOSS_LOG,
):
    device = resolve_device(device)
    print(f"[SETUP] Using device: {device}")

    print("\n[INFO] Loading dataset...")
    dataset = FloorplanDataset(text_tokens_path, image_tensors_path, augment=augment)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if len(dataloader) == 0:
        raise ValueError("No training batches available. Check the dataset inputs.")

    print(f"[INFO] Dataset loaded. Total samples: {len(dataset)}")
    print(f"[INFO] Steps per epoch: {len(dataloader)} (batch size = {batch_size})")

    print("\n[INFO] Initializing AthenaModel...")
    model = AthenaModel(
        text_model_name=text_model_name,
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    print("[INFO] Model initialized and moved to device.")

    print("\n[INFO] Defining loss function and optimizer...")
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n[INFO] Starting training for {epochs} epochs...")
    total_steps = epochs * len(dataloader)
    global_step = 0

    start_time = time.time()
    loss_log = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0

        print(f"\n=== EPOCH [{epoch + 1}/{epochs}] ===")

        for batch_idx, (text_input, real_image) in enumerate(dataloader):
            global_step += 1
            text_input = text_input.to(device)
            real_image = real_image.to(device)
            optimizer.zero_grad(set_to_none=True)
            generated_image = model(text_input, real_image)
            loss = loss_function(generated_image, real_image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_percent = (global_step / total_steps) * 100

            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"Batch [{batch_idx + 1}/{len(dataloader)}] | "
                f"Step {global_step}/{total_steps} "
                f"({progress_percent:.2f}% done) | "
                f"Batch Loss: {loss.item():.6f}"
            )

        avg_loss = total_loss / len(dataloader)
        loss_log.append(avg_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"[EPOCH SUMMARY] Epoch {epoch + 1} finished.")
        print(f"  - Average Epoch Loss: {avg_loss:.6f}")
        print(f"  - Epoch Duration: {epoch_duration:.1f} sec")

    total_duration = time.time() - start_time
    print("\n[INFO] Training complete!")
    print(f"Total training time: {total_duration:.1f} seconds")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Final model saved at {checkpoint_path}")

    if loss_log_path:
        loss_log_path = Path(loss_log_path)
        loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(loss_log_path, "w", encoding="utf-8") as f:
            json.dump(loss_log, f, indent=2)
        print(f"[INFO] Loss log saved at {loss_log_path}")

    return checkpoint_path


def main():
    from .defaults import DEFAULT_CHECKPOINT, DEFAULT_PROCESSED_IMAGES, DEFAULT_TOKENIZED_TEXTS

    train(DEFAULT_TOKENIZED_TEXTS, DEFAULT_PROCESSED_IMAGES, DEFAULT_CHECKPOINT)


if __name__ == "__main__":
    main()
