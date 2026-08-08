from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import T5Model


class AthenaModel(nn.Module):
    def __init__(
        self,
        text_embedding_dim=512,
        image_feature_dim=512,
        latent_dim=1024,
        text_model_name="t5-small",
        pretrained_backbone=True,
    ):
        super().__init__()

        self.text_encoder = T5Model.from_pretrained(text_model_name)
        self.text_projection = nn.Linear(
            self.text_encoder.config.d_model,
            text_embedding_dim,
        )

        resnet = _resnet18(pretrained_backbone)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_projection = nn.Linear(512, image_feature_dim)

        self.fusion_layer = nn.Linear(text_embedding_dim + image_feature_dim, latent_dim)
        self.fusion_activation = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, text_tokens, image_tensor, attention_mask=None):
        text_features = self.text_encoder.encoder(
            input_ids=text_tokens,
            attention_mask=attention_mask,
        ).last_hidden_state
        text_features = text_features.mean(dim=1)
        text_features = self.text_projection(text_features)

        image_features = self.image_encoder(image_tensor)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_projection(image_features)

        fused_features = torch.cat((text_features, image_features), dim=1)
        fused_features = self.fusion_activation(self.fusion_layer(fused_features))
        fused_features = fused_features.view(fused_features.size(0), -1, 1, 1)
        return self.decoder(fused_features)


def load_model(checkpoint_path, device, text_model_name="t5-small", pretrained_backbone=False):
    model = AthenaModel(
        text_model_name=text_model_name,
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _resnet18(pretrained):
    if not pretrained:
        try:
            return models.resnet18(weights=None)
        except TypeError:
            return models.resnet18(pretrained=False)

    try:
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except (AttributeError, TypeError):
        return models.resnet18(pretrained=True)
