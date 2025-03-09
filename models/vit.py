import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16  # Using a standard ViT model

# Define the Vision Transformer Model
class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=16):
        super(VisionTransformerModel, self).__init__()
        self.vit = vit_b_16(pretrained=False)
        self.vit.heads = nn.Linear(self.vit.hidden_dim, num_classes)  # Adjust for regression or classification

    def forward(self, x):
        return self.vit(x)
