import numpy as np
import torch
import torch.nn as nn
import torchvision
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        return x.squeeze(0)


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_dim)
    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)  
        x = self.fc(x)  
        return x


class CLIPVAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.2
        ndim_proj = 128
        #part124
        self.imagecnn1=SimpleCNN(input_dim=714)
        self.imagetrans1=TransformerEncoder()
        #part35
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.backbone = model.float()
    def forward(self, x1,cliptext=None):
        #print(cliptext)
        cliptext_features=self.encode_text(cliptext)
        #print(cliptext_features)
        image_features=self.encode_image(x1)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        cliptext_features = cliptext_features / cliptext_features.norm(dim=1, keepdim=True)
        #print(logits)
        logits_per_image = (image_features @ cliptext_features.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
    def encode_text(self, text):
        encoded_text = self.backbone.encode_text(text)
        return encoded_text
    def encode_image(self, image):
        image_features1 = self.imagecnn1(image)
        image_features1 = self.imagetrans1(image_features1)
        return image_features1

