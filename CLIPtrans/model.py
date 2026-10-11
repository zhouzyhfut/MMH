import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def get_input_dims_from_dataset(dataset):
    audio_path = dataset.train_color_image[0]
    text_path = dataset.train_thermal_image[0]
    video_path = dataset.train_depth_image[0]

    audio = np.load(audio_path).mean(axis=0)
    text = np.load(text_path).mean(axis=0)
    video = pd.read_csv(video_path).values.mean(axis=0)
    return [audio.shape[-1], text.shape[-1], video.shape[-1]]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        return x.squeeze(0)


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, output_dim=512, validate_input=True):
        super().__init__()
        self.input_dim = input_dim
        self.validate_input = validate_input
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
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        if self.validate_input and x.shape[1] != self.input_dim:
            return
        x = x.unsqueeze(1)
        x = self.conv_stack(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class AVTCrossAttentionFixed(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_a, x_t, x_v):
        q = x_a.unsqueeze(1)
        kv = torch.stack([x_v, x_t], dim=1)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = self.norm1(q + self.dropout(attn_out))
        fused = x + self.dropout(self.mlp(x))
        return self.norm2(fused).squeeze(1)


class embed_net(nn.Module):
    def __init__(self, class_num, dataset=None, cnn_output_dim=512):
        super().__init__()
        assert dataset is not None, 'dataset must be provided to infer input dims'
        input_dims = get_input_dims_from_dataset(dataset)
        self.input_dims = input_dims
        self.imagecnn1 = SimpleCNN(input_dim=714, output_dim=cnn_output_dim, validate_input=False)
        self.imagetrans1 = TransformerEncoder(input_dim=cnn_output_dim)
        self.cnn1 = SimpleCNN(input_dims[0], cnn_output_dim)
        self.cnn2 = SimpleCNN(input_dims[1], cnn_output_dim)
        total_fused_dim = cnn_output_dim
        self.transformer = TransformerEncoder(input_dim=total_fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, class_num),
        )
        self.crossattention = AVTCrossAttentionFixed(cnn_output_dim)
        self.cls_audio = nn.Linear(cnn_output_dim, class_num)
        self.cls_text = nn.Linear(cnn_output_dim, class_num)
        self.cls_video = nn.Linear(cnn_output_dim, class_num)
        self.trans_audio = TransformerEncoder(input_dim=cnn_output_dim)
        self.trans_text = TransformerEncoder(input_dim=cnn_output_dim)

    def forward(self, audio=None, text=None, video=None):
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        audio1 = self.trans_audio(self.cnn1(audio))
        pred1 = self.cls_audio(audio1)

        if text.dim() == 3:
            text = text.mean(dim=1)
        text1 = self.trans_text(self.cnn2(text))
        pred2 = self.cls_text(text1)

        if video.dim() == 3:
            video = video.mean(dim=1)
        video1 = self.encode_video(video)
        pred3 = self.cls_video(video1)

        logits = self.classifier(self.crossattention(audio1, text1, video1))
        return logits, pred1, pred2, pred3, audio1, text1, video1

    def encode_video(self, video):
        return self.imagetrans1(self.imagecnn1(video))
