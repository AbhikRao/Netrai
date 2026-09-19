import torch
import torch.nn as nn
import timm


class NetrAIModel(nn.Module):
    """EfficientNet-B4 with dual-head output for DR grading."""
    def __init__(self, model_name='efficientnet_b4', num_classes=5, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.grade_head = nn.Linear(256, num_classes)
        self.ref_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.head(features)
        grade_logits = self.grade_head(features)
        ref_logits = self.ref_head(features).squeeze(1)
        return grade_logits, ref_logits
