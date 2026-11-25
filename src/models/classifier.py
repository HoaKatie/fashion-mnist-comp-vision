"""Small CNN trained on real images, used only to evaluate the generated ones (not one of the compared models)"""
import torch.nn as nn

def _block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

class EvalClassifier(nn.Module):
    def __init__(self, channels, num_classes, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            *_block(channels, 32), *_block(32, 32), nn.MaxPool2d(2),  # 16x16
            *_block(32, 64), *_block(64, 64), nn.MaxPool2d(2),        # 8x8
            *_block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feature_dim), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(feature_dim, num_classes)

    def embed(self, x):
        """Feature vector used for FID / precision / recall"""
        return self.features(x)

    def forward(self, x):
        return self.fc(self.dropout(self.embed(x)))