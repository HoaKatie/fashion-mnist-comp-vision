"""CNN for the ten clothing classes. I use its features and labels to compare the generated images."""

import torch.nn as nn


def _block(c_in, c_out):
    # Find small patterns with a convolution, then use batch norm and ReLU
    # Padding keeps the width and height the same here
    return [
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]


class EvalClassifier(nn.Module):
    def __init__(self, channels, num_classes, feature_dim=128):
        super().__init__()
        # Max pooling cuts width and height in half after each of the first two blocks
        self.features = nn.Sequential(
            *_block(channels, 32),
            *_block(32, 32),
            nn.MaxPool2d(2),  # Down to 16 by 16
            *_block(32, 64),
            *_block(64, 64),
            nn.MaxPool2d(2),  # Down to 8 by 8
            # Average each channel over space, then make the 128 features used by default
            *_block(64, 128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        # Drop some features during training, then give each class a score
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(feature_dim, num_classes)

    def embed(self, x):
        # Keep these features for comparing real and generated images
        return self.features(x)

    def forward(self, x):
        # Pass the raw scores to the training loss
        return self.fc(self.dropout(self.embed(x)))
