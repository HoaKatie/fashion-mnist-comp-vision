"""Define the CNN that recognizes the ten Fashion-MNIST clothing classes.

train_classifier.py teaches this network using real labels. compare.py uses
the saved CNN both to describe images in feature space and to count the
predicted clothing classes. The classifier is separate from the GAN's
real-or-fake discriminator.
"""

import torch.nn as nn

def _block(c_in, c_out):
    # Look at nearby pixels, normalize each feature channel, and apply ReLU.
    # Padding keeps both spatial dimensions the same through this block.
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]

class EvalClassifier(nn.Module):
    def __init__(self, channels, num_classes, feature_dim=128):
        super().__init__()
        # 1. Learn local features at each size. Max pooling halves width and height.
        self.features = nn.Sequential(
            *_block(channels, 32), *_block(32, 32), nn.MaxPool2d(2),  # 16x16
            *_block(32, 64), *_block(64, 64), nn.MaxPool2d(2),        # 8x8
            # 2. Average each channel over the image and turn it into a feature vector.
            *_block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feature_dim), nn.ReLU(inplace=True),
        )
        # 3. Drop some features during training. Then give each clothing class a score.
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(feature_dim, num_classes)

    def embed(self, x):
        # These features are what we compare when we score two sets of images.
        return self.features(x)

    def forward(self, x):
        # The training loss takes the raw class scores from the final layer.
        return self.fc(self.dropout(self.embed(x)))
