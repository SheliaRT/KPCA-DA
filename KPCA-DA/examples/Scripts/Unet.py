import torch
import torch.nn as nn


class U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(U_Net, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ##nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(32,16,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,1,kernel_size=3, padding=1),
             nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        # Middle
        x2 = self.middle(x1)
        # Decoder
        x3 = self.decoder(x2)
        return x3

# Example usage
# Assuming input image with 3 channels and output segmentation mask with 1 channel
in_channels = 3
out_channels = 1
model = U_Net(in_channels, out_channels)
input_tensor = torch.randn((1, in_channels, 256, 256))  # Batch size of 1, image size 256x256
output_tensor = model(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape:", output_tensor.shape)
