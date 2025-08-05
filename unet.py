#UNet Model

import torch
import torch.nn as nn
from PIL import Image

def double_conv(in_channels, out_channels):
    """(Convolution => [BN] => ReLU) * 2"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_colors):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_colors = num_colors

        # The input to the first layer is the image channel + 1 color channel
        self.inc = double_conv(n_channels + 1, 64)

        # Encoder
        self.down1 = self.down_block(64, 128)
        self.down2 = self.down_block(128, 256)
        self.down3 = self.down_block(256, 512)
        self.down4 = self.down_block(512, 1024)

        # Decoder
        self.up1 = self.up_block(1024, 512)
        self.up2 = self.up_block(512, 256)
        self.up3 = self.up_block(256, 128)
        self.up4 = self.up_block(128, 64)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels)
        )

    def up_block(self, in_channels, out_channels):
        return nn.ModuleDict({
            'up': nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            'conv': double_conv(in_channels, out_channels)
        })

    def forward(self, x_image, x_color_idx):
        # x_image: (B, C, H, W) -> e.g., (16, 1, 128, 128)
        # x_color_idx: (B, 1) -> e.g., (16, 1)

        # 1. Create the color condition tensor
        color_condition = (x_color_idx.float() / (self.num_colors - 1)) * 2 - 1
        color_plane = color_condition.view(-1, 1, 1, 1).expand(-1, 1, x_image.size(2), x_image.size(3))

        # 2. Concatenate the image and the color plane
        x = torch.cat([x_image, color_plane], dim=1) # Shape: (B, C+1, H, W)

        # 3. U-Net Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1['up'](x5)
        x = torch.cat([x4, x], dim=1)
        x = self.up1['conv'](x)

        x = self.up2['up'](x)
        x = torch.cat([x3, x], dim=1)
        x = self.up2['conv'](x)

        x = self.up3['up'](x)
        x = torch.cat([x2, x], dim=1)
        x = self.up3['conv'](x)

        x = self.up4['up'](x)
        x = torch.cat([x1, x], dim=1)
        x = self.up4['conv'](x)

        logits = self.outc(x)
        return self.sigmoid(logits)