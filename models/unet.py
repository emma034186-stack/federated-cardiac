import torch
import torch.nn as nn
import torch.nn.functional as F
from config import IN_CHANNELS, NUM_CLASSES


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Lightweight 2-D U-Net for cardiac MRI segmentation.
    Encoder: 4 downsampling stages (16 → 32 → 64 → 128 → 256 channels)
    Decoder: 4 upsampling stages with skip connections
    Output: NUM_CLASSES logits per pixel
    """

    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_ch=16):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(5)]  # [16, 32, 64, 128, 256]

        # Encoder
        self.enc1 = DoubleConv(in_channels, ch[0])
        self.enc2 = DoubleConv(ch[0], ch[1])
        self.enc3 = DoubleConv(ch[1], ch[2])
        self.enc4 = DoubleConv(ch[2], ch[3])
        self.bottleneck = DoubleConv(ch[3], ch[4])
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4   = nn.ConvTranspose2d(ch[4], ch[3], 2, stride=2)
        self.dec4  = DoubleConv(ch[4], ch[3])
        self.up3   = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3  = DoubleConv(ch[3], ch[2])
        self.up2   = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2  = DoubleConv(ch[2], ch[1])
        self.up1   = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1  = DoubleConv(ch[1], ch[0])

        self.final = nn.Conv2d(ch[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)


def build_model():
    return UNet()
