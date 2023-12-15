import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # left
        self.left_conv_1 = DoubleConv(1, 48)
        self.down_1 = nn.MaxPool2d(2, 2)

        self.left_conv_2 = DoubleConv(48, 96)
        self.down_2 = nn.MaxPool2d(2, 2)

        self.left_conv_3 = DoubleConv(96, 192)
        self.down_3 = nn.MaxPool2d(2, 2)

        self.left_conv_4 = DoubleConv(192, 384)
        self.down_4 = nn.MaxPool2d(2, 2)

        # center
        self.center_conv = DoubleConv(384, 768)

        # right
        self.up_1 = nn.ConvTranspose2d(768, 384, 2, 2)
        self.right_conv_1 = DoubleConv(768, 384)

        self.up_2 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.right_conv_2 = DoubleConv(384, 192)

        self.up_3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.right_conv_3 = DoubleConv(192, 96)

        self.up_4 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.right_conv_4 = DoubleConv(96, 48)

        # output
        self.output = nn.Conv2d(48, 4, 1, 1, 0)

    def forward(self, x):
        # left
        x1 = self.left_conv_1(x)
        x1_down = self.down_1(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.down_2(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.down_3(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.down_4(x4)

        # center
        x5 = self.center_conv(x4_down)

        # right
        x6_up = self.up_1(x5)
        temp = torch.cat((x6_up, x4), dim=1)
        x6 = self.right_conv_1(temp)

        x7_up = self.up_2(x6)
        temp = torch.cat((x7_up, x3), dim=1)
        x7 = self.right_conv_2(temp)

        x8_up = self.up_3(x7)
        temp = torch.cat((x8_up, x2), dim=1)
        x8 = self.right_conv_3(temp)

        x9_up = self.up_4(x8)
        temp = torch.cat((x9_up, x1), dim=1)
        x9 = self.right_conv_4(temp)

        # output
        output = self.output(x9)

        return output
    