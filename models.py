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
    def __init__(self, input_channels=1, output_channels=4):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.left_conv_1 = DoubleConv(input_channels, 48)

        self.left_conv_2 = DoubleConv(48, 96)

        self.left_conv_3 = DoubleConv(96, 192)

        self.left_conv_4 = DoubleConv(192, 384)


        self.center_conv = DoubleConv(384, 768)


        self.up_1 = nn.ConvTranspose2d(768, 384, 2, 2)
        self.right_conv_1 = DoubleConv(768, 384)

        self.up_2 = nn.ConvTranspose2d(384, 192, 2, 2)
        self.right_conv_2 = DoubleConv(384, 192)

        self.up_3 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.right_conv_3 = DoubleConv(192, 96)

        self.up_4 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.right_conv_4 = DoubleConv(96, 48)


        self.output = nn.Conv2d(48, output_channels, 1, 1, 0)

    def forward(self, x):

        x1 = self.left_conv_1(x)
        x1_down = self.pool(x1)

        x2 = self.left_conv_2(x1_down)
        x2_down = self.pool(x2)

        x3 = self.left_conv_3(x2_down)
        x3_down = self.pool(x3)

        x4 = self.left_conv_4(x3_down)
        x4_down = self.pool(x4)


        x5 = self.center_conv(x4_down)


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

        output = self.output(x9)

        return output


class UnetPlusPlus(nn.Module):
    def __init__(self, input_channels=1, output_channels=4, deep_supervision=False):
        super().__init__()

        self.deep_supervision = deep_supervision
        
        self.conv_3_1 = DoubleConv(512*2, 512)
 
        self.conv_2_2 = DoubleConv(256*3, 256)
        self.conv_2_1 = DoubleConv(256*2, 256)
 
        self.conv_1_1 = DoubleConv(128*2, 128)
        self.conv_1_2 = DoubleConv(128*3, 128)
        self.conv_1_3 = DoubleConv(128*4, 128)
 
        self.conv_0_1 = DoubleConv(64*2, 64)
        self.conv_0_2 = DoubleConv(64*3, 64)
        self.conv_0_3 = DoubleConv(64*4, 64)
        self.conv_0_4 = DoubleConv(64*5, 64)
 
 
        self.stage_0 = DoubleConv(input_channels, 64)
        self.stage_1 = DoubleConv(64, 128)
        self.stage_2 = DoubleConv(128, 256)
        self.stage_3 = DoubleConv(256, 512)
        self.stage_4 = DoubleConv(512, 1024)
 
        self.pool = nn.MaxPool2d(2, 2)
    
        self.up_3_1 = nn.ConvTranspose2d(1024, 512, 4, 2, 1) 
        self.up_2_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1) 
        self.up_2_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1) 
        self.up_1_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) 
        self.up_1_2 = nn.ConvTranspose2d(256, 128, 4, 2, 1) 
        self.up_1_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1) 
        self.up_0_1 = nn.ConvTranspose2d(128, 64, 4, 2, 1) 
        self.up_0_2 = nn.ConvTranspose2d(128, 64, 4, 2, 1) 
        self.up_0_3 = nn.ConvTranspose2d(128, 64, 4, 2, 1) 
        self.up_0_4 = nn.ConvTranspose2d(128, 64, 4, 2, 1) 
 

        self.final_super_0_1 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_channels, 3, padding=1),
        )        
        self.final_super_0_2 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_channels, 3, padding=1),
        )        
        self.final_super_0_3 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_channels, 3, padding=1),
        )        
        self.final_super_0_4 = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, output_channels, 3, padding=1),
        )        
 
  
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.up_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.conv_0_1(x_0_1)
        
        x_1_1 = torch.cat([self.up_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.conv_1_1(x_1_1)
        
        x_2_1 = torch.cat([self.up_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.conv_2_1(x_2_1)
        
        x_3_1 = torch.cat([self.up_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.conv_3_1(x_3_1)
 
        x_2_2 = torch.cat([self.up_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.conv_2_2(x_2_2)
        
        x_1_2 = torch.cat([self.up_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.conv_1_2(x_1_2)
        
        x_1_3 = torch.cat([self.up_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.conv_1_3(x_1_3)
 
        x_0_2 = torch.cat([self.up_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.conv_0_2(x_0_2)
        
        x_0_3 = torch.cat([self.up_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.conv_0_3(x_0_3)
        
        x_0_4 = torch.cat([self.up_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.conv_0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)
    