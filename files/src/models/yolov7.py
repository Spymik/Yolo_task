import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # YOLOv7 uses SiLU instead of LeakyReLU
        
    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

class ELAN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN_Block, self).__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBlock(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        
        # Final 1x1 conv to combine features
        self.final_conv = ConvBlock(mid_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        concat = torch.cat([x1, x3, x5, x6], dim=1)
        return self.final_conv(concat)

class YOLOv7(nn.Module):
    def __init__(self, grid_size=7, num_boxes=3, num_classes=80):
        super(YOLOv7, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            # Initial convolution
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            
            # First E-ELAN stage
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ELAN_Block(128, 128),
            
            # Second E-ELAN stage
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ELAN_Block(256, 256),
            
            # Third E-ELAN stage
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ELAN_Block(512, 512),
        )
        
        self.conv_layers = nn.Sequential(
            # Fourth E-ELAN stage
            ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ELAN_Block(1024, 1024),
            
            # Additional convolutions for feature refinement
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * (grid_size // 32) ** 2, 4096),
            nn.Dropout(0.5),
            nn.SiLU(),
            nn.Linear(4096, grid_size * grid_size * (num_boxes * 5 + num_classes)),
        )
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
        return x

# Example usage:
# model = YOLOv7(grid_size=7, num_boxes=3, num_classes=80)
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
