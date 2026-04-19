import torch.nn as nn
import torch

class ResidualBlock2d(nn.Module):
    """2D Residual block with skip connection."""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, stride=1):
        super().__init__()
        padding = ((kernel_size[0] - 1) * dilation // 2, (kernel_size[1] - 1) * dilation // 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection for channel/stride changes
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out

class SleepCNN(nn.Module):
    def __init__(self, channels=4, num_classes=3):
        super().__init__()
        
        # ========== MULTI-SCALE INPUT LAYER ==========
        # Tři paralelní kernely různých velikostí pro zachycení různých frekvencí
        # Kernely: (frekvence, čas) - menší v čase, větší v frekvencích
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=(7,3), stride=(2,1), padding=(3,1)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=(3,7), stride=(2,1), padding=(1,3)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=(5,5), stride=(2,1), padding=(2,2)),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )
        ])
        
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        
        self.res_block1 = ResidualBlock2d(48, 64, kernel_size=(5,3), dilation=1)
        self.res_block2 = ResidualBlock2d(64, 64, kernel_size=(5,3), dilation=1)
        self.res_block3 = ResidualBlock2d(64, 128, kernel_size=(5,3), dilation=1, stride=2)
        self.res_block4 = ResidualBlock2d(128, 128, kernel_size=(3,3), dilation=1)
        self.res_block5 = ResidualBlock2d(128, 256, kernel_size=(3,3), dilation=1, stride=2)
        
        # Classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x_scales = [layer(x) for layer in self.multi_scale]
        x = torch.cat(x_scales, dim=1)
        x = self.maxpool(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
