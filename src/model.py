import torch.nn as nn
import torch

class ResidualBlock1d(nn.Module):
    """Residual blok s skip connectionem pro 1D konvoluci."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - pokud se změní počet channels, projekce
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
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
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=65, stride=2, padding=32),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=129, stride=2, padding=64),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(channels, 16, kernel_size=257, stride=2, padding=128),
                nn.BatchNorm1d(16),
                nn.ReLU(),
            )
        ])
        
        # Concatenace 3 cest = 48 channels
        
        # ========== RESIDUAL BLOKY ==========
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.res_block1 = ResidualBlock1d(48, 64, kernel_size=17, dilation=1)
        self.res_block2 = ResidualBlock1d(64, 64, kernel_size=9, dilation=2)
        self.res_block3 = ResidualBlock1d(64, 128, kernel_size=9, dilation=2, stride=2)
        self.res_block4 = ResidualBlock1d(128, 128, kernel_size=9, dilation=4)
        self.res_block5 = ResidualBlock1d(128, 256, kernel_size=9, dilation=4, stride=2)
        
        # ========== CLASSIFIER ==========
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Multi-scale vstup
        x_scales = [layer(x) for layer in self.multi_scale]
        x = torch.cat(x_scales, dim=1)  # [batch, 48, time]
        
        # MaxPool po multi-scale
        x = self.maxpool(x)
        
        # Residual bloky
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        out = self.classifier(x)
        return out
        

if __name__ == "__main__":
    model = SleepCNN()
    # Vygenerujeme falešná data: 8 epoch (batch_size), 4 kanály, 2000 vzorků
    dummy_data = torch.randn(8, 4, 2000) 
    
    predictions = model(dummy_data)
    print(f"Tvar vstupu: {dummy_data.shape}")
    print(f"Tvar výstupu: {predictions.shape}")