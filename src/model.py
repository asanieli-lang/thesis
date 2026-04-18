import torch.nn as nn
import torch

class SleepCNN(nn.Module):
    def __init__(self, channels=4, num_classes=3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=16, kernel_size=65, stride=2, padding=32),
            nn.BatchNorm1d(16), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4) 
        )
                
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=17, stride=2, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        out =self.classifier(x)
        return out
        

if __name__ == "__main__":
    model = SleepCNN()
    # Vygenerujeme falešná data: 8 epoch (batch_size), 4 kanály, 2000 vzorků
    dummy_data = torch.randn(8, 4, 2000) 
    
    predictions = model(dummy_data)
    print(f"Tvar vstupu: {dummy_data.shape}")
    print(f"Tvar výstupu: {predictions.shape}")