import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        Q = self.query(x).view(batch_size, seq_len, 
                                self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, 
                              self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, 
                                self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
                                batch_size, seq_len, hidden_dim)
        context = self.out(context)

        return context[:, -1, :], attn_weights

class ResidualBlock2d(nn.Module):
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


class SequenceCNN(nn.Module):    
    def __init__(
        self,
        channels=4,
        num_classes=3,
        lstm_hidden=128,
        attn_dropout=0.1,
        num_heads=2,
        lstm_layers=2,
        lstm_dropout=0.5,
        classifier_dropout1=0.5,
        classifier_dropout2=0.3
    ):
        super().__init__()
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

        self.res_block1 = ResidualBlock2d(48, 64, kernel_size=(5,3))
        self.res_block2 = ResidualBlock2d(64, 64, kernel_size=(5,3))
        self.res_block3 = ResidualBlock2d(64, 128, kernel_size=(5,3), stride=2)
        self.res_block4 = ResidualBlock2d(128, 128, kernel_size=(3,3))
        self.res_block5 = ResidualBlock2d(128, 256, kernel_size=(3,3), stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        lstm_internal_dropout = lstm_dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_internal_dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout1),
            nn.Linear(lstm_hidden, 32), 
            nn.ReLU(),  
            nn.Dropout(p=classifier_dropout2), 
            nn.Linear(32, num_classes) 
        )
        self.attention = MultiHeadAttention(
            hidden_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=attn_dropout
        )
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.reshape(batch_size * seq_len, channels, height, width)
        x_scales = [layer(x) for layer in self.multi_scale]
        x = torch.cat(x_scales, dim=1)  
        
        x = self.maxpool(x)
        

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        
        x = self.global_avg_pool(x)
        x = x.view(batch_size * seq_len, -1) 
        
        x = x.reshape(batch_size, seq_len, 256)
        lstm_out, _ = self.lstm(x)                  
        context, attn_weights = self.attention(lstm_out) 
        
        x = self.classifier(context)
        return x
 