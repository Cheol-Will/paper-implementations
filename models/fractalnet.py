import torch
import torch.nn as nn
import torch.nn.functional as f

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class FractalBlock(nn.Module):
    """
        Buidling Block of Fractal Structure 
        Xavier, Initialization and local drop path with p = 0.15
    """
    def __init__(self, col, in_channels, out_channels, join=True):
        super(FractalBlock, self).__init__()
        self.join = join
        self.col = col # for Drop-path
        self.out_channels = out_channels
        if col == 2:
            self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=3, padding=1)
        else: 
            self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = FractalBlock(col-1, in_channels, out_channels)
            self.conv3 = FractalBlock(col-1, out_channels, out_channels, join=False)

    def local_drop(self, X, p_drop):
        p_alive = torch.full((X.shape[1],), 1 - p_drop, dtype=X.dtype, device=X.device)
        idx = torch.bernoulli(p_alive).bool()
        if not any(idx):
            idx[0] = True
        X = X[:, idx, :, :, :]

        if len(X.shape) == 4:
            X = X.unsqueeze(1) # If only one path is selected, change dim (batch_size, channels, heigth, width) -> (batch_size, 1, channels, heigth, width)  

        return X

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv3(self.conv2(x))
        out = torch.cat([a, b], dim=1)

        if self.join:
            batch_size, concat_channels, height, width = out.shape
            out = out.view(batch_size, concat_channels//self.out_channels, self.out_channels, height, width)
            out = self.local_drop(out, p_drop=0.15) 
            out = out.mean(dim=1) # avg join over groups

        return out

class FractalNet(nn.Module):

    def __init__(self, col, channel_list, num_classes=10):
        super(FractalNet, self).__init__()
        self.fractal_blocks = self._make_layer(col, channel_list)
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel_list[4], num_classes)
        )

    def _make_layer(self, col, channel_list):
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(channel_list):
            layers.append(FractalBlock(col, in_channels, out_channels))
            layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

            if i < len(channel_list)-1:
                # transition layer
                layers.append(ConvBlock(in_channels, channel_list[i+1], kernel_size=1, padding=1))
                in_channels = channel_list[i+1]

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.fractal_blocks(x)
        x = self.clf(x)

        return x