import torch
import torch.nn as nn
import torch.nn.functional as func
import math
"""
Please refer to the main project document for further information of this classes
"""

class RBSRCNN(nn.Module):
    """
    Residual Block Super-Resolution CNN (RBSRCNN) for image super-resolution.
    This model uses a sequence of residual blocks to learn fine-grained details in super-resolving images.
    """
    def __init__(self):
        super(RBSRCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),  
            nn.ReLU(True)
        )

        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),  
            nn.ReLU(True)
        )

        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.residuals(out) + out
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))  
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))  
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual

class CNNM1(nn.Module):
    """
    SRCNN as Model 1 (CNNM1) for image super-resolution.
    A straightforward CNN architecture designed for enhancing image resolution with minimal layers.
    """
    def __init__(self) -> None:
        super(CNNM1, self).__init__()
        self.features = nn.Sequential(    
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)), 
            nn.ReLU(True)
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
        

class CNNM2(nn.Module):
    """
    ExSRCNN as Model 2 (CNNM2) for image super-resolution.
    Features a deeper architecture with multiple convolutional layers and a batch normalization layer 
    to enhance the model's ability to learn complex transformations.
    """
    def __init__(self):
        super(CNNM2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (7,7), (1,1), (3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        layers = []
        for _ in range(15): 
            layers.append(nn.Conv2d(64, 64,(3,3), (1,1), (1,1)))
            if _ == 8:
              layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(True)
            )
        self.map = nn.Sequential(*layers)


        self.reconstruction = nn.Sequential(
            nn.Conv2d(64,1 ,(3,3), (1,1), (1,1)) )

    def forward(self, x):
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

        return out
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
        
