from fastai import *
from fastai_audio import *


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, residual=False):
        super().__init__()
        self.res = residual
        self.conv = nn.Conv2d(in_channels, 
                         out_channels,
                         kernel_size=kernel_size, 
                         stride=stride,
                         padding=padding, 
                         bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.res:
            out = residual + out
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class AudioCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        layers = []

        # B1 to B5
        in_channels = 1
        num_filters = [16, 32, 64, 128, 256]
        for out_channels in num_filters:
            layers += [ResBlock(in_channels,  out_channels, kernel_size=3, padding=1, residual=False),
                       ResBlock(out_channels, out_channels, kernel_size=3, padding=1, residual=True),
                       nn.MaxPool2d(2)]
            in_channels = out_channels
        
        # B6
        layers += [ResBlock(256, 512, kernel_size=3, padding=1), nn.MaxPool2d(2)]
        
        # F1
        layers += [ResBlock(512, 1024, kernel_size=3, padding=1)]
        
        # F2
        layers += [nn.Conv2d(1024, n_classes, 3, padding=1)]
        
        # Reshape 
        layers += [
            PoolFlatten()
        ]
                
        self.layers = nn.Sequential(*layers)

        # from ResNet 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        
    def forward(self, x):
        return self.layers(x)