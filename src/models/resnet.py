import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Resnet8(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()

        self.initial_layer = nn.Sequential(nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(3, 1))
        self.residual_blocks = nn.ModuleList([ResidualBlock(16, 16) for _ in range(3)])
        self.avg_pooling = nn.AvgPool2d(30)
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.get_predictions(x)
        return x

    def get_embedding(self, x):
        x = self.initial_layer(x)
        x = self.residual_blocks[0](x)
        return x

    def get_predictions(self, x):
        x = self.residual_blocks[1](x)
        x = self.residual_blocks[2](x)
        x = self.avg_pooling(x).squeeze()
        x = self.classifier(x)
        return x

class ResidualBlock3(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, stride=1, downsample = None):
        super().__init__()
        if isinstance(stride, int):
            stride = [stride] * 3
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride =stride[0], padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride =stride[1], padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride =stride[2], padding=0),
            nn.BatchNorm2d(out_channels),

        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Resnet55(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        downsample1 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0)
        self.layer1_1 = ResidualBlock3(16, 16, 64, downsample=downsample1)
        self.layer1_2 = nn.ModuleList([ResidualBlock3(64, 16, 64, stride=1) for _ in range(5)])
        downsample2 = nn.Conv2d(64, 128, 1, 2)
        self.layer2_1 = ResidualBlock3(64, 32, 128, stride=[1,2,1], downsample=downsample2)
        self.layer2_2 = nn.ModuleList([ResidualBlock3(128, 32, 128, 1)])
        downsample3 = nn.Conv2d(128, 256, 1, stride=2)
        self.layer3_1 = ResidualBlock3(128, 64, 256, [1, 2, 1], downsample=downsample3)
        self.layer3_2 = nn.ModuleList([ResidualBlock3(256, 64, 256, 1) for _ in range(5)])
        self.avg_pool = nn.AvgPool2d(8, 1)
        self.classifier = nn.Linear(256, n_classes)


    def forward(self, x):
        x = self.layer1_1(x)
        for layer in self.layer1_2:
            x = layer(x)
        x = self.layer2_1(x)
        for layer in self.layer2_2:
            x = layer(x)
        x = self.layer3_1(x)
        for layer in self.layer3_2:
            x = layer(x)
        x = self.avg_pool(x).squeeze()
        x = self.classifier(x)
        return x