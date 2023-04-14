from torch import nn

# 1 MNIST Net Module
class YLMnistNet(nn.Module):
    def __init__(self):
        super(YLMnistNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv1 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.pool0 = nn.AvgPool2d(kernel_size=(2, 2))
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.linear0 = nn.Linear(16*4*4, 120)
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.layers = [self.conv0, self.pool0, self.conv1, self.pool1, self.flatten, self.linear0, self.relu, self.linear1, self.relu, self.linear2, self.relu]

    def forward(self, x):
        output = self.conv0(x)
        output = self.pool0(output)
        output = self.conv1(output)
        output = self.pool1(output)
        output = self.flatten(output)
        output = self.linear0(output)
        output = self.relu(output)
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        return output

    # get depth of MNIST Net
    def __len__(self):
        return len(self.layers)

    # get specified layer
    def __getitem__(self, item):
        return self.layers[item]

    def __name__(self):
        return 'YNMNISTNET'


# 2 Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if not in_channel == out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

# 3 Residual net 10, acc is about 91.5%
class ResNet(nn.Module):
    '''
    ResidualNet 10
    '''
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.residual_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self.residual_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.residual_layer(ResidualBlock, 256, 1, stride=2)
        self.layer4 = self.residual_layer(ResidualBlock, 512, 1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def residual_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, kernel_size=4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

