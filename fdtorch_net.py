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

