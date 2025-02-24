from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷定义层卷积层,1个输入通道，6个输出通道，5*5的filter,28+2+2=32
        # 左右、上下填充padding
        # MNIST图像大小28，LeNet大小是32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # 定义第二层卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 定义3个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播
    def forward(self, img):
        # 先卷积，再调用relue激活函数，然后再最大化池化
        x = F.max_pool2d(F.relu(self.conv1(img)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # num_flat_features=16*5*5
        # x = x.view(-1, self.num_flat_features(x))

        # 第一个全连接
        x = F.relu(self.fc1(x.view(img.shape[0], -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.softmax(x, dim=1)
def Net():
    return LeNet()