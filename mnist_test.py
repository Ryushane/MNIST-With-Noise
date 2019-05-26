import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 输入尺寸 1*28*28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,2), # padding = 2 让输出尺寸等于输入尺寸
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 前向传播 输入为x
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出为维度为1的值，将其铺成一维向量
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
