import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acc_list = []
# 定义网络结构

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
# 设置可以手动输入命令
parser = argparse.ArgumentParser()
# 模型保存路径
parser.add_argument('--outf',default='./model/', help='folder to output images and model checkpoints')
# 模型加载路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加载路径
opt = parser.parse_args()

# 超参数设置
EPOCH = 8 #
BATCH_SIZE  = 64 #
LR = 0.001 #学习率

# 数据预处理方式
transform = transforms.ToTensor()

# 定义训练数据集
trainset = tv.datasets.MNIST(
    root='data/',
    train=True,
    download=True,
    transform=transform)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root='data/',
    train=False,
    download=True,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# loss function and optimizer
net = LeNet().to(device)
criterion = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9)

# 开始训练
if __name__ == "__main__":
    # for epoch in range(EPOCH):
    #     sum_loss = 0.0
    #
    #     # load data
    #     for i, data in enumerate(trainloader):
    #         inputs, labels = data
    #         inputs, lables = inputs.to(device), labels.to(device)
    #
    #         # 梯度清零
    #         optimizer.zero_grad()
    #
    #         # 这里虚区的是tensor形式的
    #         outputs = net(inputs).cuda()
    #         labels = labels.cuda()
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 输出loss
    #         sum_loss += loss.item()
    #         if i % 100 == 99:
    #             print('[%d,%d] loss: %.03f'
    #                 %(epoch + 1, i+1, sum_loss / 100))
    #             sum_loss = 0.0
    #
    #         # 每个epoch输出一次准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                # 这里load进来的也是tensor
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                #print(images.size())
                # print(images)
                outputs = net(images)
                # 按行选最大的值虚区，并返回其索引
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print(100 * correct / total)
            # print('第%d个epoch的准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
            acc_list.append(100 * correct / total)
        # torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
    # plt.plot(acc_list)
    # plt.savefig("acc_curve.png")
    # plt.show()