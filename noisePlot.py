from mnist_test import *
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_acc = np.zeros(0)
# 不同噪声测试数量
TESTNUM = 10
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
    download=False,
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
    download=False,
    transform=transform)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )

# loss function and optimizer
# net = LeNet().to(device)
# criterion = nn.CrossEntropyLoss() #交叉熵损失函数
# optimizer = optim.SGD(net.parameters(), lr = LR, momentum = 0.9)
net = LeNet()
net.load_state_dict(torch.load('./model/net_008.pth', map_location='cpu'))

def accget(sigma):
    accuracy = []
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            # 这里load进来的也是tensor
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images += torch.rand(images.size()) * sigma
            # 加入噪声后如果超出区间，则进行截取
            images = np.minimum(images, 1)
            # np.where(images>255, images, 255)
            # print(np.shape(images))
            # for i, itensor in enumerate(images):
            #     for j, jtensor in enumerate(itensor):
            #         for k, ktensor in enumerate(jtensor):
            #             for l, ltensor in enumerate(ktensor):
            #                 if(ltensor > 1):
            #                     images[i][j][k][l] = 1
            # print(images.size())
            # print(images)
            outputs = net(images)
            # 按行选最大的值虚区，并返回其索引
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        acc_rate = 100 * correct / total
    return acc_rate


if __name__ == "__main__":
    acc = []
    for i in range(TESTNUM):
        #np.append(multi_acc, accget(i/TESTNUM))
        acc.append(accget(i/TESTNUM))
    #print(acc)
    x = np.arange(0,1,1/TESTNUM)
        # np.append(multi_acc,acc_mean)
    plt.plot(x, acc)
    plt.savefig("acc_noise")
    plt.title("Accurancy in different noise intensity")
    plt.xlabel("Intensity of noise")
    plt.ylabel("Accurancy on test data")
    plt.show()