'''ResNet-18 Image classfication for cifar-10 with PyTorch

Author 'Sun-qian'.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1): #需要确定输入维度和输出维度，步长默认为1
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(  #残差主体部分
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False), #第一个卷积把输入维度转成输出维度，步长可变
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),#第二个卷积输入和输出的维度都是一样的，都为输出维度，步长为1
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()  #一般而言，shortcut不做操作，输入x就输出x
        if stride != 1 or inchannel != outchannel: #但是block的步长万一不是1 或者 block的输入维度与输出维度不一致
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), #shortcut的卷积，若步长不为1或者输入维度与输出维度不一致，就会触发
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):  #block的前向反馈，包含主体的2个卷积和shortcut可能触发的一个卷积
        out = self.left(x)     #主体
        out += self.shortcut(x)#shortcut
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64  #输入网络后，遇到的第一个卷积是3x3，64层的
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False), #输入维度是3，因为一开始输入是RGB，卷积是3x3，64
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1) #4个卷积是 3x3，64层的，其中有两个block，每个block2个卷积
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)#4个卷积是 3x3，128层的，其中有两个block，每个block2个卷积
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)#4个卷积是 3x3，256层的，其中有两个block，每个block2个卷积
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)#4个卷积是 3x3，512层的，其中有两个block，每个block2个卷积
        self.fc = nn.Linear(512, num_classes)   #最后的全连接层，512层要转成10层，因为cifar数据集是10分类的

    def make_layer(self, block, channels, num_blocks, stride):#制造layer，一个layer 两个block
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1],或[2,1]
        layers = []
        for stride in strides:#制造不同步长(1或者2)的block，一个block两个卷积，可能触发shortcut的第三个卷积（l不同layer之间的卷积会触发）
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)             #只有最开始的一个卷积
        out = self.layer1(out)          #4个64层3x3卷积
        out = self.layer2(out)          #4个128层3x3卷积
        out = self.layer3(out)          #4个256层3x3卷积
        out = self.layer4(out)          #4个512层3x3卷积
        out = F.avg_pool2d(out, 4)      #最后的输出是4x4的，所以这里第二个参数为4
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock)

