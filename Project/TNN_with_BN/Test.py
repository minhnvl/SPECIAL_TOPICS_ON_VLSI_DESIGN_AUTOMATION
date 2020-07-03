import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np



def Pre_training():
    transform = transforms.ToTensor()
    EPOCH , BATCH_SIZE, LR = Parameter_Training()

    trainset = tv.datasets.CIFAR10(
        root='./cifar_data',
        train=True,
        download=True,
        transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        )
    testset = tv.datasets.CIFAR10(
        root='./cifar_data',
        train=False,
        download=True,
        transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        )
    
    return trainset, trainloader, testset, testloader
def Parameter_Training():
    EPOCH = 1
    BATCH_SIZE = 32     
    LR = 0.001  
    
    return EPOCH , BATCH_SIZE, LR
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5,self).__init__()
        self.conv1 = nn.Conv2d(3,32,kernel_size = 5)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,kernel_size = 5)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*5*5,512)
        self.fc2 = nn.Linear(512,10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.bn_conv1(x),2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(self.bn_conv2(x),2))
        x = x.view(-1,64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  
Label_Cifar10 = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = 125
trainset, trainloader, testset, testloader = Pre_training()
EPOCH , BATCH_SIZE, LR = Parameter_Training()

PATH_Model = "./model_Cifar10/Lenet_Cifar10_022.pth"
net = LeNet5()

net.load_state_dict(torch.load(PATH_Model))
net.eval()
imagetest = trainset[index][0]
labeltest = trainset[index][1] 

Image_to_test = imagetest.unsqueeze(0)
outputs = net(Image_to_test)
_, predicted = torch.max(outputs.data, 1)

List_x = []
List_percent = []
sumx = 0
for x in outputs.data[0]:
    if (x < 0):
        x =0.0
    sumx += x
    List_x.append(float(x))
List_percent = [y/sumx for y in List_x]
fig, axs = plt.subplots(1,2,figsize=(14,6))

Result_predict = Label_Cifar10[int(predicted[0])]
ImageShow= imagetest.numpy()
ImageShow= np.transpose(ImageShow, (1, 2, 0))
rects1 = axs[1].bar(Label_Cifar10, List_percent, color='b')
text = 'The result is the %s' %Result_predict

print(text)
axs[1].set_ylim([0,0.8])
axs[0].imshow(ImageShow)
axs[0].set_xlabel(Result_predict)
axs[0].set_xticks([])
axs[0].set_yticks([])
plt.show()
