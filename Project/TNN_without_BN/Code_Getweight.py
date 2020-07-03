################ MINHNVL ########################
############### CAID LAB  #######################
########### Lenet - Cifar10 #####################
############## 06/2020 ##########################
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
import numpy as np
import os
import struct
Path_model = "./model_Cifar10/Lenet_Cifar10_001.pth"
Path_weight = "./weight/"


# Class model Lenet using Cifar-10 datadset
class LeNet_Cifar10(nn.Module):
    def __init__(self):
        super(LeNet_Cifar10, self).__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

######## Convert Ternary #######
def Ternarize(tensor):
    output = torch.zeros(tensor.size())
    delta = Delta(tensor)
    alpha = Alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1,-1):
            pos_one = (w > delta[i]).type(torch.FloatTensor)
            neg_one = -1 * (w < -delta[i]).type(torch.FloatTensor)
        out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
        output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
    return output   

def Alpha(tensor,delta):
        Alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1,-1).abs()
            for w in absvalue:
                truth_value = w > delta[i] #print to see
            count = truth_value.sum()
            abssum = torch.matmul(absvalue,truth_value.type(torch.FloatTensor).view(-1,1))
            Alpha.append(abssum/count)
        alpha = Alpha[0]
        for i in range(len(Alpha) - 1):
            alpha = torch.cat((alpha,Alpha[i+1]))
        return alpha

def Delta(tensor):
    n = tensor[0].nelement()
    if(len(tensor.size()) == 4):     #convolution layer
        delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
    elif(len(tensor.size()) == 2):   #fc layer
        delta = 0.7 * tensor.norm(1,1).div(n)
    # print(delta)
    return delta
###########################

############## Save .dat ##############
def save_params(filename, data):
    with open(filename, 'w') as f:
        f.write("\n".join(data))
    f.close()
    print('Save file:', filename)
######################

###### Get Weight form model #####
def Get_weight():
    model = LeNet_Cifar10()
    model.load_state_dict(torch.load(Path_model))
    model.eval()
    params = model.named_parameters()
    return params
if __name__ == "__main__":
    params = Get_weight()
    if os.path.exists(Path_weight) == False:
        os.mkdir(Path_weight)
    # print(model.conv2.weight.data)
    # a = Ternarize(model.conv2.weight.data)

    ### Convert weight from floating point to ternary and save ####
    for name, data_weight in params:
        if ("bias" in name):
            convert = list(map(lambda x: str(x.item()),list(data_weight.flatten())))
        else:
            print("Coverting %s to Ternary...." %name)
            ter_data = Ternarize(data_weight)
            convert = list(map(lambda x: "1" if x > 0 else "-1" if x < 0 else "0",ter_data.flatten()))
        save_params(Path_weight +  name.replace(".","_") + ".dat", convert)

################ MINHNVL ########################
############### CAID LAB  #######################
########### Lenet - Cifar10 #####################
############## 06/2020 ##########################