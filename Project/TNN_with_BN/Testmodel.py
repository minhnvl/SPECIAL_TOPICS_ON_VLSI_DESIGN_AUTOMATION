import os
import random
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import util as U
import model as M

Path_model = "./model_Cifar10/Lenet_Cifar10_092.pth"
Path_weight = "./weight/"


model = M.LeNet5()
model.load_state_dict(torch.load(Path_model))
model.eval()
params = model.named_parameters()

print (len(enumerate(params)))
for name, data_weight in params:
    print(name)
    print(data_weight)
#     if ("bias" in name):
#         convert = list(map(lambda x: str(x.item()),list(data_weight.flatten())))
#     else:
#         ter_data = Ternarize(data_weight)
#     print(ter_data)
