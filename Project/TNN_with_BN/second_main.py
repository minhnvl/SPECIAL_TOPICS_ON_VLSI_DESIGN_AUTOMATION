import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
import model as M
import util as U

def ParseArgs():
    parser = argparse.ArgumentParser(description='Ternary-Weights-Network Pytorch Cifar10 Example.')
    parser.add_argument('--batch-size',type=int,default=128,metavar='N',
                        help='batch size for training(default: 100)')
    parser.add_argument('--test-batch-size',type=int,default=100,metavar='N',
                        help='batch size for testing(default: 100)')
    parser.add_argument('--epochs',type=int,default=100,metavar='N',
                        help='number of epoch to train(default: 100)')
    parser.add_argument('--lr-epochs',type=int,default=20,metavar='N',
                        help='number of epochs to decay learning rate(default: 20)')
    parser.add_argument('--lr',type=float,default=1e-3,metavar='LR',
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--momentum',type=float,default=0.9,metavar='M',
                        help='SGD momentum(default: 0.9)')
    parser.add_argument('--weight-decay','--wd',type=float,default=1e-5,metavar='WD',
                        help='weight decay(default: 1e-5)')
    parser.add_argument('--no-cuda',action='store_true',default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed',type=int,default=2,metavar='S',
                        help='random seed(default: 1)')
    parser.add_argument('--log-interval',type=int,default=100,metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def main():
    args = ParseArgs()
    if args.cuda:
        print("GPU is used: %d"%(args.seed))
        torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # print(args.batch_size)
    BATCH_SIZE = args.batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    learning_rate = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    ###################################################################
    ##             Load Train Dataset                                ##
    ###################################################################
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=BATCH_SIZE, shuffle=True,**kwargs)
    ###################################################################
    ##             Load Test Dataset                                ##
    ###################################################################
    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar_data', train=False, download=False,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=True,**kwargs)

    model = M.LeNet5()
    if args.cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    #optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    ternarize_op = U.TernarizeOp(model)
    
    best_acc = 0.0 

    List_loss = []
    List_Percent_Training = []
    List_Percent_Testing = []
    for epoch_index in range(1,args.epochs+1):
        adjust_learning_rate(learning_rate,optimizer,epoch_index,args.lr_epochs)
        Accuracy_Training, train_loss = train(args,epoch_index,train_loader,model,optimizer,criterion,ternarize_op)
        Accuracy_Testing = test(args,model,test_loader,criterion,ternarize_op)
        if Accuracy_Testing > best_acc:
            best_acc = Accuracy_Testing
            ternarize_op.Ternarization()
            ternarize_op.Restore()
            U.save_model(model,epoch_index)
        List_Percent_Training.append(Accuracy_Training)
        List_Percent_Testing.append(Accuracy_Testing)
        List_loss.append(train_loss)
    return List_Percent_Training, List_Percent_Testing, List_loss
    
def train(args,epoch_index,train_loader,model,optimizer,criterion,ternarize_op):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)

        optimizer.zero_grad()
        ternarize_op.Ternarization()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        ternarize_op.Restore()
        optimizer.step()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        train_loss += criterion(output,target).item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_index, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        

    Accuracy_Training = 100. * correct/len(train_loader.dataset)
    
    print('\nTraining Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        Accuracy_Training))
    return Accuracy_Training, train_loss
def test(args,model,test_loader,criterion,ternarize_op):
    model.eval()
    test_loss = 0
    correct = 0

    ternarize_op.Ternarization()
    for data,target in test_loader:
        if args.cuda:
            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)
        output = model(data)
        
        test_loss += criterion(output,target).item()
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    Accuracy_Testing = 100. * correct/len(test_loader.dataset)

    test_loss /= len(test_loader)
    print('Testing set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return Accuracy_Testing
    
def adjust_learning_rate(learning_rate,optimizer,epoch_index,lr_epoch):
    lr = learning_rate * (0.1 ** (epoch_index // lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        return lr
def Show_Grapth(List_Percent_Training, List_Percent_Testing, List_loss):
    # List_Percent_Training, List_Percent_Testing, List_loss = Processing_Training()
    fig, axs = plt.subplots(2)
    fig.suptitle('Accuracy and Loss with BN')
    axs[0].plot(List_Percent_Training,label = "Training")
    axs[0].plot(List_Percent_Testing,label = "Testing")
    axs[0].legend(loc='lower right')
    axs[0].set_ylabel('Accuracy (%)')
    # axs[0].set_xlabel('Epoch')
    # axs[0].set_ylabel('%')
    axs[0].set_ylim([0,100])
    axs[1].plot(List_loss)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    plt.show()
if __name__ == '__main__':
    List_Percent_Training, List_Percent_Testing, List_loss = main()
    Show_Grapth(List_Percent_Training, List_Percent_Testing, List_loss)
