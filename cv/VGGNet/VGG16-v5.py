
from torchvision import datasets,transforms
from torchvision.utils import  make_grid
from torch.utils.data import DataLoader
from torch import nn,optim,tensor
from matplotlib import pyplot as plt
import numpy as np
from torch import nn
import torch
import time

class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )


        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc=nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256,10)
        )

    def forward(self,x):
        x=self.conv(x)
        x = x.view(-1, 512)
        x=self.fc(x)
        return x

epoch_num=10
batch_size=32
lr=0.01
step_size=10
num_print=100

def transforms_RandomHorizontalFlip():
    transform_train=transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    transform=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.485,0.456,0.406),(0.226,0.224,0.225))])

    train_dataset=datasets.CIFAR10(root='data',train=True,transform=transform_train,download=True)
    test_dataset=datasets.CIFAR10(root='data',train=False,transform=transform,download=True)

    return train_dataset,test_dataset

train_dataset,test_dataset=transforms_RandomHorizontalFlip()

train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=Vgg16_net().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=lr,momentum=0.8,weight_decay=0.001)
schedule=optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)

loss_list=[]
start=time.time()
for epoch in range(epoch_num):
    ww = 0
    running_loss=0.0
    for i,(inputs,labels) in enumerate(train_loader,0):
        inputs,labels=inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss=criterion(outputs,labels).to(device)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        loss_list.append(loss.item())

        if(i+1)%num_print==0:
            print('[%d epoch,%d]  loss:%.6f' %(epoch+1,i+1,running_loss/num_print))
            running_loss=0.0

    lr_1=optimizer.param_groups[0]['lr']
    print("learn_rate:%.15f"%lr_1)
    schedule.step()

end=time.time()
print("time:{}".format(end-start))

model.eval()
correct=0.0
total=0
with torch.no_grad():
    print("=======================test=======================")
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=model(inputs)

        pred=outputs.argmax(dim=1)
        total+=inputs.size(0)
        correct+=torch.eq(pred,labels).sum().item()

print("Accuracy of the network on the 10000 test images:%.2f %%" %(100*correct/total) )
print("===============================================")

