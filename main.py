from __future__ import print_function, division
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from BagDataset.Train import BagDataset
from BagDataset.Test_noncontrol import BagDataset1
from BagDataset.Test_control import BagDataset2
cudnn.benchmark = True
plt.ion()   # interactive mode
from tqdm import tqdm
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import visdom
from fcn import VGGNet, FCN8s, FCNs,UNet,NestedUNet
import seaborn as sns
import matplotlib.patches as patches
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)
from configuration import lr,momentum,w_decay,step_size,gamma,epo_num, batch_size,ratio,ratio1,ALPHA,BETA,GAMMA ,K

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


# 下面开始训练网络



# 在训练网络前定义函数用于计算Acc 和 mIou
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

 # 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu






def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated



def distance(inp):
    """Imshow for Tensor."""#torch.Size([1, 1, 56, 112])
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated








def train(epo_num, show_vgg_params=False):
    #实例化数据集
    #vis = visdom.Visdom()


    transform = transforms.Compose([
    transforms.ToTensor(), 
    #transforms.RandomRotation((180,180),center=(34,53)),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
    #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
    #transform = transforms.Compose([
    #transforms.ToTensor(), 
    #transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])
     


    #transform = transforms.Compose([
    #transforms.ToTensor(), 
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    bag = BagDataset(transform)
    print('Train',len(bag))
    train_dataset=bag
    
    
    bag1 = BagDataset1(transform)
    print('Test_noncontrol',len(bag1))
    test_dataset=bag1
   
    bag2 = BagDataset2(transform)
    print('Test_control',len(bag2))
    test_dataset_control=bag2   
    
    
    
    
  
    #利用DataLoader生成一个分batch获取数据的可迭代对象
    
    
    
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)
    test_dataloader_control = DataLoader(test_dataset_control, batch_size, shuffle=False, num_workers=4)

  



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    #fcn_model = FCN8s(pretrained_net=vgg_model, n_class=2)
    
    #vgg_model = VGGNet(requires_grad=True, remove_fc=True)    
    #fcn_model = FCNs(pretrained_net=vgg_model, n_class=1)
    
    
    fcn_model = UNet(num_classes=1, input_channels=5)
    
    
    fcn_model = fcn_model.to(device)
    #criterion = nn.BCELoss()  
    # 这里只有两类，采用二分类常用的损失函数BCE
    criterion = FocalTverskyLoss()
    criterion1 = nn.L1Loss() 
    #criterion2=nn.L1Loss(reduction='mean')
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss() 
    #criterion = nn.BCELoss()  
    # 随机梯度下降优化，学习率0.001，惯性分数0.7
    #optimizer = optim.SGD(fcn_model.parameters(), lr=1e-4, momentum=0.7)


    #optimizer = optim.SGD(fcn_model.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    optimizer = optim.Adam(fcn_model.parameters(), lr=lr,weight_decay=w_decay)
    #optimizer = optim.Adagrad(fcn_model.parameters(), lr=lr)
    #optimizer = optim.SGD(fcn_model.parameters(), lr=lr,
     #                           momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    # 记录训练过程相关指标
    all_train_loss = []
    all_train_loss_l1 = []
    all_test_loss = []
    all_test_loss_control = []
    test_Acc = []
    test_mIou = []
    # start timing
    prev_time = datetime.now()
    eps=[]
    #for epo in tqdm(range(epo_num)):
    for epo in range(1,epo_num+1):
        
        eps.append(epo)
        # 训练
        train_loss = 0
        train_loss_l1 = 0
        fcn_model.train()
        for index, (bag, bag_msk,name,locx,locy,locpx,locpy,imgC,imgD,loc4px,loc4py,shuttle_vx,shuttle_vy) in enumerate(train_dataloader):
            
            
            M=locy
          
            bag = bag.to(device)
            
            bag_msk = bag_msk.to(device)
            
            optimizer.zero_grad()
            output = fcn_model(bag)
            #output = nn.Softmax(dim=1)(output)
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            #print(output.shape)#torch.Size([16, 1, 56, 112])
            #print(a)
            output1=output.clone().detach()
          #  output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            #output = nn.Softmax(dim=1)(output)
           # print(output.shape)
           # print(a)
            loss_batch=0
            loss_batch_l1=0
            for j in range(len(bag_msk)): 
                    
                    #print(bag_msk[j,:,locx[j],locy[j]])
                    #print(output[j,:,locx[j],locy[j]]) 
                    #print(a)  
                    
                    loss = criterion(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])+K*(criterion1(output[j,:,0:-1,:],output[j,:,1:,:])+criterion1(output[j,:,:,0:-1],output[j,:,:,1:]))
                    #print(criterion1(output[j,:,0:-1,:],output[j,:,1:,:])+criterion1(output[j,:,:,0:-1],output[j,:,:,1:]))
                    #print(criterion(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]]))
                    #print(a)
                    loss_batch=loss+loss_batch
                    
                    output1[output1 <= 0.5] = 0  
                    loss_l1 = criterion1(output1[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])  
                    loss_batch_l1=loss_l1+loss_batch_l1
            #print(output[:,:,locx,locy].shape)
            #loss = criterion(output*bag_msk,bag_msk)
                
            loss=loss_batch/float(batch_size)
            loss_l1=loss_batch_l1/float(batch_size)
            loss.backward()     # 需要计算导数，则调用backward
            optimizer.step()
            #scheduler.step()
            #lr_scheduler.step()
            iter_loss = loss.item()    # .item()返回一个具体的值，一般用于loss和acc
            iter_loss_l1 = loss_l1.item()
            
            train_loss += iter_loss
            train_loss_l1 += iter_loss_l1
            
            output_np = output.cpu().detach().numpy().copy() 
            #print(output.shape)
            #print(a)
            #output_np = np.argmax(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() 
    
        all_train_loss.append(train_loss/len(train_dataloader))
        all_train_loss_l1.append(train_loss_l1/len(train_dataloader))
            # 每15个bacth，输出一次训练过程的数据
   

        
        # 验证
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk,name,locx,locy,locpx,locpy,loc4px,loc4py,imgC,imgD,shuttle_vx,shuttle_vy) in enumerate(test_dataloader):
                Mt=locy 
                #print(bag)
                #print(a)               
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                #print('o',output)
                #print(a)
                #output = nn.Softmax(dim=1)(output)
                
                #output = nn.Softmax(dim=1)(output)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                #output = torch.softmax(output,dim=1) # output.shape is torch.Size([4, 2, 160, 160])                
                loss_batch=0
                for j in range(len(bag_msk)): 
                    output[output <= 0.5] = 0        
                    loss = criterion1(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])
                    loss_batch=loss+loss_batch
            
                
                loss=loss_batch/float(batch_size)
                iter_loss = loss.item()
                
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() 
                #print(output_np.shape)

                
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() 
 
        all_test_loss.append(test_loss/len(test_dataloader))



        # 验证
        test_loss_control = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk,name,locx,locy,locpx,locpy,loc4px,loc4py,imgC,imgD,shuttle_vx,shuttle_vy) in enumerate(test_dataloader_control):
                Mt=locy 
                #print(bag)
                #print(a)               
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)

                optimizer.zero_grad()
                output = fcn_model(bag)
                #print('o',output)
                #print(a)
                #output = nn.Softmax(dim=1)(output)
                
                #output = nn.Softmax(dim=1)(output)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                #output = torch.softmax(output,dim=1) # output.shape is torch.Size([4, 2, 160, 160])                
                loss_batch_control=0
                for j in range(len(bag_msk)): 
                    output[output <= 0.5] = 0        
                    loss_control = criterion1(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])
                    loss_batch_control=loss_control+loss_batch_control
            
                
                loss_control=loss_batch_control/float(batch_size)
                iter_loss_control = loss_control.item()
                
                test_loss_control += iter_loss_control

                output_np = output.cpu().detach().numpy().copy() 
                #print(output_np.shape)

                
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() 
 
        all_test_loss_control.append(test_loss_control/len(test_dataloader_control))




















        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        
        #print('<---------------------------------------------------->')
        #print('epoch: %f'%epo)
        print('epoch = %d, train loss = %f,train loss (L1) = %f, test L1 loss (control) = %f, test L1 loss (non-control) = %f, %s'
                %(epo, train_loss/len(train_dataloader),train_loss_l1/len(train_dataloader), test_loss_control/len(test_dataloader_control),test_loss/len(test_dataloader), time_str))
        
 
        if np.mod(epo, 1) == 0:
            # 只存储模型参数
            path = './epo'+str(epo_num)+'_lr'+str(lr)+'_w'+str(w_decay)+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/'

  # Check whether the specified path exists or not
            isExist = os.path.exists(path)

            if not isExist:
  
  # Create a new directory because it does not exist 
                os.makedirs(path)
 
            torch.save(fcn_model.state_dict(), './epo'+str(epo_num)+'_lr'+str(lr)+'_w'+str(w_decay)+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/'+'model_{}.pth'.format(epo))
            #print('saving checkpoints/fcn_model_{}.pth'.format(epo))
 







    print('Train-index',all_train_loss.index(min(all_train_loss))+1)
    print('Train-index_l1',all_train_loss_l1.index(min(all_train_loss_l1))+1)
    print('Test-index',all_test_loss.index(min(all_test_loss))+1)


    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()


    
    ax.plot(eps,all_train_loss_l1,color='blue',label='Train')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d')) 

    ax.plot(eps,all_test_loss,color='red',label='Test_non-control')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d')) 
    
    ax.plot(eps,all_test_loss_control,color='orange',label='Test_control')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))     
    ax.legend()
    fig.savefig('epo'+str(epo_num)+'_lr'+str(lr)+'_w'+str(w_decay)+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'.png')









if __name__ == "__main__":

    train(epo_num, show_vgg_params=False)





