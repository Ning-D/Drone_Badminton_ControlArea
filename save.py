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
from BagDataset.All import BagDataset
cudnn.benchmark = True
plt.ion()   # interactive mode
from tqdm import tqdm
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
import torch.optim as optim
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
from PIL import Image
import visdom
from fcn import VGGNet, FCN8s, FCNs,UNet,NestedUNet
import matplotlib.patches as patches
np.set_printoptions(threshold=np.inf)
import seaborn as sns
import sys
from configuration import lr,momentum,w_decay,step_size,gamma,epo_num , batch_size, ratio    ,ALPHA,BETA,GAMMA ,K
from scipy.ndimage import gaussian_filter
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

batch_size=1

import argparse
parser = argparse.ArgumentParser()
np.set_printoptions(threshold=np.inf)
parser.add_argument('--checkpoint_path', type=str, default='1.pth',
                    help='checkpoint path')

args = parser.parse_args()
checkpoint_path = args.checkpoint_path
if not os.path.isfile(checkpoint_path) or not checkpoint_path.endswith('.pth'):
    print("Not a valid checkpoint path! Please modify path in parser.py --checkpoint_path")
    sys.exit(1)



def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


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
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated



W = 112  # width of heatmap
H = 56  # height of heatmap
SCALE = 8 # increase scale to make larger gaussians

height = 2160
width = 3840








def test(show_vgg_params=False):

    transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])

    bag = BagDataset(transform)
    test_size = len(bag) 
    test_dataset=bag
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fcn_model = UNet(num_classes=1, input_channels=5)
    
    fcn_model = fcn_model.to(device)
    fcn_model.load_state_dict(torch.load(str(checkpoint_path)))
    
    
    
    criterion = nn.L1Loss()  
    optimizer = optim.Adam(fcn_model.parameters(), lr=lr,weight_decay=w_decay)
    all_train_loss = []
    all_test_loss = []
    test_Acc = []
    test_mIou = []
    # start timing
    prev_time = datetime.now()
    eps=[]
    All_control_area_T1=[]
    All_control_area_T2=[]
    control_area_T1=0
    control_area_T2=0
    previous_game_name='MD12' 
    previous_rally_name='0'
    ['MD12','MD13','MD14','MD15','MD16','MD17']
    test_loss = 0
    fcn_model.eval()
    with torch.no_grad():
        for index, (bag, bag_msk,name,locx,locy,locpx,locpy,loc4px,loc4py,imgC,imgD,shuttle_vx,shuttle_vy) in enumerate(test_dataloader):
                
            b=bag
            px=locpx
            bb=bag_msk
            
            py=locpy
            
            locsx=locx
            locsy=locy
            M=locy
           
            
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output) 
                     
            loss_batch=0
            for j in range(len(bag_msk)): 
                  
                    output[output < 0.5] = 0           
                    loss = criterion(output[j,:,locx[j],locy[j]],bag_msk[j,:,locx[j],locy[j]])
                    loss_batch=loss+loss_batch
        
                
            loss=loss_batch/float(batch_size)
            iter_loss = loss.item()
                
            test_loss += iter_loss
            output_np = output.cpu().detach().numpy().copy() 
           
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() 
            px = px.cpu().detach().numpy().copy() 
            py = py.cpu().detach().numpy().copy() 
            locsx = locsx.cpu().detach().numpy().copy() 
            locsy = locsy.cpu().detach().numpy().copy() 
            loc4px=loc4px.cpu().detach().numpy().copy() 
            loc4py=loc4py.cpu().detach().numpy().copy() 
            M=M.cpu().detach().numpy().copy() 
            shuttle_vx=shuttle_vx.cpu().detach().numpy().copy().reshape(56,112)
            shuttle_vy=shuttle_vy.cpu().detach().numpy().copy().reshape(56,112) 
       
            
            keyword = 'Flip'
            
            for j in range(len(bag)):
               

               im=output_np[j][0].reshape(56,112)
               name=np.array(name)
               
               
               n=name[j].replace('.csv','.png')

               
               if keyword in n:
                   
                   break

               fig, axes = plt.subplots(nrows=1, ncols=2,
                               sharex=True, sharey=False,
                               figsize=(19, 8))
               ax1, ax2 = axes                
            
               im1=im
               if M[j]>int(W/2):
                   im[:,:int(W/2)]=0

               else:
                   im[:,int(W/2):]=0
           
               p1,p2=np.where(im>=0.5)
               game_name=os.path.splitext(n)[0].split('_')[0]
               rally_name=os.path.splitext(n)[0].split('_')[1]
               hit_name=os.path.splitext(n)[0].split('_')[2]
               if game_name==previous_game_name and rally_name==previous_rally_name:
                 
                   if (loc4px[0][0]-width/2)* (M[j]-int(W/2))>0:
                   
                       control_area_T1=control_area_T1+len(p1)
                       print('T1-Name = %s, %d'
                %(n, len(p1))) 
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', im, delimiter=",")
                       
                       
                       
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (locsx[j], locsy[j]), delimiter=",")
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (shuttle_vx[locsx[j], locsy[j]],shuttle_vy[locsx[j], locsy[j]]), delimiter=",")
                       
                   else:
                       control_area_T2=control_area_T2+len(p1)
                       
                       
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                            
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', im, delimiter=",")
                       
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                            
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (locsx[j], locsy[j]), delimiter=",")
                       print('T2-Name = %s, %d'
                %(n, len(p1))) 
                
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (shuttle_vx[locsx[j], locsy[j]],shuttle_vy[locsx[j], locsy[j]]), delimiter=",")
                       
                       
                       
               else:
                  
                   print('control_area_T1',control_area_T1)
                   print('control_area_T2',control_area_T2)
                
                   All_control_area_T1.append(control_area_T1)
                   All_control_area_T2.append(control_area_T2)
                   control_area_T1=0
                   control_area_T2=0
                   if (loc4px[0][0]-width/2)* (M[j]-int(W/2))>0:
                   
                       control_area_T1=len(p1)
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', im, delimiter=",")
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (locsx[j], locsy[j]), delimiter=",")
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T1_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (shuttle_vx[locsx[j], locsy[j]],shuttle_vy[locsx[j], locsy[j]]), delimiter=",")
                       
                       
                       print('T1-Name = %s, %d'
                %(n, len(p1))) 
                   else:
                       
                       
                       
                       control_area_T2=len(p1)
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', im, delimiter=",")
                       
                       
                       
                       
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                       
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (locsx[j], locsy[j]), delimiter=",")
                       print('T2-Name = %s, %d'
                %(n, len(p1)))                   
                        
                       PATH ='./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'
                       if not os.path.exists(PATH):
                           os.makedirs(PATH)                             
                       np.savetxt('./outputcsv/'+'_A'+str(ALPHA)+'_B'+str(BETA)+'_G'+str(GAMMA)+'_K'+str(K)+'/Estimation_T2_shuttle_v/'+str(game_name)+'/'+str(rally_name)+'/'+str(hit_name)+'.csv', (shuttle_vx[locsx[j], locsy[j]],shuttle_vy[locsx[j], locsy[j]]), delimiter=",")
                   
                   
               previous_game_name=game_name
               previous_rally_name=rally_name
               sns.heatmap(im, cmap='Blues',square=True,annot=False,ax=ax1,cbar_kws={"shrink": 0.641},vmax=1, vmin=0, center=0.5)
           
               imgC=imgC.reshape(-1,H,W)
               imgD=imgD.reshape(-1,H,W)
               
               for c in range(2):
                  ax1.scatter(py[j][c], px[j][c], s=100,facecolors='none', linewidths=2, edgecolors="orange")
                  ax1.quiver(py[j][c], px[j][c],imgC[j,px[j][c],py[j][c]],imgD[j,px[j][c], py[j][c]],angles='xy', scale_units='xy', scale=1)
                  if py[j][c]==0 and px[j][c]==0:
                      print(str(n))
                      print(a)
       
               ax1.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
               ax1.annotate((locsy[j][0],locsx[j][0]), xy = (locsy[j], locsx[j]), size = 15, color = "red")
               ax1.vlines(x=170*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=3670*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=1920*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1,linestyle = "dashed")
               ax1.vlines(x=370*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=3470*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=1395*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax1.vlines(x=2445*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
      
               ax1.hlines(y=280*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1880*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=405*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1755*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1080*H/height, xmin=170*W/width,xmax=1395*W/width,colors='g',linewidth=1)
               ax1.hlines(y=1080*H/height, xmin=2445*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               

               
               smoothed_matrix = gaussian_filter(im1, sigma=2,mode='nearest')
               if M[j]>int(W/2):
                   smoothed_matrix[:,:int(W/2)]=0
               else:
                   smoothed_matrix[:,int(W/2):]=0                       
               
               
               
              
               sns.heatmap(smoothed_matrix, cmap='Blues',square=True,annot=False,ax=ax2,cbar_kws={"shrink": 0.641},vmax=1, vmin=0, center=0.5)
               
               for c in range(2):
               
                  ax2.scatter(py[j][c], px[j][c], s=100,facecolors='none', linewidths=2, edgecolors="orange")
                  ax2.quiver(py[j][c], px[j][c],imgC[j,px[j][c],py[j][c]],imgD[j,px[j][c], py[j][c]],angles='xy', scale_units='xy', scale=1)

               ax2.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
                   
                   
               ax2.scatter(locsy[j], locsx[j], s=50,facecolors='none', linewidths=2, edgecolors="red")
               ax2.vlines(x=170*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=3670*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=1920*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1,linestyle = "dashed")
               ax2.vlines(x=370*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=3470*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=1395*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
               ax2.vlines(x=2445*W/width, ymin=280*H/height,ymax=1880*H/height,colors='g',linewidth=1)
      
               ax2.hlines(y=280*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1880*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=405*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1755*H/height, xmin=170*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1080*H/height, xmin=170*W/width,xmax=1395*W/width,colors='g',linewidth=1)
               ax2.hlines(y=1080*H/height, xmin=2445*W/width,xmax=3670*W/width,colors='g',linewidth=1)
               
              
               
               
               
               
               
               
               
               
               
               c_path=checkpoint_path.replace('.','')
               c_path=c_path.replace('pth','')              
               
               
               
               
               path = './output/'+c_path
               isExist = os.path.exists(path)
               if not isExist:
                   os.makedirs(path)
               plt.ioff()
               
               
               fig.tight_layout()
               plt.savefig(str(path)+'/'+str(n))  
               plt.close(fig)
              
   
        all_test_loss.append(test_loss/len(test_dataloader))






           


















if __name__ == "__main__":

    test(show_vgg_params=False)





