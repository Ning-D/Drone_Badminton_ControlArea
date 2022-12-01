import argparse
import queue
import cv2
import numpy as np
import os
import natsort
from PIL import Image, ImageDraw
import pandas as pd
Points_dir='./Points/' 

count=1
pdir=natsort.natsorted(os.listdir(Points_dir))
print(pdir)
previous_name='BD.csv'
for i in range(len(pdir)):

  # cdir=natsort.natsorted(os.listdir(os.path.join(Points_dir,pdir[i])))
   #
   dirname=pdir[i]
   print(dirname)
   df = pd.read_csv(str(Points_dir)+'/'+str(dirname),sep=',',na_values="None")
   
   df=df.fillna(-1)

   dname=dirname.replace('.csv','')
  # df.replace("Non-Point", 0)
  # df.replace('Point', 1)  
  
   for j in range(len(df)):
       print('DDDDDDDDDD',dname.split('_')[0])
       print(previous_name)
       if dname.split('_')[0]!=previous_name:
           count=1
       print(count)
       new=df[['Point']].iloc[j]
       print(new)
       #new=new.reset_index(drop=True)
       
       
      
       PATH = './Estimation_region/Points/'+str(dname.split('_')[0])+'/'
       if not os.path.exists(PATH):
           os.makedirs(PATH)
       
       
       new.to_csv('./Estimation_region/Points/'+str(dname.split('_')[0])+'/'+str(count)+'.txt',header=False,index=False) 
       #np.savetxt('./Estimation_region/Points/'+str(dname.split('_')[0])+'/'+str(count)+'.csv',new, delimiter=" ", fmt="%s")         
       count=count+1
       previous_name=dname.split('_')[0]
       #print()
