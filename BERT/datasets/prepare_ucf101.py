#!/usr/bin/env python
# coding: utf-8

# ## 关键帧提取

# In[2]:


import os
import sys
import glob
import shutil
import codecs
from tqdm import tqdm_notebook as tqdm

import pandas as pd
import numpy as np
import time
from multiprocessing import Pool

#get_ipython().run_line_magic('pylab', 'inline')
from PIL import Image


# In[3]:


IN_PATH = './ucf101_frames'
OUT_PATH = './settings/ucf101'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# In[4]:


video_file_paths = glob.glob(IN_PATH + '/*')
print(video_file_paths)


# In[40]:


classid_fn = 'ucfTrainTestlist/classInd.txt'
f = open(classid_fn, "r")
lines = f.readlines()
f.close() 
print(lines)
classid = {} 
for l in lines: 
    l = l.strip('\n')
    l = l.split()
    id = int(l[0])
    act = l[1].lower()
    classid[act] = id 
print(classid)    


# In[18]:



def create_train_file(infile,outfile): 
    
    f = open(infile, "r")
    lines = f.readlines()
    f.close() 
    
    lines_setting = []
    for l in lines: 
        l = l.strip('\n')
        l = l.split()
        id = int(l[1])
        act = l[0]
        
        act = act.split('/')[-1]
        act = act.split('.')[0]
        
        act_dir = IN_PATH + '/' + act 
        
        numjpgs = len(glob.glob(act_dir + '/*.jpg')) 
        l_s = act + ' ' + str(numjpgs) + ' ' + str(id-1)
        
        lines_setting.append(l_s)
    
    f = open(outfile, 'w') 
    for l in lines_setting:
        f.write('{}\n'.format(l))
    f.close()    


# In[21]:


for i in range(3):
    infile = 'ucfTrainTestlist/trainlist{:02}.txt'.format(i+1)
    outfile = OUT_PATH + '/' + 'train_rgb_split{}.txt'.format(i+1)
    print(infile)
    print(outfile)
    create_train_file(infile,outfile)


# In[34]:


def create_test_file(infile,outfile,classid): 
    
    f = open(infile, "r")
    lines = f.readlines()
    f.close() 
    
    lines_setting = []
    for l in lines: 
        l = l.strip('\n')
        l = l.split()
        act = l[0]
        
        act = act.split('/')[-1]
        act = act.split('.')[0]
        
        cla = act.split('_')[1].lower() 
        
        if cla in classid:
            id = classid[cla]
        else:
            print("action not belonging to 101 classes : {}".format(cla))
            id = -1         
        act_dir = IN_PATH + '/' + act 
        
        numjpgs = len(glob.glob(act_dir + '/*.jpg')) 
        l_s = act + ' ' + str(numjpgs) + ' ' + str(id-1)
        
        lines_setting.append(l_s)
    
    f = open(outfile, 'w') 
    for l in lines_setting:
        f.write('{}\n'.format(l))
    f.close()    


# In[39]:


for i in range(3):
    infile = 'ucfTrainTestlist/testlist{:02}.txt'.format(i+1)
    outfile = OUT_PATH + '/' + 'val_rgb_split{}.txt'.format(i+1)
    print(infile)
    print(outfile)
    create_test_file(infile,outfile,classid)


# In[28]:


print(classid)


# In[ ]:




