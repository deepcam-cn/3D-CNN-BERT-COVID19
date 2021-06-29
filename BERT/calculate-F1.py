#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tqdm 


# In[2]:


import skimage, os
from skimage.morphology import ball, disk, dilation,binary_dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing, binary_opening 
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from glob import glob
from skimage.io import imread


# In[3]:


from sklearn.metrics import f1_score, precision_recall_curve


# In[4]:


result_file = 'validate_results_split5.txt'
fp = open(result_file)
lines = fp.readlines() 
lines = [x.strip() for x in lines]
fp.close()
#print(lines[:10])


# In[6]:


y_true = []
y_pred = [] 
y_prob = [] 

for l in lines: 
    l_splits = l.split(',') 
    #print(l_splits)
    
    #change covid class label to 1 
    y_true.append(1-int(l_splits[1]))
    y_pred.append(1-int(l_splits[2])) 
    
    
    prob0 = float(l_splits[3])
    prob1 = float(l_splits[4])

    score = np.exp(prob0)/(np.exp(prob0)+np.exp(prob1))

    y_prob.append(score)


# In[7]:


y_pred = np.asarray(y_pred)
y_true = np.asarray(y_true)
y_prob = np.asarray(y_prob)


# In[8]:


F1_1 = f1_score(y_true, y_pred, average='micro')
F1_2 = f1_score(y_true, y_pred, average='macro')
print(F1_1,F1_2)

exit() 

# In[9]:


TP = 0
FP = 0 
FN = 0 
TN = 0 
for t,p in zip(y_true,y_pred):    
    #print(t,p)    
    if t==1 and p==1:
        TP +=1 
    elif t==1 and p==0:
        FN +=1 
    elif t==0 and p ==1: 
        FP +=1 
    elif t==0 and p==0:
        TN +=1 
precision = TP/(TP+FP)        
recall    = TP/(TP+FN) 

F1 = 2*precision*recall/(precision+recall)        
print(TP,FP,FN,TN)        
print(precision,recall,F1)    


# In[10]:


precision, recall, thresholds = precision_recall_curve(y_true, y_prob)


# In[10]:


print(len(precision),len(recall),len(thresholds))


# In[11]:


f1s = [] 
for i in range(len(recall)):
    f1 = 2*precision[i]*recall[i]/(precision[i]+recall[i])
    f1s.append(f1)


# In[12]:


max(f1s)


# In[13]:


result_file = 'scripts/eval3/val_predictions.txt'
fp = open(result_file)
lines = fp.readlines() 
lines = [x.strip() for x in lines]
fp.close()
print(lines[:10])


# In[40]:


y_true = []
y_pred = [] 

for l in lines: 
    l_splits = l.split() 
    #print(l_splits)
    
    scan_id = l_splits[0]
    
    #print(scan_id)
    
    if 'non-covid' in scan_id:
        true = 1
    else:
        true = 0 
        
    #print(true)    
        
    preds = [] 
    for i in range(2, len(l_splits)):
        pred = int(l_splits[i])
        preds.append(pred)
    
    preds = np.asarray(preds)
          
    #print(preds)    
    #print(preds==0)
    #print(np.sum(preds==0))
    
    if np.sum(preds==0) >= len(preds)/2: 
        pred = 0
    else:
        pred = 1 
        
       
    #change covid class label to 1 
    y_pred.append(1-pred) 
    y_true.append(1-true)
        


# In[41]:


F1_1 = f1_score(y_true, y_pred, average='micro')
F1_2 = f1_score(y_true, y_pred, average='macro')
print(F1_1,F1_2)


# In[ ]:




