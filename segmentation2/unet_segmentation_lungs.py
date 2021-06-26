#!/usr/bin/env python
# coding: utf-8

# In[2]:


from models.UNet import *

import matplotlib.pyplot as plt
import numpy as np
import os
import time


# # Load Data

# In[1]:!whic


x_train = np.load('./dataset/x_train.npy')
y_train = np.load('./dataset/y_train.npy')
x_test = np.load('./dataset/x_test.npy')
y_test = np.load('./dataset/y_test.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# # Segmentation Class U-Net

# In[3]:


seg_model = UNet(img_shape = x_train[0].shape, num_of_class = 1,learning_rate = 2e-4, do_drop = True, drop_rate = 0.5)


# In[4]:


seg_model.show_model()


# # Train Model

# In[5]:


history = seg_model.train(x_train, y_train, epoch = 100, batch_size = 64)


# In[6]:


plot_dice(history)


# In[7]:


plot_loss(history)


# # Show result

# In[8]:


preds = seg_model.predict(x_test)


# In[9]:


show_num = 10
fig, ax = plt.subplots(show_num, 3, figsize=(15, 50))

for i, pred in enumerate(preds[:show_num]):
    ax[i, 0].imshow(x_test[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')

