#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path
sys.executable
import matplotlib.pyplot as plt
import numpy as np
import os
import time


# In[2]:


from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D,     MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from data_utils import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[3]:


class UNet(object):
    def __init__(self, img_shape, num_of_class, actf = 'relu',
        learning_rate = 0.001,  drop_rate = 0.5, do_batch_norm = False, do_drop = False):

        '''
        Arguments :

        img_shape - shape of input image (64, 64, 1)
        actf - activation function for network training
        learning_rate - learning rate for training
        drop_rate - dropout rate
        do_batch_norm - whether to run for batchnormalization
        do_drop - whether to run for dropout
        '''

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.num_of_class = num_of_class
        self.drop_rate = drop_rate
        self.do_batch_norm = do_batch_norm
        self.do_drop = do_drop

        self.model = self.build_model()

    # encoding block(conv - conv - pool)
    def enc_conv_block(self, inputs, feature_maps, filter_size = (3, 3),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        conv1 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool = MaxPooling2D(pooling_filter_size, strides = pooling_strides)(conv2)

        return pool, conv2

    # decoding block(concat - upconv - upconv)
    def dec_conv_block(self, inputs, merge_inputs, feature_maps, filter_size = (3, 3), conv_strides = 1,
                           up_conv_strides = (2, 2)):

        merge = Concatenate(axis = 3)([Conv2DTranspose(feature_maps, filter_size,
                                                       activation = self.actf, strides = up_conv_strides, kernel_initializer = 'he_normal',
                                                       padding = 'same')(inputs), merge_inputs])

        conv1 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(merge)
        conv2 = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                           padding = 'same', kernel_initializer = 'he_normal')(conv1)

        return conv2

    # encoder
    def encoding_path(self, inputs):

        enc_conv1, concat1 = self.enc_conv_block(inputs, 64)
        enc_conv2, concat2 = self.enc_conv_block(enc_conv1, 128)
        enc_conv3, concat3 = self.enc_conv_block(enc_conv2, 256)
        enc_conv4, concat4 = self.enc_conv_block(enc_conv3, 512)

        return concat1, concat2, concat3, concat4, enc_conv4

    # decoder
    def decoding_path(self, dec_inputs, concat1, concat2, concat3, concat4):

        dec_conv1 = self.dec_conv_block(dec_inputs, concat4, 512)
        dec_conv2 = self.dec_conv_block(dec_conv1, concat3, 256)
        dec_conv3 = self.dec_conv_block(dec_conv2, concat2, 128)
        dec_conv4 = self.dec_conv_block(dec_conv3, concat1, 64)

        return dec_conv4
    # build network
    def build_model(self):
        inputs = Input(self.img_shape)

        # Contracting path
        concat1, concat2, concat3, concat4, enc_path = self.encoding_path(inputs)

        # middle path
        mid_path1 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(enc_path)
        mid_path1 = Dropout(self.drop_rate)(mid_path1)
        mid_path2 = Conv2D(1024, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(mid_path1)
        mid_path2 = Dropout(self.drop_rate)(mid_path2)

        # Expanding path
        dec_path = self.decoding_path(mid_path2, concat1, concat2, concat3, concat4)
        segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(dec_path)


        model = Model(inputs = inputs, outputs = segmented)
        model.compile(optimizer = Adam(lr = self.learning_rate),
                          loss = 'binary_crossentropy', metrics = [dice_coef])

        return model

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, shuffle = True):

        
        earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        checkpoint = ModelCheckpoint('checkpoint/model-{epoch:03d}-{val_loss:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')  
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

        
        self.history = self.model.fit(X_train, Y_train, callbacks=[earlystopping,checkpoint,reduce_lr_loss ], validation_split = val_split,
                                          epochs = epoch, batch_size = batch_size, shuffle =  shuffle)
        return self.history

    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show u-net architecture
    def show_model(self):
        return print(self.model.summary())


# # Load Data

# # Segmentation Class U-Net

# In[4]:


seg_model = UNet(img_shape = (512,512,1), num_of_class = 1,learning_rate = 1e-5, do_drop = True, drop_rate = 0.5)


# In[5]:


seg_model.show_model()


# # Train Model

# In[6]:


seg_model.model = keras.models.load_model('checkpoint/model-last.h5', custom_objects={"dice_coef": dice_coef})


# In[7]:


import cv2 

ct_scan_dir = '/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/val/covid/ct_scan_0/'
ct_scan_files = [] 

imgs = [] 
for i in range(150,160):
    fn_full = ct_scan_dir+'{}.jpg'.format(i) 
    img = cv2.imread(fn_full,0) 
    img = np.expand_dims(img,axis=2)
    imgs.append(img)
x_test = np.asarray(imgs) 
print(x_test.shape) 


# In[8]:


preds1 = seg_model.predict(x_test[:5,:,:,:])
preds2 = seg_model.predict(x_test[5:10,:,:,:])
print(preds1.shape)
print(preds2.shape)
preds = np.concatenate((preds1, preds2), axis=0)


# In[9]:


show_num = 10
fig, ax = plt.subplots(show_num, 2, figsize=(15, 50))
for i, pred in enumerate(preds[:show_num]):
    ax[i, 0].imshow(x_test[i].squeeze(), cmap='gray')    
    ax[i, 1].imshow(pred.squeeze(), cmap='gray')


# In[11]:


covidx_dir = '/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/'
covidx_img_dir= covidx_dir 
covidx_mask_dir = covidx_dir + 'mask/'

if not os.path.exists(covidx_mask_dir):
    os.makedirs(covidx_img_dir_cropped)
print(covidx_dir)
print(covidx_img_dir)
print(covidx_mask_dir)


# In[36]:


#subsets = ['test'] # ,'val','train']
subsets = ['val','train']

from tqdm import tqdm

debug = False 

for subset in subsets:     
    
    cls = ['covid','non-covid'] 
        
    for c in cls: 
        cls_path = os.listdir(covidx_img_dir+"{}/{}/".format(subset,c))
        num_scans = len(cls_path) 
        
        
        pbar = tqdm(total=num_scans)
        
        
        
        for ind in range(num_scans):            
            
            pbar.update()
            
            
            s = "ct_scan_{}".format(ind)
            s_path = "{}/{}/{}".format(subset,c,s)
            #print(s_path)
            
            sfiles = os.listdir(covidx_img_dir+s_path) 
            sfiles = [x for x in sfiles if not x.startswith('.')]
            sfiles.sort(key = lambda x: int(x.split('.')[0]))
            
            
            fout_dir  = covidx_img_dir_cropped+"{}/{}/{}".format(subset,c,s)                   
            #print(fout_dir)
            
            if not os.path.exists(fout_dir):
                os.makedirs(fout_dir)

            count = 0     

            for f in sfiles:
                
                count += 1 
                
                fn = "{}/{}/{}/{}".format(subset,c,s,f)                
                f_full = covidx_img_dir + fn  
                fout_full = covidx_img_dir_cropped + fn                   
                if os.path.exists(fout_full):
                    continue 
                
                img = cv2.imread(f_full, cv2.IMREAD_UNCHANGED)
                if img.shape != (512,512):
                    img = cv2.resize(img,(512,512)) 
                                
                img = np.expand_dims(img,axis=2)
                img = np.expand_dims(img,axis=0)
                
                mask = seg_model.predict(img)
                
                
                mask = mask.squeeze()
                #print(mask.shape)
                
                mask = ((mask > 0.5)*255).astype(np.uint8) 
                
                fout_full = covidx_img_dir_cropped + fn    
                #print(fout_full)
                
                if debug: 
                    fig, (ax1,ax2) = plt.subplots(1,2)
                    ax1.imshow(img.squeeze(),cmap='gray')
                    ax2.imshow(mask.squeeze(),cmap='gray')

                if not os.path.exists(fout_full):
                    #print("writing image")
                    cv2.imwrite(fout_full,mask)
                #if count>5:
                #    break 
            #break 
                
        #break 
                


# In[ ]:




