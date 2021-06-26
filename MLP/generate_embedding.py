#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:52:03 2019

@author: esat
"""

import os, sys
import numpy as np

import time
import argparse

from ptflops import get_model_complexity_info

import torch,cv2
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import pickle,json

import argparse


datasetFolder="../BERT/datasets"

sys.path.insert(0, "../BERT/")
sys.path.insert(0,'.')
sys.path.insert(0,'../BERT/datasets/')


import models
from VideoSpatialPrediction3D_bert_embedding import VideoSpatialPrediction3D_bert
import video_transforms

from arcface import ArcFace 

from utils import RandomResampler, SymmetricalResampler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--settings', metavar='DIR', default='../BERT/datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='covid',
                    choices=["ucf101", "hmdb51", "smtV2", "window","videoreloc","covid"],
                    help='dataset: ucf101 | hmdb51 | smtV2')

parser.add_argument('--dataset_root', type=str, default='/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/',
                    help='dataset root directory')

parser.add_argument('--desc', type=str, default='rrr', help='descripton of the channels of image')


parser.add_argument('--subset', '-t', default='val',
                    choices=["train", "val","test"],
                    help='subset: train | val | test')


parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34_bert10',
                    choices=model_names)

parser.add_argument('-s', '--split', default=5, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')

parser.add_argument('-refer_nums', default=10, type=int, metavar='Ref_N',
                    help='The number of refers in each category')

parser.add_argument('-w', '--window', default=3, type=int, metavar='V',
                    help='validation file index (default: 3)')


parser.add_argument('-v', '--val', dest='window_val', action='store_true',
                    help='Window Validation Selection')

parser.add_argument('--gpu_id',type = int,default=3, help='foo help')

parser.add_argument('--modelfile', type=str, default='model_best.pth.tar',help='model filename')


multiGPUTest = False
multiGPUTrain = True
ten_crop_enabled = False
num_seg = 32
num_seg_3D=1


debug = False 

result_dict = {}

def buildModel(model_path,num_categories):
    model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg_3D)
    params = torch.load(model_path)

    if multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    elif multiGPUTest:
        model=torch.nn.DataParallel(model)
        new_dict={"module."+k: v for k, v in params['state_dict'].items()} 
        model.load_state_dict(new_dict)
        
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
 
    '''
    HEAD = ArcFace(in_features = 512, out_features = 180, device_id = gpu_id)
    HEAD.load_state_dict(params['state_dict_head'] )
    HEAD.eval() 
    ''' 
    return model # ,HEAD

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0) 

    #print("batch_size = ",batch_size) 

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    target2 = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target2)

    #print("pred = {}, target = {}, correct = {}".format(pred,target,correct)) 

    correct = correct.cpu().numpy() 
    #print("correct  = {}".format(correct)) 

    correct = np.sum(correct) 
    #print("correct  = {}".format(correct)) 
     
    return correct  


def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)

    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8    
    else:
        length=16
     
    step = int(length/2) 

    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)  #  '_2021_01_07_18_27_43'

    model_path = os.path.join('../BERT/',modelLocation,args.modelfile)

    print("model_path = {}".format(model_path)) 

    if not os.path.exists(model_path):
        model_path = os.path.join(modelLocation,'model_best.pth.tar') 
        print("model_path = {}".format(model_path)) 


    dataset = args.dataset_root 
    print("dataset root directory = {}".format(dataset))  

    val_setting_file = "%s_rgb_split%d.txt" % (args.subset, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)

    print("val_split_file = {}".format(val_split_file)) 

    if not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))



    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='smtV2':
        num_categories = 174
    elif args.dataset=='window':
        num_categories = 3
    elif args.dataset=='videoreloc':
        num_categories = 160
    elif args.dataset=='covid':
        num_categories = 2

    model_start_time = time.time()
    spatial_net = buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))
    
    # flops, params = get_model_complexity_info(spatial_net, (3,length, 112, 112), as_strings=True, print_per_layer_stat=False)
    # #flops, params = get_model_complexity_info(spatial_net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    f_val = open(val_split_file , "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))
    
    scale = 1.0
    input_size = int(224 * scale)
    width = height = int(input_size*1.25)  # 1.25    
    print("scale= {}, input_size = {}, width = {}, height = {}".format(scale,input_size,width,height)) 
  
    clip_mean = [0.43216, 0.394666, 0.37645] 
    clip_std = [0.22803, 0.22145, 0.216989] 

    normalize = video_transforms.Normalize(mean=clip_mean,
                                std=clip_std)

    val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])

    dims = (height, width,3)    


    id_of_class = []
    val_data_len = len(val_list)

    for line in val_list:
        id_of_class.append(line.split()[1])

    num_of_class = len(set(id_of_class))

    print("id_of_class = {}".format(set(id_of_class)) ) 
    print("num_of_class = {}".format(num_of_class)) 

    #result_list = []

    output_path = './features_{}_{}'.format(args.subset,args.desc)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    classes = {"covid":0,"non-covid":1} 
    
    if True:
        
        Predictions = dict() 

        for line in tqdm(val_list):
            # line_info = line.split(" ")
            # clip_path = os.path.join(data_dir,line['id'])
            # duration = int(line_info[1])
            # input_video_label = int(line_info[2]) 

            target = classes[line.split()[1]]              
            print("line = {}".format(line)) 

            fn_splits = line.split()[0].split('/')  
            fn = "{}_{}_{}".format(fn_splits[0], fn_splits[1],fn_splits[2]) 

            feature_file = output_path+'/features_{}.pkl'.format(fn)
            print(feature_file) 

            #if os.path.exists(feature_file):
            #    continue 

            img_dir = line.split()[0] 
            mask_dir = 'mask/' + img_dir 

            img_path = os.path.join(dataset, img_dir)
            mask_path = os.path.join(dataset, mask_dir)
 
            first_slice = int(line.split()[2]) 
            last_slice = int(line.split()[3]) 

            print(img_dir,mask_dir,img_path,first_slice,last_slice) 

            slices_all = [] 
            for s in range(first_slice,last_slice+1):
                slices_all.append("{}.jpg".format(s))     
   
            #print("slices_all = {}".format(slices_all))  
           
            slices_num = len(slices_all) 
            interval = int(slices_num // length)
            #print(len(slices_all),interval) 

            slices_len = 0  
            slices_total = []        
            slices_to_process = dict()  
            i = 0 
            for i in range(interval): 
                #print("i = {}, interval = {}".format(i,interval))                
                slices = SymmetricalResampler.resample(slices_all, length, i-interval+1)
                slices_len += len(slices)
                slices_total.extend(slices)
                #print("i = {}, len= {}, slices = {}".format(i,slices_len,slices))  
                slices_to_process[i] = slices 
                print(i)
                print(slices_to_process[i])
                        
            if slices_num - len(slices_total) > 0:  
                i = interval
                slices_remain = set(slices_all) - set(slices_total)
                slices_remain = list(slices_remain) 
                slices_remain.sort(key=lambda x: int(x.split('.')[0])) 
                slices_to_process[i] = SymmetricalResampler.resample(slices_remain, length) 
                print(i)
                print(slices_to_process[i])     

            #06/26 added sequential slices 
            if slices_num > length: 
                i +=1

                if slices_num == length * interval:  
                    avg_len = int(slices_num/interval)
                else: 
                    avg_len = int(slices_num // (interval+1))
                print(slices_len, interval,avg_len) 
                slices_total = []   
                slices_selected = [] 
                for j in range(interval):
                    slices_selected = slices_all[j*avg_len:(j+1)*avg_len]  
                    slices_selected = SymmetricalResampler.resample(slices_selected, length)  
                    slices_to_process[i+j] = slices_selected
                    slices_total.extend(slices_selected)
                    print(i+j)
                    print(slices_to_process[i+j]) 

                if slices_num - len(slices_total) > 0: 
                    j +=1 
                    slices_remain = set(slices_all) - set(slices_total)
                    slices_remain = list(slices_remain) 
                    slices_remain.sort(key=lambda x: int(x.split('.')[0])) 
                    slices_to_process[i+j] = SymmetricalResampler.resample(slices_remain, length)   
                    print(i+j) 
                    print(slices_to_process[i+j]) 

            #print(slices_to_process)  
            #input("dbg")    

            features = [] 
            preds = [] 

            for i in slices_to_process: 
            
                slices = slices_to_process[i] 
                #print(slices) 

                imageList = [] 

                
                for s in slices: 
                    img_file  = os.path.join(img_path,s) 
                    mask_file = os.path.join(mask_path,s+'_refined.jpg') 

                    #print(img_file,mask_file) 

                    img  = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE) 
                    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE) 


                    img_mask = img.copy() 
                    black_ind = mask ==0 
                    img_mask[black_ind] = 0 
                     
                    img_merged = cv2.merge([img,mask,img_mask])
                    img_merged = cv2.resize(img_merged, (height,width), cv2.INTER_LINEAR)

                    #img_mask = cv2.resize(img_mask, (height,width), cv2.INTER_LINEAR)
                    #img_merged = cv2.cvtColor(img_mask,cv2.COLOR_GRAY2RGB) # cv2                     

                    '''  #will check later 
                    th, tw = input_size
                    x1 = int(round((width - tw) / 2.))
                    y1 = int(round((height - th) / 2.))
                    img_merged = img_merged[y1:y1+th, x1:x1+tw, :]
                    '''
 
                    imageList.append(img_merged) #CenterCrop 


                rgb_list=[] 
                for ind in range(len(imageList)):   
                    cur_img = imageList[ind].copy() 
                    cur_img_tensor = val_transform(cur_img)

                    rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))

                       
                input_data=np.concatenate(rgb_list,axis=0) 
                #print("input_data.shape = {}".format(input_data)) 
  
                with torch.no_grad():
                    imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
                    #print("imgDataTensor.shape = {}".format(imgDataTensor)) 

                    imgDataTensor = imgDataTensor.view(-1,length,3,input_size,input_size).transpose(1,2)

                    output, input_vectors, sequenceOut, maskSample, embedding = spatial_net(imgDataTensor)

                    _, pred = output.topk(1, 1, True, True)
                    pred = pred.cpu().numpy()[0][0]
                    preds.append(pred) 

                    #features.append(F.normalize(embedding, p=2, dim=1).data)
                    embedding = embedding.cpu().numpy().squeeze(axis=0)  
                    features.append(embedding)                   

                if debug: 
                    print("segment_index = {}, segment_s = {}, clip_len = {}".format(segment_index, segment_s, clip_len)) 
                    print("features len = {}".format(len(features)) )
                    input("debug feature") 
 
            features_final = np.asarray(features) # .cpu().numpy            
            #print(features_final.shape) 
            #input("dbg") 
            Predictions[fn] = preds 
                
            with open(feature_file, 'wb') as pk_file:
                print("saving feature {}".format(feature_file)) 
                pickle.dump(features_final, pk_file)
    
        fp = open("{}_predictions.txt".format(args.subset),'w') 
        for fn in Predictions:
            preds = Predictions[fn] 
            if "no-covid" in fn:
                true = 1
            else:
                true = 0 

            fp.write("{} {} ".format(fn,true))
            for p in preds:
                fp.write("{} ".format(p))
            fp.write("\n")
        fp.close()  

if __name__ == "__main__":
    main()

