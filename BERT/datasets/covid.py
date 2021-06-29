import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2
from PIL import Image
import random

from .utils import RandomResampler, SymmetricalResampler, SymmetricalSequentialResampler

cv2_loader = True 
cv2.setNumThreads(0) #fix the opencv resize stuck issue 


def pil_loader(root,path,size):

    img_path = os.path.join(root,path) 
    img = Image.open(img_path)

    if img.verify():
        print("Could not load file %s" % (frame_path))
        input("debugging not enough frame images") 
        sys.exit()


    bands = img.getbands()
    if len(bands) != 3:  # changed >3 to !=3 
        #print(len(bands)) 
        img = img.convert('RGB')

    img = img.resize(size, Image.ANTIALIAS) 

    return img


def cv2_loader(root,path,size,is_color):

    if is_color:
        flag = cv2.IMREAD_COLOR         # > 0
    else:
        flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    flag = cv2.IMREAD_GRAYSCALE 

    img_path = os.path.join(root,path) 
    img = cv2.imread(img_path, flag)
    if img is None:
       print("Could not load file %s" % (frame_path))
       input("debugging not enough frame images") 
       sys.exit()

    min_max_norm = False 

    if min_max_norm: 
        img = img.astype('int16').astype('float') 
        min = np.min(img) 
        max = np.max(img) 
        img = np.uint8((img-min)/(max-min)*255) 

    img = cv2.resize(img, size, interpolation)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) # cv2.COLOR_BGR2RGB)
    return img 


#include a global bbox for every scan, and mask for every slice 
def cv2_loader2(root,path,size,is_color,bbox):

    if is_color:
        flag = cv2.IMREAD_COLOR         # > 0
    else:
        flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR



    #already read grayscale 
    flag = cv2.IMREAD_GRAYSCALE 
    img_path = os.path.join(root,path) 
    #print(img_path) 

    img = cv2.imread(img_path, flag)
    #print(img.shape) 

    if img is None:
       print("Could not load file %s" % (img_path))
       input("debugging not enough frame images") 
       sys.exit()
    '''
    if img.shape != (512,512): #this happens 
        img = cv2.resize(img,(512,512), interpolation) 
    ''' 
    
    mask_path = os.path.join(root,'mask', path+'_refined.jpg') 
    #print(mask_path) 
    mask = cv2.imread(mask_path, flag)
    if mask is None:
       print("Could not load file %s" % (mask_path))
       input("debugging not enough mask images") 
       sys.exit()
 
    #print(img.shape,mask.shape) 

    xmin,ymin,xmax,ymax = bbox  

    #print(xmin,ymin,xmax,ymax)  
    
    #print(np.max(mask),np.min(mask))   
    
    img_mask = img.copy() 
    black_ind = mask ==0 
    img_mask[black_ind] = 0 

    '''
    img_crop = img[ymin:ymax,xmin:xmax] 
    img_mask_crop = img_mask[ymin:ymax,xmin:xmax] 

    #print(img_crop.shape,img_mask_crop.shape) 


    #img_merged = cv2.merge([img,img_crop,img_mask_crop])
    onlyonce = False 
    if onlyonce: 
        cv2.imwrite("debug/img.jpg",img) 
        cv2.imwrite("debug/img_crop.jpg",img_crop) 
        cv2.imwrite("debug/img_mask.jpg",img_mask) 
        cv2.imwrite("debug/img_mask_crop.jpg",img_mask_crop) 
        onlyonce = False 

    img_crop = cv2.resize(img_crop, size, interpolation)
    img_mask_crop = cv2.resize(img_mask_crop, size, interpolation)
    img = cv2.resize(img, size, interpolation)
         

    img = cv2.resize(img, size, interpolation)
    #mask = cv2.resize(mask, size, interpolation)
    img_mask = cv2.resize(img_mask, size, interpolation)
    
    img_mask = np.expand_dims(img_mask, axis=2)
    img      = np.expand_dims(img, axis=2)

    img_merged = np.concatenate((img,img_mask), axis=2)
    print(img_merged.shape) 
    ''' 

    img_merged = cv2.merge([img,mask,img_mask])

    #06/23 evening 11:17 added 
    #img_merged = img_merged[ymin:ymax,xmin:xmax,:] 

    img_merged = cv2.resize(img_merged, size, interpolation)

    #img_merged = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) # cv2.COLOR_BGR2RGB)

    return img_merged 

def find_classes(dir):

    print("dir = {}".format(dir))  

    #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes = ["covid","non-covid"] 
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    print("classes = {}".format(classes)) 
    print("class_to_idx = {}".format(class_to_idx)) 

    return classes, class_to_idx

def make_dataset(root, source):

    if not os.path.exists(source):
        print("Setting file %s for UCF101 dataset doesn't exist." % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                #clip_path = os.path.join(root, line_info[0])
                clip_path =  line_info[0]

                #print("clip_path = {}".format(clip_path)) 
                target = line_info[1]
                start = int(line_info[2])
                stop  = int(line_info[3])
                duration = stop-start+1 

                try: 
                    xmin = int(line_info[4])
                    ymin = int(line_info[5])
                    xmax = int(line_info[6])
                    ymax = int(line_info[7])
                    bbox = (xmin,ymin,xmax,ymax) 
                except:
                    bbox =(0,0,-1,-1) 

                item = (clip_path, duration, start, stop, target,bbox)
                clips.append(item)
    return clips


def ReadSegmentRGBResample(root, path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration,start,stop,is_train):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    debug = False 

    sampled_list = []

    #print("path = {}, duration = {}, start = {}, stop = {}".format(path, duration,start,stop)) 

   
    if debug: 
        print("offsets = {}".format(offsets)) 
        print("(new_height = {}, new_width = {}, new_length = {})".format(new_height, new_width, new_length)) 
        print("duration = {}, start = {}, stop = {}".format(duration,start,stop)) 


    slices_all = [] 
    for ind in range(start,stop+1):
        frame_name = name_pattern % (ind)
        slices_all.append(frame_name) 

    if debug:
        print("slices_all = {}".format(slices_all)) 

    if is_train:
        slices = RandomResampler.resample(slices_all, new_length)
    else:
        slices = SymmetricalResampler.resample(slices_all, new_length)

    if debug: 
        print("slices = {}".format(slices)) 

    cnt = 0     
    for s in slices: 
        frame_path = path + "/" + s 

        if debug: 
            print(cnt, s,frame_path)  

        if cv2_loader: 
            cv_img = cv2_loader(root, frame_path,  (new_width, new_height), is_color)            
        else: 
            cv_img = pil_loader(root, frame_path,  (new_width, new_height))

        if debug: 
            print("cv_img.type = {}, cv_img.shape = {}".format(type(cv_img), cv_img.shape)) 

        sampled_list.append(cv_img)


    #print("sampled_list.len = {}, shape = {}".format(len(sampled_list),sampled_list[0].shape)) 
    clip_input = np.concatenate(sampled_list, axis=2)
    if debug: 
        print("clip_input.shape = {}".format(clip_input.shape))        

    #clip_input = Image.fromarray(clip_input)  

    return clip_input

def ReadSegmentRGBMaskResample(root, path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration,start,stop,is_train,bbox):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    debug = False 

    sampled_list = []

    #print("path = {}, duration = {}, start = {}, stop = {}".format(path, duration,start,stop)) 

   
    if debug: 
        print("offsets = {}".format(offsets)) 
        print("(new_height = {}, new_width = {}, new_length = {})".format(new_height, new_width, new_length)) 
        print("duration = {}, start = {}, stop = {}".format(duration,start,stop)) 


    slices_all = [] 
    for ind in range(start,stop+1):
        frame_name = name_pattern % (ind)
        slices_all.append(frame_name) 

    if debug:
        print("slices_all = {}".format(slices_all))
     
    if is_train:
        slices = RandomResampler.resample(slices_all, new_length)
    else:
        slices = SymmetricalResampler.resample(slices_all, new_length)

    '''
    slices_selects, num_sets = SymmetricalSequentialResampler(slices_all, new_length)
    #print("slices_selects = {}".format(slices_selects))
    select = random.randint(0, len(slices_selects)-1)
    slices = slices_selects[select]            
    #print("seleced slices = {}".format(slices)) 
    ''' 

    if debug: 
        print("is_train = {}, slices = {}".format(is_train,slices)) 

    cnt = 0     
    for s in slices: 
        frame_path = path + "/" + s 

        if debug: 
            print(cnt, s,frame_path)  

        if cv2_loader: 
            cv_img = cv2_loader2(root, frame_path,  (new_width, new_height), is_color,bbox)
        else: 
            cv_img = pil_loader(root, frame_path,  (new_width, new_height))

        if debug: 
            print("cv_img.type = {}, cv_img.shape = {}".format(type(cv_img), cv_img.shape)) 

        sampled_list.append(cv_img)


    #print("sampled_list.len = {}, shape = {}".format(len(sampled_list),sampled_list[0].shape)) 
    clip_input = np.concatenate(sampled_list, axis=2)
    if debug: 
        print("clip_input.shape = {}".format(clip_input.shape)) 

    #clip_input = Image.fromarray(clip_input)  

    return clip_input, slices 





def ReadSegmentRGB(root, path, offsets, new_height, new_width, new_length, is_color, name_pattern, duration,start,stop):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    debug = False 

    sampled_list = []
   
    if debug: 
        print("offsets = {}".format(offsets)) 
        print("(new_height = {}, new_width = {}, new_length = {})".format(new_height, new_width, new_length)) 
        print("duration = {}, start = {}, stop = {}".format(duration,start,stop)) 



    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            loaded_frame_index = length_id + offset   
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)

            moded_loaded_frame_index += start-1 

            frame_name = name_pattern % (moded_loaded_frame_index)
            frame_path = path + "/" + frame_name
            if debug: 
                print(offset,length_id,loaded_frame_index, moded_loaded_frame_index, frame_name)  
                print("frame_path={}".format(frame_path))    

            if cv2_loader: 
                cv_img = cv2_loader(frame_path,  (new_width, new_height), is_color)
            else: 
                cv_img = pil_loader(frame_path,  (new_width, new_height))

            if debug: 
                print("cv_img.type = {}, cv_img.shape = {}".format(type(cv_img), cv_img.shape)) 

            sampled_list.append(cv_img)

    #print("sampled_list.len = {}, shape = {}".format(len(sampled_list),sampled_list[0].shape)) 
    clip_input = np.concatenate(sampled_list, axis=2)
    if debug: 
        print("clip_input.shape = {}".format(clip_input.shape)) 

    return clip_input


def ReadSegmentFlow(path, offsets, new_height, new_width, new_length, is_color, name_pattern,duration):
    if is_color:
        cv_read_flag = cv2.IMREAD_COLOR         # > 0
    else:
        cv_read_flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    sampled_list = []
    for offset_id in range(len(offsets)):
        offset = offsets[offset_id]
        for length_id in range(1, new_length+1):
            loaded_frame_index = length_id + offset
            moded_loaded_frame_index = loaded_frame_index % (duration + 1)
            if moded_loaded_frame_index == 0:
                moded_loaded_frame_index = (duration + 1)
            frame_name_x = name_pattern % ("x", moded_loaded_frame_index)
            frame_path_x = path + "/" + frame_name_x
            cv_img_origin_x = cv2.imread(frame_path_x, cv_read_flag)
            frame_name_y = name_pattern % ("y", moded_loaded_frame_index)
            frame_path_y = path + "/" + frame_name_y
            cv_img_origin_y = cv2.imread(frame_path_y, cv_read_flag)
            if cv_img_origin_x is None or cv_img_origin_y is None:
               print("Could not load file %s or %s" % (frame_path_x, frame_path_y))
               sys.exit()
               # TODO: error handling here
            if new_width > 0 and new_height > 0:
                cv_img_x = cv2.resize(cv_img_origin_x, (new_width, new_height), interpolation)
                cv_img_y = cv2.resize(cv_img_origin_y, (new_width, new_height), interpolation)
            else:
                cv_img_x = cv_img_origin_x
                cv_img_y = cv_img_origin_y
            sampled_list.append(np.expand_dims(cv_img_x, 2))
            sampled_list.append(np.expand_dims(cv_img_y, 2))

    clip_input = np.concatenate(sampled_list, axis=2)
    return clip_input


class covid(data.Dataset):

    def __init__(self,
                 root,
                 source,
                 phase,
                 modality,
                 name_pattern=None,
                 is_color=True,
                 num_segments=1,
                 new_length=1,
                 new_width=0,
                 new_height=0,
                 transform=None,
                 target_transform=None,
                 video_transform=None,
                 ensemble_training = False,
                 is_train = True):

        classes, class_to_idx = find_classes(root)
        clips = make_dataset(root, source)

        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))

        self.root = root
        self.source = source
        self.phase = phase
        self.modality = modality

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.clips = clips
        self.ensemble_training = ensemble_training
        self.is_train = is_train 

        if name_pattern:
            self.name_pattern = name_pattern
        else:
            if self.modality == "rgb" or self.modality == "CNN" :
                self.name_pattern = "%d.jpg"  #"img_%05d.jpg"
            elif self.modality == "flow":
                self.name_pattern = "flow_%s_%05d.jpg"


        self.is_color = is_color
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_width = new_width
        self.new_height = new_height

        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform

    def __getitem__(self, index):

        #print("Starting index {} data generator".format(index))  


        path, duration, start, stop, target,bbox = self.clips[index]
        target = self.class_to_idx[target] 
      
        #print("index = {}, path = {}, duration = {}, target = {}".format(index, path, duration, target)) 

        duration = duration - 1
        average_duration = int(duration / self.num_segments)
        average_part_length = int(np.floor((duration-self.new_length) / self.num_segments))
        offsets = []


        for seg_id in range(self.num_segments):
            if self.phase == "train":
                if average_duration >= self.new_length:
                   
                    # No +1 because randint(a,b) return a random integer N such t
                    offset = random.randint(0, average_duration - self.new_length)
                    # No +1 because randint(a,b) return a random integer N such that a <= N <= b.
                    offsets.append(offset + seg_id * average_duration)
                elif duration >= self.new_length:

                    offset = random.randint(0, average_part_length)
                    offsets.append(seg_id*average_part_length + offset)
                else:
                    increase = random.randint(0, duration)
                    offsets.append(0 + seg_id * increase)
            elif self.phase == "val":
                if average_duration >= self.new_length:
                    offsets.append(int((average_duration - self.new_length + 1)/2 + seg_id * average_duration))
                elif duration >= self.new_length:
                    offsets.append(int((seg_id*average_part_length + (seg_id + 1) * average_part_length)/2))
                else:
                    increase = int(duration / self.num_segments)
                    offsets.append(0 + seg_id * increase)
            else:
                print("Only phase train and val are supported.")
       
        #print("num_segments = {},new_length = {} ,average_duration = {}, duration = {}, offsets = {}".format(self.num_segments,self.new_length, average_duration , average_part_length, offsets )) 
 

        if self.modality == "rgb" or self.modality == "CNN":
            clip_input,slices = ReadSegmentRGBMaskResample(self.root, path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration, 
                                        start, 
                                        stop, self.is_train,bbox  
                                        )
        elif self.modality == "flow":
            clip_input = ReadSegmentFlow(self.root, path,
                                        offsets,
                                        self.new_height,
                                        self.new_width,
                                        self.new_length,
                                        self.is_color,
                                        self.name_pattern,
                                        duration
                                        )
            
        else:
            print("No such modality %s" % (self.modality))

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)   
            #print("Finishing index {} data generator".format(index))  


        #print("clip_input.shape = {}, target={}".format(clip_input.shape,target)) 
        #input('dbg dataloaded') 

        return clip_input, target, path, slices  


            


    def __len__(self):
        return len(self.clips)
