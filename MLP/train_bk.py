import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import numpy as np 
from torch.optim import lr_scheduler
import shutil
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--desc', type=str, default='rrr', help='descripton of the channels of image')
parser.add_argument('--gpu_id',type = int,default=0, help='foo help')

parser.add_argument('--activation', type=str, default='Sigmoid', help='descripton of the channels of image')
parser.add_argument('--pooling', type=str, default='both', help='descripton of the channels of image')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)


class CovidFeatureDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, root):
        'Initialization'
        self.root = root
        feature_files = os.listdir(root) 
        #print(feature_files) 

        self.ids = [] 
        self.data = [] 
        self.labels = [] 

        classes = {'covid':0, 'non-covid':1} 

        for f in feature_files:

            #print(f) 

            f_splits = f.split('_') 
            label = classes[f_splits[2]] 
            s_id = int(f_splits[-1].split('.')[0]) 

            with open(os.path.join(root,f), 'rb') as pk_file:
                features = pickle.load(pk_file)     
           
            #print(features.shape) 
            
            feature_max = np.amax(features,axis = 0) 
            feature_max = np.expand_dims(feature_max, axis=0) 
            feature_avg = np.mean(features,axis = 0) 
            feature_avg = np.expand_dims(feature_avg, axis=0) 

            feature_both = np.concatenate((feature_max,feature_avg),axis=1) 
            #print(feature_max.shape, feature_avg.shape,feature_both.shape) 
            self.data.append((feature_max,feature_avg,feature_both)) 
            self.ids.append(s_id) 
            self.labels.append(label) 

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sid = self.ids[index]
        # Load data and get label
        X = self.data[index] 
        y = self.labels[index]

        return X,y,sid 


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
 '''

  def __init__(self,input_size=512,activation="ReLU"):
    super().__init__()

    if activation=="Sigmoid":
        act = nn.Sigmoid()
    elif  activation=="Tanh":
        act = nn.Tanh()
    else:
        act = nn.ReLU()

    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_size, 128),
      act, # nn.Sigmoid(), #Sigmoid, ReLU or Tanh
      nn.Dropout(0.5), 
      nn.Linear(128, 32),
      act, #nn.Sigmoid(),
      nn.Dropout(0.5),
      nn.Linear(32, 2)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
 

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)
 
  
def train(mlp,train_loader,feature_index,optimizer,loss_function): 

    # Set current loss value
    total_loss = 0 
    # Iterate over the DataLoader for training data
    count = 0
    correct = 0 
    # switch to train mode
    mlp.train()

    for i, data in enumerate(train_loader):      

      #print("i={}".format(i)) 

      # Get inputs
      inputs, targets, ids = data
      
      inputs = inputs[feature_index].cuda() 
      targets= targets.cuda()  

      #print(inputs.shape) 
      #print(targets.shape) 
      #print(ids.shape) 
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)

      #print(outputs.shape) 
      
      prec1 = accuracy(outputs.data, targets)

      #print(prec1.item) 

      correct += prec1.item()


      # Compute loss
      loss = loss_function(outputs, targets)
      total_loss += loss* outputs.size(0)
      count +=  outputs.size(0)

      #print(correct,count) 
      
      # Perform backward pass
      loss.backward()      
      # Perform optimization
      optimizer.step()
      
      #if (i+1) % 10 == 0:
      #    print('i = {} acc = {} loss = {}'.format(i, correct/count,total_loss/count))
      #input('dbg') 

    return correct, total_loss, count 




if __name__ == '__main__':
  
  train_feature_dir = 'features_train_{}'.format(args.desc)
  val_feature_dir = 'features_val_{}'.format(args.desc)
  print("train_feature_dir = {}".format(train_feature_dir))    
  print("val_feature_dir = {}".format(val_feature_dir))    

  train_dataset = CovidFeatureDataset(root=train_feature_dir)
  val_dataset = CovidFeatureDataset(root=val_feature_dir)

  print("training data length = {}, validation data length = {}".format(len(train_dataset),len(val_dataset))) 

  pooling = args.pooling # "both" 
  activation = args.activation 

  poolings = {"max":0,"avg":1,"both":2} 
  

  saveLocation = "./checkpoint/"
  if not os.path.exists(saveLocation):
      os.makedirs(saveLocation) 

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
  val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

  
  # Initialize the MLP
  if pooling == "Max":
      mlp = MLP(input_size=512,activation=activation).cuda()
      feature_index = 0 
  elif pooling == "Avg":
      mlp = MLP(input_size=512,activation=activation).cuda()
      feature_index = 1
  else: 
      mlp = MLP(input_size=1024,activation=activation).cuda()
      feature_index = 2 
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

  best_prec1 = 0  
  best_loss = 100  
  is_best = False 

  # Run the training loop
  for epoch in range(0, 100): # 5 epochs at maximum
    
    # Print epoch
    #print(f'Starting epoch {epoch+1}')
    
    correct, total_loss, count = train(mlp,train_loader,feature_index,optimizer,loss_function) 
    '''

   
    # Set current loss value
    total_loss = 0 
    # Iterate over the DataLoader for training data
    count = 0
    correct = 0 
    # switch to train mode
    mlp.train()

    for i, data in enumerate(train_loader):  
      #print("i={}".format(i)) 

      # Get inputs
      inputs, targets, ids = data
      
      inputs = inputs[feature_index].cuda() 
      targets= targets.cuda()  

      #print(inputs.shape) 
      #print(targets.shape) 
      #print(ids.shape) 
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)

      #print(outputs.shape) 
      
      prec1 = accuracy(outputs.data, targets)

      #print(prec1.item) 

      correct += prec1.item()


      # Compute loss
      loss = loss_function(outputs, targets)
      total_loss += loss* outputs.size(0)
      count +=  outputs.size(0)

      #print(correct,count) 
      
      # Perform backward pass
      loss.backward()      
      # Perform optimization
      optimizer.step()
      
      #if (i+1) % 10 == 0:
      #    print('i = {} acc = {} loss = {}'.format(i, correct/count,total_loss/count))
      #input('dbg') 
    ''' 

    prec1_train = correct/count
    avg_loss = total_loss/count 

    print("Train Epoch =  {} Prec@1 = {} loss = {}".format(epoch, prec1_train,avg_loss)) 

    # evaluate on validation set
    if prec1_train > 0.85: 
    
        mlp.eval()

        correct = 0 
        count = 0 
        prec1 = 0.0
        total_loss = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):      
                # Get inputs
                inputs, targets, ids = data
                inputs = inputs[feature_index].cuda() 
                targets= targets.cuda()  
      
                # Perform forward pass
                outputs = mlp(inputs)
    
                prec1 = accuracy(outputs.data, targets)
                correct += prec1.item()
 
                # Compute loss
                loss = loss_function(outputs, targets)
                total_loss += loss* outputs.size(0)
                count +=  outputs.size(0)

        avg_loss = total_loss/count 
        scheduler.step(avg_loss)

        prec1 = correct/count     
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        best_loss = min(best_loss,avg_loss)  
 
        print("Validation Epoch =  {} Prec@1 = {} loss = {}".format(epoch, prec1,avg_loss)) 

        checkpoint_name = "%s_%s_%03d_%f_%f_%s" % (activation,pooling, epoch + 1, prec1_train, prec1, "checkpoint.pth.tar")
        if is_best: # or (epoch + 1)%1==0:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': mlp.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                     }, is_best, checkpoint_name, saveLocation)
    
  checkpoint_name = "%s_%03d_%f_%f_%s" % (pooling,epoch + 1, prec1_train, prec1, "checkpoint.pth.tar")
  save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': mlp.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_name, saveLocation)



  # Process is complete.
  print('Training process has finished.')

