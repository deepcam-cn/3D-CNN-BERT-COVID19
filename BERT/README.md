# 3D-CNN-BERT for COVID19 Classification and Embedding Feature Generation 

We use the Pytorch implementation of [Late Temporal Modeling in 3D CNN Architectures with BERT for Action Recognition], which implements late temporal modeling on top of the 3D CNN architectures with main focus on BERT. This architecture was originally used for video action recogntion. We modify it and apply it onto 3D CNN classification on 3D CT-scan volumes.   

!BERT.png 

## Dependency 

## Dataset preparation
Follow the instructions in preprocessing and segmentation to prepare the dataset 

## Traing 
python train.py --split=5 --arch=rgb_r2plus1d_32f_34_bert10  --workers=8 --batch-size=4 --iter-size=16 --print-freq=100 --dataset=covid --dataset_root=/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/  --lr=1e-5

The data generator is defined in dataset/covid.py 
Set the dataset_root to your own root directory where datasets are saved.  

## Evuation on validation dataset  
python train.py --split=5 --arch=rgb_r2plus1d_32f_34_bert10  --workers=8 --batch-size=4 --iter-size=16 --print-freq=100 --dataset=covid --dataset_root=/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/  --lr=1e-5 --evaluate 

## Produce prediction on test dataset 
python train.py --split=5 --arch=rgb_r2plus1d_32f_34_bert10  --workers=8 --batch-size=4 --iter-size=16 --print-freq=100 --dataset=covid --dataset_root=/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/  --lr=1e-5 --evaluate --test 

## Reference implementatoin
https://github.com/artest08/LateTemporalModeling3DCNN












