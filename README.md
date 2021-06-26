# 3D-CNN-BERT-COVID19

Implementation of "A 3D CNN Network with BERT For Automatic COVID-19 Diagnosis From CT-Scan Images" for ICCV-2021 MIA COV19D Competition. 

There are Four parts in this project
## Preprocess
Preprocess the CT-scan volume images: check the image size, extract bounding box and percentage of the the lung in the whole image, select images for 3D CNN

## Segmentation
A UNet segmentation network is trained. It is used to segment lung mask of an image. 

## BERT
A 3D CNN network with BERT for CT-scan volume classification and embedding feature extraction 

## MLP
A simple MLP is trained on the extracted 3D CNN-BERT features. This helps the classification accuracy when there are more than one set of images in a CT-scan volume.  

# License
The code of 3D-CNN-BERT-COVID19 is released under the MIT License. There is no limitation for both academic and commercial usage.
