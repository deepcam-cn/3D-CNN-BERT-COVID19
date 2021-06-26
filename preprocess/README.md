# Steps to preprocess the datasets 

### Step 1: Check the size of image, resize images whose size is not 512x512 to 512x512. 
    Run check-ICCV-MAI-image-size.ipynb, 
    
### Step 2: Find bounding box of the lung part (including bones, tissues), and find the percetage of lung in whole image for every image. An annotation file is generated for the traning, validation, test datasets, respectively.  
    Run crop-images-ICCV-MAI.ipynb 
    
### Step 3: Generate list file traning, validation, test dataset, repectively. These list files are used in the 3D CNN-BERT network. 
    Run generate-BERT-train-val-list-lungmask.ipynb
    
### Step 4 (Optional): Refine the UNet segmentation lung mask. If the original lung mask is saved, then refine lung mask in this step.   
    Run refine-unet-segment-ICCV-MAI.ipynb
    
    
    
    
    
    
