CT Lung Images Segmentation
===
CT lung images segmentation implementation using UNet. 

![u-net-architecture](img/u-net-architecture.png)
 
Overview

### Data
1. The Kaggle dataset at https://www.kaggle.com/kmader/finding-lungs-in-ct-data.
2. The CNBC dataset at http://ncov-ai.big.ac.cn/download


### Pre-Processing
1. On the Kaggle dataset, after importing the image data from the alpha channel, convert unsigned int image to int and resize 512 x 512
2. On the CNBC dataset, merge lung field mask, GC lesions and consolidation lesion marks all to the lung field mask.

Requirement
---
* Python
* Keras
* Python packages : numpy, matplotlib, opencv, and so on...

### Reference Implementations
---
+ https://github.com/IzPerfect/CT-Image-Segmentation

