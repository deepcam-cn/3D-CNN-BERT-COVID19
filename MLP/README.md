MLP Classification on extracted 3D CNN-BERT embeddings
===
 
### Generate embedding data 

python generate_embedding.py --split=5 --subset=train --desc=addSeq_affine --gpu_id=2 --modelfile=047_97.330729_90.909091_checkpoint.pth.tar

Features will be generated and saved in directory featurs_subset_desc. 


### Traing 

python train.py --desc=addSeq_affine --gpu_id=2 --pooling=both --activation=Sigmoid

Models are saved in directory checkpoint, where the traning and validation accuracies are part of the file name.  




