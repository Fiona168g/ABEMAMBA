# Getting started

### Dependencies
Please install following essential dependencies:
```
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
```
The pre-processing procedure can be found in [Ouyang et al.](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535).

### Data
The dataset can be downloaded by:
1) [Pre-processed CHAOS-T2 data and supervoxels](https://drive.google.com/drive/folders/1elxzn67Hhe0m1PvjjwLGls6QbkIQr1m1?usp=share_link)
2) [Pre-processed SABS data and supervoxels](https://drive.google.com/drive/folders/1pgm9sPE6ihqa2OuaiSz7X8QhXKkoybv5?usp=share_link)
3) [Pre-processed CMR data and supervoxels](https://drive.google.com/drive/folders/1aaU5KQiKOZelfVOpQxxfZNXKNkhrcvY2?usp=share_link)

### Training
First, compile `./supervoxels/felzenszwalb_3d_cy.pyx` using Cython by running (`python ./supervoxels/setup.py build_ext --inplace`), then execute `./supervoxels/generate_supervoxels.py` 
Second, download pre-trained ResNet-101 weights (https://download.pytorch.org/models/resnet101-63fe2227.pth). Place the downloaded weights in your checkpoints folder, then update the absolute path in the code at `./models/encoder.py`.  
Finally, execute the training script by running `./script/train.sh`

### Inference
Run the test script with `./script/test.sh` 

### Acknowledgement
Our code is based the works: 
[SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), 
[ADNet](https://github.com/sha168/ADNet),
[QNet](https://github.com/ZJLAB-AMMI/Q-Net)



