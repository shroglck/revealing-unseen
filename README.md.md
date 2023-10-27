# CTx
Code for CTx-Net for vedio actiton recognition under occlusion

### Installation

The code uses **Python 3.9** and it is tested on PyTorch GPU version 1.11, with CUDA-11.6

### Setup CTx-Net Virtual Environment

```
virtualenv --no-site-packages <your_home_dir>/.virtualenvs/CTx
source <your_home_dir>/.virtualenvs/CTx/bin/activate
```

### Clone the project and install requirements

```
git clone https://github.com/anonymous-rep/video-occ.git
cd video-occ
pip install -r requirements.txt
```

# 


### Dataset ###
All of the used datasets can be downloaded from [link](https://www.crcv.ucf.edu/data1/occlusion/)
 

#### Evaluate the classification performance of a model

Run the following command in the terminal to evaluate a model on the full test dataset:
```
python compose_tester.py 
```


## Initializing CTx-net Parameters

CTx-Net parameters (vMF kernels and mixture models ) are initialized by clustering the feature vectors
Furthermore, we initialize the mixture models by EM-type learning.
The initial cluster assignment for the EM-type learning is computed based on the similarity of the vMF encodings of the training images.

To train the model 
 
```
python train.py --Dataset {Dataset Name} --chechpoint_path {path to save the checkpoint} \
--simmat_save_name {Directory to save similarity matrix} \
--mix_model_save {Directory to save mixture model weights} \
--dict_name {Directory to save vmf kernels} \
--data_path {Path to Dataset}
``` 
##Testing##
To test a model on UCF-101-O or UCF-101-Y-OCC
```
python test_with_occ.py --data_path {data path} --arch {model name} --checkpoint {path to checkpoint}
```
To test a model on K-400-O
```
python kin_train.py --arch {model_name} --data_path {data_path}
```



## Acknowledgement 

This code has been adapted from
```
@inproceedings{CompNet:CVPR:2020,
  title = {Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion},
  author = {Kortylewski, Adam and He, Ju and Liu, Qing and and Yuille, Alan},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jun,
  year = {2020},
  month_numeric = {6}
}

@article{kortylewski2021compositional,
  title={Compositional convolutional neural networks: A robust and interpretable model for object recognition under occlusion},
  author={Kortylewski, Adam and Liu, Qing and Wang, Angtian and Sun, Yihong and Yuille, Alan},
  journal={International Journal of Computer Vision},
  volume={129},
  number={3},
  pages={736--760},
  year={2021},
  publisher={Springer}
}

```

