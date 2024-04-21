import torch
import torch.nn as nn
import numpy as np
import pytorchvideo
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from Initialization_Code.vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Code.vMFMM import *
import glob
import pickle
import os
import torchvision
from PIL import Image
import pytorchvideo
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    RandomCrop,
    Lambda
)
from feature_extractor_models import feature_mvit,feature_mvit_kin


def train(model,train_loader,vMF_kappa=76,fname='',num_classes =101):
    
    img_per_cat = 72
    samp_size_per_img = 20
    imgs_par_cat =np.zeros(num_classes)
    bool_load_existing_cluster = False
    bins = 4
    loc_set = []
    feat_set = []
    nfeats = 0
    vc_num = 768
    fname = []
    for ii,i in enumerate(train_loader):
        try:
            x = i["video"]
            y= i["label"]
            y = int(y.detach().numpy())
            if y in list(range(num_classes)) and imgs_par_cat[y]<img_per_cat:
                x = x.cuda()
                fname = []

                with torch.no_grad():
                    tmp = model(x).detach().cpu().numpy()
                    tmp = tmp.squeeze(0)
                    height, width = tmp.shape[-2:]
                    tmp = tmp[:,offset:height - offset, offset:width - offset]
                    gtmp = tmp.reshape(tmp.shape[0], -1)
                    if gtmp.shape[1] >= samp_size_per_img:
                        rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
                    else:
                        rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
                    tmp_feats = gtmp[:, rand_idx].T
                    cnt = 0
                    for rr in rand_idx:
                        ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
                        hi = (ihi+offset)*(x.shape[2]/height)-Apad
                        wi = (iwi + offset)*(x.shape[3]/width)-Apad
                        loc_set.append([y, ii,hi,wi,hi+Arf,wi+Arf])
                        feat_set.append(tmp_feats[cnt,:])

                        cnt+=1

                imgs_par_cat[y]+=1
        except:
            pass


    feat_set = np.asarray(feat_set)
    loc_set = np.asarray(loc_set).T
    
    new_model = vMFMM(768, 'k++')
    new_model.fit(feat_set, 76, max_it=150)
    with open(dict_dir+'dictionary_{}_{}.pickle'.format(fname,"768"), 'wb') as fh:
        pickle.dump(new_model.mu, fh)

    

   
def main_vc(dataset,kappa=76,checkpoint='',data_path='',fname=''):
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
        )
    
    if dataset == "UCF101":
        train_dataset, test_dataset =  get_ucf101(root = 'Data_256',frames_path =data_path)
        model = feature_mvit(checkpoint).cuda()
        num_classes = 101
    elif dataset == "Kinetics":
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(data_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),
           transform=train_transform
      )
        model = feature_mvit_kin().cuda()
        num_classes=400
    
        
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    train(model,train_loader,fname,num_classes)
    
if __name__ == "__main__":
        main_vc("Kinetics",76)
    
