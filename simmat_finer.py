from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from Initialization_Code.vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
import numpy as np
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Code.vMFMM import *
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor
import pickle
import os
import numpy as np
import pytorchvideo
import torchvision
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    RandomCrop,
)
from Kinetics.dataset import labeled_video_dataset
from feature_extractor_models import feature_mvit,feature_mvit_kin

    
def main_sim(dataset = "UCF101",sim_dir= '',checkpoint='',data_path='',fname='',save_name = ''):
    if dataset == "UCF101":
        model = feature_mvit(checkpoint)
        num_classes = 101
    elif dataset == "Kinetics":
        model = feature_mvit_kin()
        num_classes = 400#
    model = model.cuda()
    train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(16),
                    RandomCrop(224),
                 ]
                ),
              ),
            ]
        )
    
    
    with open("/home/sh009885/code/ucf101-supervised/ucf101-supervised-main/models/init_vgg/dictionary_vgg/dictionary_fin_mvit_pretrained_768_p.pickle",'rb') as fh:#open(dict_dir+'dictionary_{}_{}.pickle'.format(fname,"768"), 'rb') as fh:
        centers = pickle.load(fh)
#
    paral_num = 10
    nimg_per_cat =  72
    imgs_par_cat =np.zeros(num_classes)
    occ_level='ZERO'
  
    for cat in range(num_classes):
        N= 72
        if dataset == "UCF101":
            train_dataset, test_dataset =  get_ucf101(cl= cat,root = 'Data_256',frames_path =data_path)
 #      train_loader = DataLoader(
        elif dataset == "Kinetics":
            train_dataset = labeled_video_dataset(cl = cat,data_path = os.path.join(data_path, "train"),clip_sampler=pytorchvideo.data.make_clip_sampler("random",2),transform = train_transform)
        train_loader = DataLoader(

           train_dataset,
       batch_size=1,
       num_workers=4,
        drop_last=True,
        pin_memory=True)

    
        savename = os.path.join("/home/sh009885/code/ucf101-supervised/ucf101-supervised-main",sim_dir,'{}_simmat_pretrained_mthrh045_{}_K{}.pickle'.format(save_name,cat,"768"))
        ii =0
        if  not os.path.exists(savename) or True:
            r_set = [None for nn in range(N)]
            
            for iii,i in enumerate(train_loader):
                x = i["video"]
                y = i["label"]
                y = int(y.detach().numpy())
                cat_idx = cat
                if y == cat and imgs_par_cat[cat]<N :
                    with torch.no_grad():
                        x = x.cuda()
                        
                        layer_feature = model(x).detach().cpu().numpy()
                        layer_feature = layer_feature.squeeze(0)
                        iheight,iwidth = layer_feature.shape[-2:]
                        lff = layer_feature.reshape(layer_feature.shape[0],-1).T
                        lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
                        r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
                        imgs_par_cat[cat_idx]+=1
                    ii+=1
            print('Determine best threshold for binarization - {} ...'.format(cat))
            nthresh=20
            magic_thhs=range(nthresh)
            coverage = np.zeros(nthresh)
            act_per_pix = np.zeros(nthresh)
            layer_feature_b = [None for nn in range(100)]
            magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])
            for idx,magic_thh in enumerate(magic_thhs):
                for nn in range(72):
                    layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
                    coverage[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0)>0)
                    act_per_pix[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0))
            coverage=coverage/144
            act_per_pix=act_per_pix/144
            best_loc=(act_per_pix>2)*(act_per_pix<15)
            if np.sum(best_loc):
                best_thresh = np.min(magic_thhs[best_loc])
            else:
                best_thresh = 0.45
            layer_feature_b = [None for nn in range(N)]
            for nn in range(N):
                layer_feature_b[nn] = (r_set[nn]<best_thresh).astype(int).T
            print('Start compute sim matrix ... magicThresh {}'.format(best_thresh))
            _s = time.time()

            mat_dis1 = np.ones((N,N))
            mat_dis2 = np.ones((N,N))
            N_sub = 144
            sub_cnt = int(math.ceil(N/N_sub))
            for ss1 in range(sub_cnt):
                start1 = ss1*N_sub
                end1 = min((ss1+1)*N_sub, N)
                layer_feature_b_ss1 = layer_feature_b[start1:end1]
                for ss2 in range(ss1,sub_cnt):
                    print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
                    _ss = time.time()
                    start2 = ss2*N_sub
                    end2 = min((ss2+1)*N_sub, N)
                    if ss1==ss2:
                        inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
                        para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))
                    else:
                        layer_feature_b_ss2 = layer_feature_b[start2:end2]
                        inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
                        para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))

                    mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
                    mat_dis2[start1:end1, start2:end2] = para_rst[:,1]

                    _ee = time.time()
                    print('comptSimMat iter time: {}'.format((_ee-_ss)/60))

            _e = time.time()
            print('comptSimMat total time: {}'.format((_e-_s)/60))

            with open(savename, 'wb') as fh:
                print('saving at: '+savename)
                print(mat_dis1.shape,mat_dis2.shape)
                pickle.dump([mat_dis1, mat_dis2], fh)


if __name__ == "__main__":
    main_sim("Kinetics")
    
