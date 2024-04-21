import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2 as cv
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
from Data_256.UCF101 import get_ucf101
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
from vc_cluster_fine import main_vc
from simmat_finer import main_sim
from mix_model_lear_finer import main_mix

def main():
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_argument('--Dataset',default="UCF101",type = str,help = "Dataset")
    parser.add_argument('--train_vMF',default=True,type = bool,help = "train vmf kernel")
    parser.add_argument('--train_simmat',default=True,type = bool,help = "train similarity matrix")
    parser.add_argument('--train_mixture_model',default=True,type = bool,help = "train mixture_moodel")
    parser.add_argument('--train',default = True,type = bool,help = "train model")
    parser.add_argument('--checkpoint_path',default = '',type = str)
    parser.add_argument('--checkpoint',default = '',type = str)
    
    parser.add_argument('--simmat_save_name',default = '',type = str)
    parser.add_argument('--mix_model_save',default='',type = str)
    parser.add_argument('--dict_name',default = '',type = str,)
    parser.add_argument('--data_path',default = '',type = str,)
    
    args = parser.parse_args()
    
    if False:
        main_vc(dataset = args.Dataset,checkpoint = args.checkpoint_path,data_path=args.data_path,fname = args.dict_name)
    if True:
        main_sim(dataset = args.Dataset,checkpoint=args.checkpoint,data_path=args.data_path,fname=args.dict_name,save_name = args.simmat_save_name)
    if args.train_mixture_model:
        main_mix(dataset = args.Dataset,checkpoint=args.checkpoint,data_path=args.data_path,fname=args.dict_name,matrix_save =args.simmat_save_name, save_name = args.mix_model_save)
    if args.train:
        if args.Dataset == "UCF101":
            from compose_train_2 import main as train
        elif args.Dataset == "Kinetics":
            from compose_train import main as train
        train(args.dict_name,args.mix_model_save,args.data_path,args.checkpoint)

if __name__ == "__main__":
    main()
        
    
    
