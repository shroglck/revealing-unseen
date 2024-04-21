from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import cv2
import torch
import json
from skimage.transform import resize
from skimage import img_as_bool
from PIL import Image
import pdb
import random
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
from torchvision import transforms
from pytorchvideo.transforms import MixVideo
from .spatial_transforms import ToTensor
import torchvision


class UCF101TRAIN(Dataset):
    def __init__(self, cl = None,root = '', train=True, fold=1, transform=None, frames_path=''):

        self.root = root
        self.cl = cl
        self.frames_path = frames_path
        self.train = train
        self.fold = fold
        self.video_paths, self.targets = self.build_paths()
        self.targets = np.array(self.targets)
        self.transform = transform
        

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path, video_label = self.video_paths[idx], self.targets[idx]
        video = self.get_video(video_path)
        sample = {"video":video,
                  "label":video_label,
                  "video_path":video_path}
        return sample#video, video_label,video_path,occ

    def get_video(self, video_path):
        no_frames = len(os.listdir(video_path))
        skip_rate = 1
        total_frames = 16*skip_rate

        if total_frames > no_frames:
            skip_rate = skip_rate -1
            if skip_rate == 0:
                skip_rate = 1
            total_frames = 16*skip_rate

        try:
            start_frame = random.randint(0, no_frames - total_frames) ## 32, 16 frames
        except:
            start_frame = 0
        video_container = []
        for item in range(start_frame, start_frame + total_frames, skip_rate):
            image_name = "frame"+str(item+1).zfill(6) + '.jpg'
            image_path = os.path.join(video_path, image_name)
            current_image = Image.open(image_path).convert('RGB')
            video_container.append(current_image)
        
        tr_transform = transforms.Compose([transforms.RandAugment()])
        tr2_transform = transforms.Compose([ToTensor(1),torchvision.transforms.Lambda(lambda x:x/255.0)])
        if self.transform is not None:
            self.transform.randomize_parameters()
            clip = [self.transform(img) for img in video_container]
            clip = [transforms.functional.normalize(tr2_transform(img),normal_mean,normal_std) for img in clip]
           
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        return clip


    def build_paths(self):
        data_paths = []
        targets = []
        if self.train:
            annotation_path = os.path.join(self.root, 'ucfTrainTestlist', f'trainlist0{self.fold}.txt')
        else:
            annotation_path = os.path.join(self.root, 'ucfTrainTestlist', f'testlist0{self.fold}.txt')
        
        annotation_data = {}
        with open(annotation_path, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            for item in data:
                annotation_data[item[0].replace('.avi','')] = int(item[1])-1

        for key in annotation_data:
            
            if self.cl is None:
                data_paths.append(os.path.join(self.frames_path, key)) 
                targets.append(annotation_data[key])
            elif self.cl == int(annotation_data[key]):
                data_paths.append(os.path.join(self.frames_path, key)) 
                targets.append(annotation_data[key])
        
        

        return data_paths, targets

