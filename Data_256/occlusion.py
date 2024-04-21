import numpy as np
import cv2 as cv
import random
import os
import xml.etree.ElementTree
import PIL.Image
import math


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    factor = min(100,factor)
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv.INTER_LINEAR if factor > 1.0 else cv.INTER_AREA
#rint(new_size,factor,factor)
    return cv.resize(im, new_size,fx=factor, fy=factor,interpolation=interp) # interpolation=interp)


def paste_over(im_src, im_dst, center,side):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    im_src` becomes visible).
    Args:
    im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
    im_dst: The target image.
    alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
    at each pixel. Large values mean more visibility for `im_src`.
    center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255
    
    occ_height,occ_width,_ = region_src.shape
    img_height,img_width,_ = im_dst.shape 
    scale = np.sqrt((side*224*224)/(np.sum(region_src[:,:,3]>0)+1e-5))#224*224
    
    im_src = resize_by_factor(im_src,scale)
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])
    
    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src
    

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255
    #print(np.sum(alpha>0)/(224*224))
    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


class occlude:
    def __init__(self,img_shape,occluder_index,occluder_size,occluder_motion):
        self.occluder_index = occluder_index
        self.occluders = self.load_occluders()
        self.occluder = [self.occluders[occluder_index]]
        self.occ_idx =  occluder_index
        self.occluder_size = occluder_size#in % of area to be covered
        self.occluder_motion = occluder_motion# String
        self.width_height = np.asarray([img_shape[1], img_shape[0]])
        self.center = self.random_placement()
        self.theta = random.randint(-89,89)
        self.scale = {0:1,10:1,20:1,30:1,40:1,50:1.1,60:1.2,80:1.27}
        self.occ_scale = {0:1,1:1.2,2:1.2,3:1.1,4:1.05,5:1.1,6:1.1}
        self.occluder_size = self.occluder_size*self.scale[self.occluder_size]*self.occ_scale.get(occluder_index,1.1)
        
        self.motion_dict = {"random_placement":self.random_placement,"random_motion":self.random_motion,"linear_motion":self.linear_motion,"circular_motion":self.circular_motion,"static":self.static,"sine_motion":self.sine_motion}
        self.motion_choice = occluder_motion
        self.motion = self.motion_dict[occluder_motion]
        self.radius = np.sqrt(np.sum((self.center-self.width_height/2)**2))
    
        #self.center = self.width_height/2
        
               
    def load_occluders(self):
        occluders = [] 
        structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
        annotations = [os.path.join("./Data_256/occ_data/attributes",path) for path in os.listdir("./Data_256/occ_data/attributes/")]
        annotations.sort()
        annotations = [path for path in annotations if ".xml" in path ]
        for annotation_path in annotations:
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')
            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
            #is_person = (obj.find('name').text == 'person')
            #is_difficult = (obj.find('difficult').text != '0')
            #is_truncated = (obj.find('truncated').text != '0')
            #if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))
        
        
        
            if not boxes:
                continue
        
        
            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')
            im_path = os.path.join("./Data_256/occ_data/images",im_filename)
            seg_path = os.path.join("./Data_256/occ_data/segmentation",seg_filename)
            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))  
       
            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv.countNonZero(object_mask) < 2:
                #Ignore small objects
                    continue

            # Reduce the opacity of the mask along the border for smoother blending
                eroded = cv.erode(object_mask, structuring_element)
                object_mask[eroded < object_mask] = 192
                object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
                object_with_mask = resize_by_factor(object_with_mask, 0.5)
                occluders.append(object_with_mask)
        return occluders

    def occlude_with_objects(self,im,epoch):
        """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
        occluders = self.occluder
        im = np.array(im)
        result = im.copy()
        h = im.shape[0]
        w,h = self.width_height
        width_height = np.asarray([im.shape[1], im.shape[0]])
        occluder = random.choice(occluders)
        center = self.motion(epoch)
        
        paste_over(im_src=occluder, im_dst=result, center=center,side = self.occluder_size/100)

        return result

    def random_placement(self,epoch=1):
        return np.random.uniform([0,0], self.width_height)
    def linear_motion(self,epoch=1):
        w,h = self.width_height
        x_step = 5
        y_step = 5*np.tan(math.pi*self.theta/180)
        st_h = self.center[1]
        st_w = self.center[0]
        new_c_h = (st_h+epoch*x_step)%h
        new_c_w = (st_w+epoch*y_step)%w
        return np.array([new_c_w,new_c_h])
    def random_motion(self,epoch):
        w,h = self.width_height
        delta_x = random.uniform(-.15,.15)*h
        delta_y = random.uniform(-.15,.15)*w
        st_h = (self.center[1] + delta_x)%h
        st_w = (self.center[0] + delta_y)%w
        self.center = np.array([st_w,st_h])
        return np.array([st_w,st_h])
    def train_randomize(self):
        self.center = self.random_placement(0)
        self.theta = random.randint(-89,89)
        motion_choices = list(self.motion_dict.keys())
        occ_choice =  random.choice(range(0,50)) 
        size_choices = [20,40]
        self.motion_choice = random.choice(motion_choices[1:3])
        size_choice = random.choice(size_choices)
        self.occluder_size = size_choice
        self.motion = self.motion_dict[self.motion_choice]
        self.occluder = [self.occluders[occ_choice]]
        self.occ_idx =  occ_choice
        
    def circular_motion(self,epoch):
        new_c_h = (self.radius/(1+epoch*.1))*np.cos(math.pi*(self.theta+epoch*20)/180)
        new_c_w = (self.radius/(1+.1*epoch))*np.sin(math.pi*(self.theta+epoch*20)/180)
        center = np.array([new_c_w,new_c_h])+self.width_height/2
        return  center
    def static(self,epoch):
        return self.center
    def sine_motion(self,epoch):
        w,h = self.width_height
        x_step = 5
        st_h = self.center[1]
        st_w = self.center[0]
        new_c_h = (st_h+epoch*x_step)%h
        new_c_2 = ((w/1.225)*np.sin(new_c_h))%w
        self.center = np.array([st_w,st_h])
        return np.array([st_w,st_h])
    
    def initialize(self):
        self.center = self.random_placement(0)
        self.theta = random.randint(-89,89)
        self.radius = np.sqrt(np.sum((self.center-self.width_height/2)**2))
        
    def test_randomizer(self):
        self.center = self.random_placement(0)
        self.theta = random.randint(-89,89)
        motion_choices = ["random_placement","circular_motion"]
        l = list(range(50))[:38]
        l2=list(range(50,100))[:12]
        occ_choice =  random.choice(l+l2) 
        size_choices = [40,50,60]
        self.motion_choice = random.choice(motion_choices)
        size_choice = random.choice(size_choices)
        #self.motion_choice = random.choice(motion_choices[1:3])
        #self.motion = self.motion_dict[self.motion_choice]
        
        self.occluder_size = size_choice*self.scale[size_choice]*self.occ_scale.get(occ_choice,1.25)
        self.motion = self.motion_dict[self.motion_choice]
        self.occluder = [self.occluders[occ_choice]]
        self.occ_idx =  occ_choice
    def set_val(self,occ_dict):
        self.center = self.random_placement(0)
        self.theta = random.randint(-89,89)
        self.occluder_size = occ_dict["occluder_size"]#size_choice*self.scale[size_choice]*self.occ_scale.get(occ_choice,1.15)
        self.motion_choice = occ_dict["motion_choice"]
        self.motion = self.motion_dict[occ_dict["motion_choice"]]
        self.occluder = [self.occluders[occ_dict["occ_choice"]]]
        self.occ_idx =  occ_dict["occ_choice"]                
    
                         
    def get_val(self):
        return  self.occluder_size, self.motion_choice, self.occ_idx                     
                             
                    
            


        
