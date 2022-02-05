import os
import cv2
import torch
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt


def gen_large_mask_v2(img_h, img_w, marg=10):
    """ img_h: int, an image height
        img_w: int, an image width
        marg: int, a margin for a box starting coordinate
        p_irr: float, 0 <= p_irr <= 1, a probability of a polygonal chain mask

        min_n_irr: int, min number of segments
        max_n_irr: int, max number of segments
        max_l_irr: max length of a segment in polygonal chain
        max_w_irr: max width of a segment in polygonal chain

        min_n_box: int, min bound for the number of box primitives
        min_n_box: int, max bound for the number of box primitives
        min_s_box: int, min length of a box side
        max_s_box: int, max length of a box side"""

    min_n_box = 1
    max_n_box = 4
    min_s_box = 50
    max_s_box = 150 #256*256

    minn_irr = 1
    maxn_irr = 4
    max_l_irr = 150
    max_w_irr = 70

    p_irr = 0.5

    inv_mask = np.ones((img_h, img_w)) #we produce a invert mask of size of image
    temp_mask = np.zeros((img_h, img_w)) #we produce a mask of size of image

    if np.random.uniform(0,1) < p_irr: # generate polygonal chain
        n = int(uniform(minn_irr, maxn_irr)) # sample number of segments
        y = int(uniform(0, img_h)) # sample a starting point
        x = int(uniform(0, img_w))

        for _ in range(n):
            a = uniform(0, 2*np.pi) # sample angle
            l = int(uniform(10, max_l_irr)) # sample segment length
            w = int(uniform(5, max_w_irr)) # sample a segment width

            # draw segment starting from (x,y) to (x_,y_) using brush of width w
            x_ = np.clip(int(x + l * np.sin(a)), 0, img_w)
            y_ = np.clip(int(y + l * np.cos(a)), 0, img_h)

            cv2.line(temp_mask, (x, y), (x_, y_), 1.0, w)
            x, y = x_, y_
        mask = temp_mask

    else: # generate Box masks
        n = int(uniform(min_n_box, max_n_box)) # sample number of rectangles

        for _ in range(n):
            h = int(uniform(min_s_box, max_s_box)) # sample box shape
            w = int(uniform(min_s_box, max_s_box))

            x_0 = int(uniform(marg, img_w - marg - w)) # sample upper-left coordinates of box
            y_0 = int(uniform(marg, img_h - marg - h))

            temp_mask[x_0:x_0+w,y_0:y_0+h]=1
            ratio = np.sum(temp_mask)/(img_h*img_w)
            if(ratio < 0.5):
                mask = temp_mask

    imask = np.abs(mask-1)

    return mask, imask

def load_images(image_file):
    my_image = plt.imread(image_file)
    return my_image

def generate_stack(image, normalize = True):
    [w,h,d] = image.shape #getting shape information

    mask, invert_mask = gen_large_mask_v2(w,h)   #generate binary mask for inpainting
    output = np.zeros((w,h,d+1)) #prepare output variable +1 dim for binary mask

    for i in range(d): #convolving mask with each dim
        output[:,:,i] = image[:,:,i]*invert_mask

    output[:,:,d] = mask #adding binary mask at last layer

    if normalize:
        output[:,:,0:3] = output[:,:,0:3]/255

    return output

class MyDataset(Dataset):
    def __init__(self, data_paths):
        self.paths = [f for f in os.listdir(data_paths) if os.path.isfile(f)] #getting name of all images in path to access with index

    def __getitem__(self, index):
        image = load_images(self.paths[index]) # load image as np.array
        stack = generate_stack(image,normalize=True) #function that generates masked image + mask pattern

        x = torch.FloatTensor(stack) # input of model is stack
        x = x.permute(2,0,1)
        y = torch.FloatTensor(image) # GT is original image
        y = y.permute(2,0,1)

        return x, y

    def __len__(self):
        return len(self.paths)
