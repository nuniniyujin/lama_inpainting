import argparse
import lpips
import pytorch_fid
import shutil
from tqdm import tqdm
import numpy as np

import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from loss import Loss
from model import Lama
from train import train_loop
from data import MyDataset
from loss_wasserstein import Loss_Wasserstein
from train_wasserstein import train_loop_Wassertstein

parser = argparse.ArgumentParser(description='LaMa model eval file')

parser.add_argument('--data_path', type=str, required=True,
                    help='Path to test data folder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Path to the model checkpoint to continue training')
parser.add_argument('--out_dir', type=str, default='./test_images',
                    help='Path to the model checkpoint to continue training')
parser.add_argument('--experiment', type=str, default='baseline', choices=['baseline', 'conv_change', 'wgan'],
                    help='Choose between experiments [baseline, conv_change, wgan]')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metric_fn = lpips.LPIPS(net='alex').to(device)

model = Lama(experiment=args.experiment) #calling a model
model = model.to(device) #puting in gpu
model.load_state_dict(torch.load(args.checkpoint_path, map_location=device)) #loading weights

dataset = MyDataset(args.data_path)
test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

metric = 0
metric_list = []

if not os.path.exists(args.out_dir):
    # Create a new directory because it does not exist 
    os.makedirs(args.out_dir)

with tqdm(total=len(test_dataloader), unit_scale=True, postfix={'lpips':0.0}, ncols=150) as pbar:
    for i, (stack, gt_image) in enumerate(test_dataloader):
        stack = stack.to(device)

        mask = stack[:,[3],:,:] #recover mask to feed Loss
        gt_image = gt_image.to(device)
        gt_image = gt_image/255 #added to normalize gt

        with torch.no_grad():
            image_reconstructed = model(stack)

        for j in range(image_reconstructed.shape[0]):
            save_image(image_reconstructed[j], args.out_dir+'/'+dataset.paths[(4*i)+j])


        metric += torch.mean(metric_fn(gt_image, image_reconstructed)).item()
        metric_list.append(torch.mean(metric_fn(gt_image, image_reconstructed)).item())

        pbar.set_postfix({'lpips':metric/(i+1)})
        pbar.update(1)

print('LPIPS mean:', np.mean(metric_list))
print('LPIPS std:', np.std(metric_list))

pytorch_fid args.data_path args.out_dir
