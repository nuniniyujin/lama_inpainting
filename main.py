import torch
from torch.utils.data import DataLoader

from loss import Loss
from model import Lama
from train import train_loop
from data import MyDataset

kappa = 10 # for adversarial loss
alpha = 30 # for high receptive field perceptual loss
beta = 100 # for feature matching loss
gamma = 0.001#for R1 loss
#hyper parameters written in the paper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAMA_loss = Loss(kappa,alpha,beta,gamma,device)

LAMA_Model = Lama()
LAMA_Model = LAMA_Model.to(device)
optimizer_g= torch.optim.Adam(LAMA_Model.parameters())

optimizer_d = torch.optim.Adam(LAMA_loss.disc.parameters()) #TBC

#LAMA_Model.load_state_dict(torch.load('/content/drive/MyDrive/ENS/DL/Inpainting_project/model_weights_conv_change_new_mask/model10epochs1200batch05-01-2022 16h48.pth'))

path= '/content/drive/MyDrive/ENS/DL/Inpainting_project/val_256'
train_dataloader = DataLoader(MyDataset(path), batch_size=8)

train_loop(LAMA_Model,optimizer_g,optimizer_d,test_dataloader,train_dataloader,LAMA_loss,device)
