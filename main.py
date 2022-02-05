import torch

from loss import Loss
from model import LAMA
from train import train_loop


kappa = 10 # for adversarial loss
alpha = 30 # for high receptive field perceptual loss
beta = 100 # for feature matching loss
gamma = 0.001#for R1 loss
#hyper parameters written in the paper


LAMA_loss = Loss(kappa,alpha,beta,gamma)

LAMA_Model = Lama()
LAMA_Model = LAMA_Model.to(device)
optimizer_g= torch.optim.Adam(LAMA_Model.parameters())

optimizer_d = torch.optim.Adam(LAMA_loss.disc.parameters()) #TBC

LAMA_Model.load_state_dict(torch.load('/content/drive/MyDrive/ENS/DL/Inpainting_project/model_weights_conv_change_new_mask/model10epochs1200batch05-01-2022 16h48.pth'))

train_loop(LAMA_Model,optimizer_g,optimizer_d,test_dataloader,test_dataloader,start_epoch=9)
