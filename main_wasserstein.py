import torch

from loss_wasserstein import Loss_Wasserstein
from model import LAMA
from train_wasserstein import train_loop_Wassertstein


alpha = 30 # for high receptive field perceptual loss
beta = 100 # for feature matching loss
LAMBDA_GP = 10
lr = 0.00005 # Learning rate

LAMA_loss = Loss_Wasserstein(alpha,beta,LAMBDA_GP,path)

LAMA_Model = Lama()
LAMA_Model = LAMA_Model.to(device)

optimizer_g = torch.optim.RMSprop(LAMA_Model.parameters(), lr=lr)
optimizer_d = torch.optim.RMSprop(LAMA_loss.disc.parameters(), lr=lr)

train_loop_Wassertstein(LAMA_Model,optimizer_g,optimizer_d,test_dataloader,test_dataloader,sav_rate=1000,disc_iter = 5,clip_weights = False)

