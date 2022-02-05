import argparse

import torch
from torch.utils.data import DataLoader

from loss import Loss
from model import Lama
from train import train_loop
from data import MyDataset
from loss_wasserstein import Loss_Wasserstein
from train_wasserstein import train_loop_Wassertstein

parser = argparse.ArgumentParser(description='LaMa model train main file')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path to train data folder')

# Loss arguments (default values are the same as in the paper)
parser.add_argument('--kappa', type=int, default=10,
                    help='Kappa value for adversarial loss')
parser.add_argument('--alpha', type=int, default=30,
                    help='Alpha value for high receptive field perceptual loss')
parser.add_argument('--beta', type=int, default=100,
                    help='Beta value for feature matching loss')
parser.add_argument('--gamma', type=float, default=0.001,
                    help='Gamma value for gradient penalization')
parser.add_argument('--lambda_gp', type=int, default=10,
                    help='Lambda GP for Wassertstein loss')
parser.add_argument('--wasserstein_weights_path', type=str, default=None,
                    help='Path for wasserstein weights')

# Train arguments
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='Number of epochs')
parser.add_argument('--model_base_dir', type=str, default='./model_weights/',
                    help='Directory to save model weights')
parser.add_argument('--image_base_dir', type=str, default='./image/',
                    help='Directory to save images from training')

# Continue training from checkpoint
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to the model checkpoint to continue training')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Start from this epoch (use when the train is resumed from a checkpoint)')

# Choose experiment
parser.add_argument('--experiment', type=str, default='baseline', choices=['baseline', 'conv_change', 'wgan'],
                    help='Choose between experiments [baseline, conv_change, wgan]')


args = parser.parse_args()

kappa = args.kappa # for adversarial loss
alpha = args.alpha # for high receptive field perceptual loss
beta = args.beta # for feature matching loss
gamma = args.gamma #for R1 loss
lambda_gp = args.lambda_gp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.experiment == 'wgan':
    LAMA_loss = Loss_Wasserstein(alpha,beta,lambda_gp,args.wasserstein_weights_path,device)
else:
    LAMA_loss = Loss(kappa,alpha,beta,gamma,device)

LAMA_Model = Lama(experiment=args.experiment)
LAMA_Model = LAMA_Model.to(device)
optimizer_g= torch.optim.Adam(LAMA_Model.parameters(), lr=args.lr)

optimizer_d = torch.optim.Adam(LAMA_loss.disc.parameters(), lr=args.lr)

if args.checkpoint_path is not None:
    LAMA_Model.load_state_dict(torch.load(args.checkpoint_path))

train_dataloader = DataLoader(MyDataset(args.data_path), batch_size=args.batch_size)

if args.experiment == 'wgan':    
    train_loop_Wassertstein(LAMA_Model, optimizer_g, optimizer_d,
                            train_dataloader, train_dataloader, LAMA_loss,
                            device, args.model_base_dir, args.image_base_dir,
                            sav_rate=1000, disc_iter=5, clip_weights=False)
else:
    train_loop(LAMA_Model, optimizer_g, optimizer_d, train_dataloader,
               train_dataloader, LAMA_loss, device, args.model_base_dir,
               args.image_base_dir, start_epoch=args.start_epoch,
               epochs=args.n_epochs)
