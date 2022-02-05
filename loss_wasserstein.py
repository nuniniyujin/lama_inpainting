import torch
import torch.nn as nn
import torch.nn.functional as F
from dilated_resnet import resnet50, ResnetDilated

from torch.nn.modules import distance

class Discriminator_Critic(nn.Module):
    def __init__(self, channels_in=3, base_mult=64, n_layers=3):
        super().__init__()
        disc = [nn.Conv2d(channels_in, base_mult, kernel_size=4, stride=2, padding=2,bias=False),
                 nn.LeakyReLU(0.2, True)]
        
        stride = 2
        for idx in range(n_layers):
            if idx == n_layers-1:
                stride = 1

            n_channels = base_mult * 2**idx
            disc += [nn.Conv2d(n_channels, n_channels*2, kernel_size=4, stride=stride, padding=2,bias=False),
                      nn.InstanceNorm2d(n_channels*2, affine=True),
                      nn.LeakyReLU(0.2, True)]
        
        disc += [nn.Conv2d(n_channels*2, 1, kernel_size=4, stride=1, padding=2,bias=False)]

        self.disc = nn.Sequential(*disc)
    
    def forward(self, img):        
        return self.disc(img)
        

class Loss_Wasserstein(nn.Module):
    def __init__(self, alpha, beta,lmbda,Path_d_weights,load=False,device):
        '''
        input:
        -- kappa : coefficient for adversarial loss
        -- alpha : coefficient high receptive field perceptual loss
        -- beta : coefficient for feature matching loss
        -- gamma : coefficient for R1 loss
        -- lmbda : coefficient for gradient penalty for discriminnator
        -- Path_d_weights : path + .pth name to load pretrained weights
        -- load (default: False): option to load pretrain weights 
        '''
        super(Loss_Wasserstein, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
        self.device = device

        self.Path_d_weights = Path_d_weights
        self.load = load

        self.init_Wasserstein(self.Path_d_weights,self.load) #changed
        self.init_perceptual()
        self.init_hrf_perceptual()

    def forward(self, img, target_img, mask, net_type):
        if net_type == 'discriminator':
            disc_loss = self.Wasserstein(img, target_img, mask, net_type=net_type)
            gp = self.gradient_penalty(img, target_img)
            return disc_loss +  self.lmbda*gp
        
        gen_loss = self.Wasserstein(img, target_img, mask, net_type=net_type)
        hrfp = self.alpha*self.hrf_perceptual(img, target_img)
        perceptual_loss = self.beta*self.perceptual(img, target_img,mask)
        return gen_loss,hrfp,perceptual_loss,gen_loss+hrfp+perceptual_loss

    def init_Wasserstein(self,Path_d_weights,load=False):
        self.disc = Discriminator_Critic().to(self.device) 
        if load:
          self.disc.load_state_dict(torch.load(Path_d_weights))

    def Wasserstein(self, img, target_img, mask, net_type):
        if net_type == 'discriminator':
            target_img.requires_grad = True
            img = img.detach()

            img_disc_pred = self.disc(img).reshape(-1)
            target_img_disc_pred = self.disc(target_img).reshape(-1)

            disc_loss = -(torch.mean(target_img_disc_pred) -torch.mean(img_disc_pred))
            return disc_loss

        elif net_type == 'generator':
            img_disc_pred = self.disc(img).reshape(-1)
            gen_loss = -torch.mean(img_disc_pred)
            return gen_loss
        else:
            raise ValueError('Unknown net_type')

    def gradient_penalty(self, img, target_img):
        batch_size, channel, height, width= img.shape
        #alpha is selected randomly between 0 and 1
        alpha= torch.rand(batch_size,1,1,1).repeat(1, channel, height, width)
        alpha = alpha.to(self.device)
        interpolatted_image=(alpha*target_img) + (1-alpha) * img
        
        # calculate the critic score on the interpolated image
        interpolated_score= self.disc(interpolatted_image)
        
        # take the gradient of the score wrt to the interpolated image
        gradient= torch.autograd.grad(inputs=interpolatted_image,
                                      outputs=interpolated_score,
                                      retain_graph=True,
                                      create_graph=True,
                                      grad_outputs=torch.ones_like(interpolated_score)                          
                                    )[0]
        gradient= gradient.view(gradient.shape[0],-1)
        gradient_norm= gradient.norm(2,dim=1)
        gradient_penalty=torch.mean((gradient_norm-1)**2)
        return gradient_penalty

    def init_hrf_perceptual(self):
        orig_resnet = resnet50(pretrained=True) 
        self.net_encoder = ResnetDilated(orig_resnet, dilate_scale=8).to(self.device)
        
        for param in self.net_encoder.parameters():
            param.require_grad = False

    def hrf_perceptual(self, img, target_img):
        img = self.net_encoder(img, return_feature_maps=True)
        target_img = self.net_encoder(target_img, return_feature_maps=True)

        loss = 0
        for x, y in zip(img, target_img):
            loss += F.mse_loss(x, y)
        return loss

    def init_perceptual(self):
        self.vgg_layers = torchvision.models.vgg19(pretrained=True).features.to(self.device)
        for param in self.vgg_layers.parameters():
            param.require_grad = False
    
    def perceptual(self, img, target_img,mask):

        loss = torch.zeros(img.shape[0]).to(self.device)
        cnt = 0
        for idx, module in self.vgg_layers._modules.items():
            img = module(img)
            target_img = module(target_img)

            if module.__class__.__name__ == 'ReLU':
                part_loss = F.mse_loss(img, target_img, reduction='none')
                loss += part_loss.mean(dim=tuple(range(4)[1:]))
                cnt += 1
        
        return (loss/cnt).sum()
