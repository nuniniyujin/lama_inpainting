import torch
import torch.nn as nn
import torch.nn.functional as F

from dilated_resnet import resnet50, ResnetDilated


from torch.nn.modules import distance
class Discriminator(nn.Module):
    def __init__(self, channels_in=3, base_mult=64, n_layers=3):
        super().__init__()
        disc = [nn.Conv2d(channels_in, base_mult, kernel_size=4, stride=2, padding=2),
                 nn.LeakyReLU(0.2, True)]

        stride = 2
        for idx in range(n_layers):
            if idx == n_layers-1:
                stride = 1

            n_channels = base_mult * 2**idx
            disc += [nn.Conv2d(n_channels, n_channels*2, kernel_size=4, stride=stride, padding=2),
                      nn.BatchNorm2d(n_channels*2),
                      nn.LeakyReLU(0.2, True)]

        disc += [nn.Conv2d(n_channels*2, 1, kernel_size=4, stride=1, padding=2)]

        self.disc = nn.Sequential(*disc)

    def forward(self, img):
        return self.disc(img)


class Loss(nn.Module):
    def __init__(self, kappa, alpha, beta, gamma):
        super(Loss, self).__init__()
        self.init_adversarial()
        self.init_perceptual()
        self.init_hrf_perceptual()

        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, img, target_img, mask, net_type):
        target_img = target_img/target_img.max()
        if net_type == 'discriminator':
            adversarial, r1 = self.adversarial(img, target_img, mask, net_type=net_type)
            return self.kappa*adversarial + self.gamma*r1

        adv_loss = self.kappa*self.adversarial(img, target_img, mask, net_type=net_type)[0]
        hrfp = self.alpha*self.hrf_perceptual(img, target_img)
        perceptual_loss = self.beta*self.perceptual(img, target_img,mask)
        return adv_loss,hrfp,perceptual_loss,adv_loss+hrfp+perceptual_loss

        #return self.kappa*self.adversarial(img, target_img, mask, net_type=net_type)[0] +\
        #       self.alpha*self.hrf_perceptual(img, target_img) +\
        #       self.beta*self.perceptual(img, target_img,mask)
               #self.beta*self.perceptual(img, target_img) # i changed according to the modification i made

    def init_adversarial(self):
        self.disc = Discriminator().to(device) ### i added to device in order to work on the same device (LAMA and Backbones for loss calculation)

    def adversarial(self, img, target_img, mask, net_type):
        if net_type == 'discriminator':
            target_img.requires_grad = True

            img = img.detach()
            img_disc_pred = self.disc(img)
            img_loss = F.softplus(img_disc_pred) #Why softplus???

            target_img_disc_pred = self.disc(target_img)
            target_img_loss = F.softplus(-target_img_disc_pred)
        elif net_type == 'generator':
            img_disc_pred = self.disc(img)
            img_loss = F.softplus(-img_disc_pred)
        else:
            raise ValueError('Unknown net_type')

        small_mask = F.interpolate(mask, size=img_loss.shape[-2:], mode='bilinear', align_corners=False)
        img_loss = img_loss*small_mask # I only try to penalize the regions outside the mask???

        if net_type == 'generator':
            return img_loss.mean(), None

        #net_type == 'discriminator'
        img_loss += (1-small_mask)*F.softplus(-img_disc_pred)
        adv_loss = (img_loss + target_img_loss).mean()

        r1_loss = self.r1(target_img, target_img_disc_pred)
        target_img.requires_grad = False

        return adv_loss, r1_loss

    def r1(self, target_img, target_img_disc_pred):
        grad = torch.autograd.grad(target_img_disc_pred.sum(), target_img, create_graph=True)[0]
        return (torch.square(grad.reshape((grad.shape[0], -1)).norm(2, dim=1))).mean()

    def init_hrf_perceptual(self):
        orig_resnet = resnet50(pretrained=True)
        self.net_encoder = ResnetDilated(orig_resnet, dilate_scale=8).to(device)### i added to device in order to work on the same device (LAMA and Backbones for loss calculation)

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
        self.vgg_layers = torchvision.models.vgg19(pretrained=True).features.to(device) ### i added to device in order to work on the same device (LAMA and Backbones for loss calculation)

        for param in self.vgg_layers.parameters():
            param.require_grad = False

    #def perceptual(self, img, target_img):
    def perceptual(self, img, target_img,mask): #i changed to use mask variable

        loss = torch.zeros(img.shape[0]).to(device) ### i added to device in order to work on the same device
        cnt = 0
        for idx, module in self.vgg_layers._modules.items():
            img = module(img)
            target_img = module(target_img)

            if module.__class__.__name__ == 'ReLU':
                part_loss = F.mse_loss(img, target_img, reduction='none')

                #Does it make sense to use the mask in the perceptual loss?
                #small_mask = F.interpolate(mask, size=img.shape[-2:], mode='bilinear', align_corners=False)
                #part_loss = part_loss * (1 - small_mask)

                # Watch out for overflow in the loss variable since division is made in the end
                loss += part_loss.mean(dim=tuple(range(4)[1:]))
                cnt += 1

        return (loss/cnt).sum()
