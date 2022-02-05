import torch
import torch.nn as nn


class Spectral_transform_block(nn.Module):
    def __init__(self, channels_in, channels_out, channels_hidden, kernel_size):
        super(Spectral_transform_block, self).__init__()

        self.channels_hidden = channels_hidden

        self.conv1 = nn.Conv2d(channels_in, channels_hidden, kernel_size=kernel_size, padding='same', bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels_hidden)

        #fourier unit
        self.conv2 = nn.Conv2d(2*channels_hidden, 2*channels_hidden, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(2*channels_hidden)

        self.conv1x1 = nn.Conv2d(channels_hidden, channels_out, kernel_size=1, bias=True)

    def forward(self, x):
        x_residual = self.relu(self.bn1(self.conv1(x)))

        #fourier unit
        batch = x_residual.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfft2(x_residual, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv2(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn2(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output_FU = torch.fft.irfft2(ffted,norm="ortho")

        #x = torch.fft.rfft2(x_residual,norm="ortho") #ortho norma added
        #x = torch.concat((x.real, x.imag), dim=1)
        #x = torch.stack((x.real, x.imag), dim=-1) #changed concat to stack
        #x = self.relu(self.bn2(self.conv2(x)))
        #x = torch.complex(x[:, :self.channels_hidden], x[:, self.channels_hidden:])
        #x = torch.fft.irfft2(x)

        x = output_FU + x_residual
        x = self.conv1x1(x) #checked
        return x

class FFC_block(nn.Module):
    def __init__(self, n_channels, global_percent, experiment):
        super(FFC_block, self).__init__()
        #I added the possibility to have different out_channels

        in_channels_global = round(n_channels*global_percent)
        in_channels_local = n_channels - in_channels_global

        out_channels_global = in_channels_global #TBD
        out_channels_local = in_channels_local #TBD

        #definition of layers
        self.conv_ll = nn.Conv2d(in_channels_local, out_channels_local, kernel_size=3, padding='same', bias=True)
        if experiment == 'conv_change':
            self.conv_lg = Spectral_transform_block(in_channels_local, out_channels_global, out_channels_global//2, 1)
        else:
            self.conv_lg = nn.Conv2d(in_channels_local, out_channels_global, kernel_size=3, padding='same', bias=True)
        self.conv_gl = nn.Conv2d(in_channels_global, out_channels_local, kernel_size=3, padding='same', bias=True)

        ## changing hidden dimension size channels_global//2 like in original implementation
        # kernel size is 1 for all spectral transform block
        self.conv_gg = Spectral_transform_block(in_channels_global, out_channels_global, out_channels_global//2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn_l = nn.BatchNorm2d(out_channels_local)
        self.bn_g = nn.BatchNorm2d(out_channels_global)

        self.channels_local = in_channels_local

    def forward(self, x):


        channels_local = self.channels_local
        x_l = x[:, :channels_local]
        x_g = x[:, channels_local:]

        ##FFC block
        x_ll = self.conv_ll(x_l)
        x_lg = self.conv_lg(x_l)
        x_gl = self.conv_gl(x_g)
        x_gg = self.conv_gg(x_g)

        x_l = x_ll + x_gl
        x_g = x_gg + x_lg


        ## FFC_BN_ACT block
        x_l = self.relu(self.bn_l(x_l))
        x_g = self.relu(self.bn_g(x_g))

        ###

        x = torch.concat((x_l, x_g), dim=1)
        return x

class FFC_conv_residual_block(nn.Module):
    def __init__(self, n_channels, global_percent, experiment):
        super(FFC_conv_residual_block, self).__init__()
        #definition of layers
        self.conv = FFC_block(n_channels, global_percent, experiment)

    def forward(self, x):

        x_ffc = self.conv(x)
        x_ffc = self.conv(x_ffc)
        x = x + x_ffc

        return x


class Lama(nn.Module):
    def __init__(self, experiment='baseline', channels_in=4, channels_out=3, down_steps=3, up_steps=3, base_mult=64, n_ffc_residual=9, global_percent=0.6):
        super(Lama, self).__init__()

        down = [nn.ReflectionPad2d(3),
                nn.Conv2d(channels_in, base_mult, kernel_size=7, bias=True), #not good
                nn.BatchNorm2d(base_mult),
                nn.ReLU(inplace=True)]
        for idx in range(down_steps):
            n_channels = base_mult * 2**idx
            down.append(nn.Conv2d(n_channels, n_channels*2, kernel_size=3, stride=2, padding=1, bias=True))
            down.append(nn.BatchNorm2d(n_channels*2))
            down.append(nn.ReLU(inplace=True))
        self.down = nn.Sequential(*down)

        ffcs = []
        for idx in range(n_ffc_residual):
            ffcs.append(FFC_conv_residual_block(n_channels*2, global_percent, experiment))
        self.ffcs = nn.Sequential(*ffcs)

        up = []
        for idx in range(up_steps):
            n_channels = base_mult * 2**(up_steps - idx)
            up.append(nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            up.append(nn.BatchNorm2d(n_channels//2))
            up.append(nn.ReLU(inplace=True))

        up.append(nn.ReflectionPad2d(3))
        up.append(nn.Conv2d(base_mult, channels_out, kernel_size=7, bias=True))
        up.append(nn.Sigmoid())
        self.up = nn.Sequential(*up)

    def forward(self, x):
        x = self.down(x)
        x = self.ffcs(x)
        x = self.up(x) #Upsampling is good


        return x
