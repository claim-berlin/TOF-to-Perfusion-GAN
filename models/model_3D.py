import torch
import functools
import torch.nn as nn

import config as c


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Discriminator_3D(nn.Module):

    def __init__(self, input_nc=2, bn=c.bn, dropout_rate=c.dropout_rate):
        super(Discriminator_3D, self).__init__()

        if c.WGAN:
            self.bn = False
        else:
            self.bn = bn
        self.dropout_rate = dropout_rate

        self.ch = c.ndf
        self.pool = 2
        self.s = 1

        if c.kernel_size == 3:
            self.k = 3
            self.padding = 1
        elif c.kernel_size == 5:
            self.k = 5
            self.padding = 2

        def discr_block(in_ch, out_ch, k=self.k, p=self.pool, s=self.s,
                        pad=self.padding):
            if self.bn:
                block1 = [nn.Conv3d(in_channels=in_ch, out_channels=out_ch,
                                    kernel_size=k, stride=s, padding=pad),
                          #nn.BatchNorm2d(out_ch),
                          nn.LeakyReLU(c.slope_relu, inplace=True),
                          nn.BatchNorm3d(out_ch)]
                block2 = [nn.Conv3d(in_channels=out_ch, out_channels=out_ch,
                                    kernel_size=k, stride=p, padding=pad),
                          #nn.BatchNorm2d(out_ch),
                          nn.LeakyReLU(c.slope_relu, inplace=True),
                          nn.BatchNorm3d(out_ch)]
            else:
                block1 = [nn.Conv3d(in_channels=in_ch, out_channels=out_ch,
                                    kernel_size=k, stride=s, padding=pad),
                          nn.InstanceNorm3d(out_ch),
                          nn.LeakyReLU(c.slope_relu, inplace=True),
                          nn.Dropout3d(self.dropout_rate)]
                block2 = [nn.Conv3d(in_channels=out_ch, out_channels=out_ch,
                                    kernel_size=k, stride=p, padding=pad),
                          nn.InstanceNorm3d(out_ch),
                          nn.LeakyReLU(c.slope_relu, inplace=True),
                          nn.Dropout3d(self.dropout_rate)]
            return block1, block2

        bl1, bl2 = discr_block(input_nc, self.ch)
        self.layer1 = nn.Sequential(*bl1)
        self.layer2 = nn.Sequential(*bl2)
        bl3, bl4 = discr_block(self.ch, 2 * self.ch)
        self.layer3 = nn.Sequential(*bl3)
        self.layer4 = nn.Sequential(*bl4)
        bl5, bl6 = discr_block(2 * self.ch, 4 * self.ch)
        self.layer5 = nn.Sequential(*bl5)
        self.layer6 = nn.Sequential(*bl6)
        bl7, bl8 = discr_block(4 * self.ch, 8 * self.ch)
        self.layer7 = nn.Sequential(*bl7)
        self.layer8 = nn.Sequential(*bl8)

        if c.WGAN:
            self.adv_layer = nn.Sequential(
                nn.Conv3d(in_channels=8 * self.ch, out_channels=1,
                          kernel_size=self.k, stride=self.s,
                          padding=self.padding))
        else:
            self.adv_layer = nn.Sequential(nn.Conv3d(in_channels=8 * self.ch,
                                                        out_channels=1,
                                                        kernel_size=8,
                                                        padding=0),
                                            nn.Sigmoid())


    def forward(self, img_pairs):
        x = self.layer1(img_pairs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x_lastconv = self.layer8(x)  # shape: (b,128,6,6)

        # final layer
        if c.WGAN:
            #print(x_lastconv.shape) 
            x_out = self.adv_layer(x_lastconv)
        else:
            #x_out = x_lastconv.view(x_lastconv.shape[0], -1)
            x_out = self.adv_layer(x_lastconv)
            x_out = torch.reshape(x_out, (x_out.shape[0], 1))

        return x_out, x_lastconv


class PatchDiscriminator_3D(nn.Module):
    
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=2, ndf=64, n_layers=c.nr_layer_d, norm_layer=c.norm_d_3d):
        """Construct a PatchGAN discriminator
        Parameters:
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator_3D, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map - might need to add sigmoid
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        x = input
        modulelist = list(self.model)
        for l in modulelist[:6]:
            x = l(x)
        last_layer = x
        for l in modulelist[6:]:
            x = l(x)
        return x, last_layer

class UnetGenerator_3D(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=80, output_nc=1, num_downs=c.nr_layer_g, ngf=c.ngf, norm_layer=nn.BatchNorm3d, use_dropout=True, ups = c.upsampling):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator_3D, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock_3D(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, ups=ups)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock_3D(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, ups=ups)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock_3D(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, ups=ups)
        unet_block = UnetSkipConnectionBlock_3D(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, ups=ups)
        unet_block = UnetSkipConnectionBlock_3D(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, ups=ups)
        self.model = UnetSkipConnectionBlock_3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, ups=ups)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

paddingTest = 1#(1,1,1)
class UnetSkipConnectionBlock_3D(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=True, ups=c.upsampling):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_3D, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=paddingTest, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if ups:
                upconv = UpsampleConvLayer_3D(inner_nc * 2, outer_nc, 
                                           kernel_size=3, stride=1, 
                                           upsample=2)      
            else:
                upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=paddingTest)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if ups:
                upconv = UpsampleConvLayer_3D(inner_nc, outer_nc, 
                                           kernel_size=3, stride=1, 
                                           upsample=2)
            else:
                upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=paddingTest, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if ups:
                upconv = UpsampleConvLayer_3D(inner_nc * 2, outer_nc,
                                           kernel_size=3, stride=1, 
                                           upsample=2)
            else:
                upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=paddingTest, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(c.dropout_rate)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UpsampleConvLayer_3D(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
              super(UpsampleConvLayer, self).__init__()
              self.upsample = upsample
              if upsample:
                  self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
              reflection_padding = kernel_size // 2
              self.reflection_pad = torch.nn.ReflectionPad3d(reflection_padding)
              self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride)

          def forward(self, x):
              x_in = x
              if self.upsample:
                  x_in = self.upsample_layer(x_in)
              out = self.reflection_pad(x_in)
              out = self.conv3d(out)
              return out
