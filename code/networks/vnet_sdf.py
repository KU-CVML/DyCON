import torch
from torch import nn
import torch.nn.functional as F

"""
Differences with V-Net
Adding nn.Tanh in the end of the conv. to make the outputs in [-1, 1].
"""

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        # transform the input
        self.convblock_skip = ConvBlock(1, n_filters_in=n_filters_in, n_filters_out=n_filters_out, normalization=normalization)

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print("skip_out: ", self.convblock_skip(x).shape) #: torch.Size([4, 16, 112, 112, 80])
        # print("res_conv: ", self.conv(x).shape) #: torch.Size([4, 16, 112, 112, 80])
        x = (self.conv(x) + self.convblock_skip(x))
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet_SDF(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super(VNet_SDF, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)
        # print('x1: {}, \nx2: {}, \nx3: {}, \nx4: {}, \nx5: {}'.format(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape))
        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up) #  torch.Size([4, 16, 112, 112, 80])
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9) # convert the 16 channel conv output to num_classes output torch.Size([4, 2, 112, 112, 80])
        out_tanh = self.tanh(out)
        out_seg = self.out_conv2(x9)
        return out_tanh, out_seg


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_tanh, out_seg = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_tanh, out_seg

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)

"""
x1: torch.Size([4, 16, 112, 112, 80]),
x2: torch.Size([4, 32, 56, 56, 40]),
x3: torch.Size([4, 64, 28, 28, 20]),
x4: torch.Size([4, 128, 14, 14, 10]),
x5: torch.Size([4, 256, 7, 7, 5])
tanh:  torch.Size([4, 2, 112, 112, 80])
seg:  torch.Size([4, 2, 112, 112, 80])
"""

# if __name__ == '__main__':
#     # compute FLOPS & PARAMETERS
#     from thop import profile
#     from thop import clever_format
#     model = VNet(n_channels=1, n_classes=2)
#     input = torch.randn(4, 1, 112, 112, 80)
#     flops, params = profile(model, inputs=(input,))
#     macs, params = clever_format([flops, params], "%.3f")
#     print(macs, params)
#     print("VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

#     # import ipdb; ipdb.set_trace()
    
if __name__ == '__main__':
    print('*'*100)
    input = torch.randn(4, 4, 112, 112, 80)
    # conv_m = ConvBlock(n_stages=1, n_filters_in=1, n_filters_out=16, normalization='batchnorm')
    # print('conv_m: ', conv_m, "\n")
    # print('*'*100)
    # resconv_m = ResidualConvBlock(n_stages=2, n_filters_in=1, n_filters_out=16, normalization='batchnorm')
    # print('resconv_m: ', resconv_m)
    # print('*'*100)
    # down_conv = DownsamplingConvBlock(n_filters_in=16, n_filters_out=2*16, normalization='batchnorm')
    # print('\ndown_conv: ', down_conv)
    # print('*'*100)
    # # out_conv = conv_m(input) # torch.Size([4, 16, 112, 112, 80])
    # # out_res = resconv_m(input)
    # # print('out_conv: ', out_conv.shape)
    # # print('out_res: ', out_res.shape)
    # down_out = down_conv(resconv_m(input))
    # print('down_out: ', down_out.shape) # torch.Size([4, 32, 56, 56, 40])

    model = VNet_SDF(n_channels=4, n_classes=2, normalization='batchnorm', has_residual=False)
    # e_f = model.encoder(input)
    # d_f = model.decoder(e_f)
    # print('tanh: ', d_f[0].shape, '\nseg: ', d_f[1].shape)
    # out = model(input)
    # print('output: ', out[0].shape, '\t seg: ', out[1].shape)
    # from torchvision import transforms as T
    # # img = sample_dic['image'][0][:, :, :, 0]
    # img = d_f[0][0][0, :, :, 0]
    # img = T.ToPILImage()(img)
    # img.save('tanh_sample.jpg')
    # seg = d_f[1][0][0, :, :, 0]
    # seg = T.ToPILImage()(seg)
    # seg.save('seg_sample.jpg')
    out_tanh, out_seg = model(input)
    print('out_tanh: ', out_tanh.shape, "\nout_seg: ", out_seg.shape)
    print('*'*100)
