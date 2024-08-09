import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class DWConv(nn.Module):
    def __init__(self, dim=768, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size, stride=stride, padding=padding, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel=2, stride=2, pad=0, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act2 = act_layer()

    def forward(self, x, H, W, D):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CubeAttention(nn.Module):
    def __init__(self, in_channel, out_channel, size, kernel=2, stride=2, pad=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.groups = 2
        self.size = size
        self.InChannel = in_channel
        self.OutChannel = out_channel
        # print("size: ", size)
        self.TransverseConv = nn.Conv3d(self.groups, (size[1])*self.groups, (size[0], 3, 3), (1, 1, 1),
                                        padding=(0, 1, 1))
        self.CoronalConv = nn.Conv3d(self.groups, (size[1])*self.groups, (3, 3, size[2]), (1, 1, 1),
                                     padding=(1, 1, 0))
        self.map = nn.Conv3d(in_channel*2, in_channel, 3, 1, 1)
        self.mpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(in_channel)

    def forward(self, x):
        H, W, D = self.size
        B, C, H, W, D= x.shape
        shortcut = x
        x_norm = self.norm(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, C, H, W, D)
        # x = x.permute(0, 2, 1)
        x = x_norm.reshape(B * C//self.groups, self.groups, H, W, D)
        tans_feature = self.TransverseConv(x).view(B, C, self.size[1], self.size[1], self.size[2])
        cor_feature = self.CoronalConv(x).view(B, C, self.size[1], self.size[0], self.size[1])


        t1 = tans_feature.permute(0, 1, 3, 2, 4)
        c1 = cor_feature.permute(0, 1, 4, 3, 2)
        f = (c1@t1).permute(0, 1, 3, 2, 4)


        fusion_feature = (cor_feature @ tans_feature)
        attn = self.attn_drop(fusion_feature.permute(0, 1, 3, 2, 4))

        x = self.proj_drop(self.act(self.map(torch.cat((shortcut*f, attn), dim=1)))) + shortcut
        return x

class CubeAttentionDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, size, kernel=2, stride=2, pad=0, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.size = size
        self.InChannel = in_channel
        self.OutChannel = out_channel
        # print("size: ", size)
        self.TransverseConv = nn.Conv3d(1, size[1]*2, (size[0], kernel, kernel), (stride, stride, stride),padding=(0, pad, pad))
        self.CoronalConv = nn.Conv3d(1, size[1]*2, (kernel, kernel, size[2]), (stride, stride, stride),padding=(pad, pad, 0))
        self.mapT = nn.Linear(size[1]*2, size[1])
        self.mapC = nn.Linear(size[1]*2, size[1])
        self.mpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(in_channel)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Conv3d(in_channel, out_channel, 1)
        self.act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

        self.convd=nn.Conv3d(in_channel, out_channel, kernel, stride=stride, padding=pad)


    def forward(self, x):
        # print("here")
        H, W, D = self.size
        B, C, H, W, D = x.shape
        shutcut = x
        x = self.norm(x.reshape(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(B*C, 1, H, W, D)
        # sag_feature = self.SagittalConv(x)
        tans_feature = self.TransverseConv(x).view(B, C, self.size[1]*2, self.size[1], self.size[2]).permute(0, 1, 3, 4, 2)
        cor_feature = self.CoronalConv(x).view(B, C, self.size[1]*2, self.size[0], self.size[1]).permute(0, 1, 3, 4, 2)
        tans_feature = self.mapT(tans_feature).permute(0, 1, 4, 2, 3)
        cor_feature = self.mapC(cor_feature).permute(0, 1, 4, 2, 3)
        tans_feature_pool = self.mpool(tans_feature)
        cor_feature_pool = self.mpool(cor_feature)
        fusion_feature = (cor_feature_pool @ tans_feature_pool).permute(0, 1, 3, 2, 4).reshape(B, C, -1)
        attn = self.softmax(fusion_feature)
        attn = self.attn_drop(attn)
        attn = self.upsample(attn.reshape(B, C, H // 2, W // 2, D // 2))
        x = shutcut + attn
        x = self.proj_drop(self.act(self.convd(x)))
        return x

class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        size = [160, 192, 160]
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn,
                                            size=[size[0]//(2**i), size[1]//(2**i), size[2]//(2**i)]))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn, size=[10, 12, 10]))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn, size=[20, 24, 20]))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn, size=[40, 48, 40]))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn, size=[80, 96, 80]))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn, size=[80, 96, 80]))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False,size=None):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if kernel_size==3 and size is not None:
            layer = nn.Sequential(
                # CubeAttentionDecoder(in_channels, out_channels, size, kernel_size, stride, padding),
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2)
            )
        else:
            if size is not None:
                layer = nn.Sequential(
                    CubeAttention(in_channels, out_channels, size, kernel_size, stride, padding),
                    conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                    nn.LeakyReLU(0.2))
            else:
                if batchnorm:
                    layer = nn.Sequential(
                        conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                        bn_fn(out_channels),
                        nn.LeakyReLU(0.2))
                else:
                    layer = nn.Sequential(
                        conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                        nn.LeakyReLU(0.2))
        return layer


    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
